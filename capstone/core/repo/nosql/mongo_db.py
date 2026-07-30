import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from core.config import Mongo_conf
from typing import Dict, List, Any

"""
student document structure in 'students' collection:
{
    "_id": student_id,
    "state": {
        "finished_communities": [],
        "current_pos": None,
        "active_resource": None,
        "mastery_map": {},
        "previous_nodes": [],
        "upcoming_nodes": [],
        "summary": str,
        "pending_proposal": None
    },
    "memo": [
        {
            "id": "session_id",
            "name": "Session Name",
            "chats": [
                {
                    "id": "chat_id",
                    "invoke": "student query",
                    "messages": [
                        {
                            "role": "student" | (Agent name) |"TA",
                            "heading": str,
                            "message": str,
                            "timestamp": str
                        }
                    ]
                }
            ]
        }
    ]
}
"""


class Mongo_DB:
    def __init__(self, config: Mongo_conf = Mongo_conf()):
        self.client = MongoClient(config.uri)
        self.db = self.client[config.db_name]
        self.students: Collection = self.db['students']
        self.learning_logs: Collection = self.db['learning_logs']
        self.learning_trees: Collection = self.db['learning_trees']
        self.course_trees: Collection = self.db['course_trees']

    def create_student(self, student_id: str):
        doc = self.students.find_one({"_id": student_id})
        if not doc:
            self.students.insert_one({
                "_id": student_id,
                "language": "vn",  # default ui/tutor language
                "state": {
                    "finished_communities": [],
                    "current_pos": None,
                    "active_resource": None,
                    "mastery_map": {},
                    "previous_nodes": [],
                    "upcoming_nodes": [],
                    "summary": "Student just started",
                    "pending_proposal": None
                },
                "memo": []
            })

    def delete_student(self, student_id: str) -> None:
        self.students.delete_one({"_id": student_id})
        self.learning_logs.delete_many({"student_id": student_id})
        self.learning_trees.delete_many({"student_id": student_id})

    def get_language(self, student_id: str) -> str:
        # read saved language, default to vn
        doc = self.students.find_one({"_id": student_id}, {"language": 1})
        return (doc or {}).get("language", "vn")

    def set_language(self, student_id: str, language: str):
        # persist language preference
        self.students.update_one(
            {"_id": student_id},
            {"$set": {"language": language}},
        )

    def get_student_state(self, student_id: str) -> dict:
        doc = self.students.find_one({"_id": student_id}, {"state": 1})
        if not doc:
            self.create_student(student_id)
            doc = self.students.find_one({"_id": student_id}, {"state": 1})
        return doc.get("state", {})

    def update_student_state(self, student_id: str, state_dict: dict):
        self.students.update_one(
            {"_id": student_id},
            {"$set": {"state": state_dict}},
            upsert=True
        )

    def log_event(self, student_id: str, action: str, node_id: str, data: dict = None):
        log_entry = {
            "student_id": student_id,
            "action": action,
            "node_id": node_id,
            "data": data or {},
            "timestamp": datetime.datetime.utcnow()
        }
        self.learning_logs.insert_one(log_entry)

    def create_session(self, student_id: str, session_id: str, session_name: str = "New Session"):
        self.students.update_one(
            {"_id": student_id, "memo.id": {"$ne": session_id}},
            {"$push": {"memo": {"id": session_id, "name": session_name, "chats": []}}}
        )

    def create_chat(self, student_id: str, session_id: str, chat_id: str, invoke_msg: str):
        initial_msg = {
            "role": "student",
            "heading": "Student Query",
            "message": invoke_msg,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.students.update_one(
            {"_id": student_id, "memo.id": session_id},
            {"$push": {"memo.$.chats": {"id": chat_id, "invoke": invoke_msg, "messages": [initial_msg]}}}
        )

    def push_chat_message(self, student_id: str, session_id: str, chat_id: str, entry: Dict[str, Any]):
        entry_with_ts = {**entry, "timestamp": datetime.datetime.utcnow().isoformat()}
        self.students.update_one(
            {"_id": student_id},
            {"$push": {"memo.$[s].chats.$[c].messages": entry_with_ts}},
            array_filters=[{"s.id": session_id}, {"c.id": chat_id}]
        )

    def get_session_data(self, student_id: str, session_id: str) -> dict:
        doc = self.students.find_one({"_id": student_id})
        if not doc or "memo" not in doc:
            return {}
        for session in doc.get("memo", []):
            if session.get("id") == session_id:
                return session
        return {}

    def push_session_tool_result(
        self, student_id: str, session_id: str, chat_id: str, entry: dict
    ) -> None:
        """Atomically append one tool result entry via $push — concurrent-safe."""
        entry_with_ts = {**entry, "timestamp": datetime.datetime.utcnow().isoformat()}
        self.students.update_one(
            {"_id": student_id},
            {"$push": {f"session_context.{session_id}.tool_results.{chat_id}": entry_with_ts}},
            upsert=True,
        )

    def push_session_thought(
        self, student_id: str, session_id: str, chat_id: str, entry: dict
    ) -> None:
        """Atomically append one thought entry via $push — concurrent-safe."""
        entry_with_ts = {**entry, "timestamp": datetime.datetime.utcnow().isoformat()}
        self.students.update_one(
            {"_id": student_id},
            {"$push": {f"session_context.{session_id}.thoughts.{chat_id}": entry_with_ts}},
            upsert=True,
        )

    def get_session_context(self, student_id: str, session_id: str) -> dict:
        """Load persisted SessionContext data (tool_results + thoughts), returns empty dict if none."""
        doc = self.students.find_one(
            {"_id": student_id},
            {f"session_context.{session_id}": 1},
        )
        if not doc:
            return {}
        return doc.get("session_context", {}).get(session_id, {})

    ## learning tree (ADR-0003)

    def get_learning_tree(self, student_id: str, course: str) -> dict:
        doc = self.learning_trees.find_one({"_id": f"{student_id}::{course}"})
        return doc or {}

    def seed_learning_tree(self, student_id: str, course: str, nodes: List[dict]):
        ## re-seed keeps mastery + discovered
        key = f"{student_id}::{course}"
        doc = self.learning_trees.find_one({"_id": key})
        existing = {n["name"]: n for n in (doc or {}).get("nodes", [])}
        merged, order = [], 0
        for n in nodes:
            prev = existing.pop(n["name"], None)
            merged.append({
                "name": n["name"], "type": n.get("type", ""),
                "topic": n.get("topic", ""), "order": order,
                "mastery": (prev or {}).get("mastery", 0),
                "status": (prev or {}).get("status", "pending"),
                "discovered": (prev or {}).get("discovered", False),
            })
            order += 1
        for leftover in existing.values():
            leftover["order"] = order
            merged.append(leftover)
            order += 1
        self.learning_trees.update_one(
            {"_id": key},
            {"$set": {"student_id": student_id, "course": course, "nodes": merged},
             "$setOnInsert": {"attempt_log": [], "active_exercise": None, "last_lecture": ""}},
            upsert=True,
        )

    def add_learning_node(self, student_id: str, course: str, node: dict):
        ## discovery: append if absent
        key = f"{student_id}::{course}"
        doc = self.learning_trees.find_one({"_id": key}) or {"nodes": []}
        names = {n["name"] for n in doc.get("nodes", [])}
        if node["name"] in names:
            return
        entry = {"name": node["name"], "type": node.get("type", ""),
                 "topic": node.get("topic", ""), "order": len(doc.get("nodes", [])),
                 "mastery": 0, "status": "pending", "discovered": True}
        self.learning_trees.update_one(
            {"_id": key}, {"$push": {"nodes": entry}}, upsert=True)

    def update_node_mastery(self, student_id: str, course: str, name: str, mastery: int):
        self.learning_trees.update_one(
            {"_id": f"{student_id}::{course}", "nodes.name": name},
            {"$set": {"nodes.$.mastery": int(mastery),
                      "nodes.$.status": "mastered" if mastery >= 4 else "in_progress"}})

    def append_attempt(self, student_id: str, course: str, record: dict):
        ## DKT log: append-only
        rec = {**record, "timestamp": datetime.datetime.utcnow().isoformat()}
        self.learning_trees.update_one(
            {"_id": f"{student_id}::{course}"},
            {"$push": {"attempt_log": rec}}, upsert=True)

    def set_active_exercise(self, student_id: str, course: str, exercise: dict | None):
        self.learning_trees.update_one(
            {"_id": f"{student_id}::{course}"},
            {"$set": {"active_exercise": exercise}}, upsert=True)

    def set_last_lecture(self, student_id: str, course: str, lecture: str):
        self.learning_trees.update_one(
            {"_id": f"{student_id}::{course}"},
            {"$set": {"last_lecture": lecture[:8096]}}, upsert=True)

    ## course tree cache

    def get_course_tree(self, course: str) -> dict:
        return self.course_trees.find_one({"_id": course}) or {}

    def put_course_tree(self, course: str, tree: dict):
        self.course_trees.update_one(
            {"_id": course},
            {"$set": {"tree": tree, "built_at": datetime.datetime.utcnow().isoformat()}},
            upsert=True)
