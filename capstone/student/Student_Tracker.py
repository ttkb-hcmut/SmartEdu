from collections import deque
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from core.repo.graph.graphdb import GraphDB
from core.repo.sql.sql_db import SQL_DB
from core.repo.nosql.mongo_db import Mongo_DB
from core.schema.wf_state import ConceptNode, StudentState, LearningProposal
from student.memo import *
from student.session_context import SessionContext

NONE_LENGTH = 20


class Page(BaseModel):
    path: str = Field(exclude=True)
    page: int
    concept: List[str] = Field(default_factory=list)
    content: str = ""

    def __str__(self):
        return f"Page {self.page}: {self.content[:50]}... Concepts: {self.concept}"

class StudentSession:
    def __init__(self, student_id: str, student_state: StudentState, memo: Memo, context: SessionContext):
        self.student_id = student_id
        self.student_state = student_state
        self.recent_pages: deque = deque(maxlen=5)
        self.memo = memo
        self.context = context
        
class Student_Tracker:
    """
    Student state, learning progress management.
    Handles multiple concurrent chat sessions per student.
    """

    def __init__(self, graphdb: GraphDB, sqldb: SQL_DB, mongodb: Mongo_DB):
        self.graphdb = graphdb
        self.sqldb = sqldb
        self.mongodb = mongodb

        # Key: session_id -> StudentSession (isolates history and session-context)
        self._sessions: Dict[str, StudentSession] = {}

        # Key: student_id -> dict (shared learning progress across all sessions of this student)
        self._student_states: Dict[str, dict] = {}

        self._session_map: Dict[str, str] = {}   # session_id -> student_id

    ################ Chat Session lifecycle ( student/api.py)

    def create_chat_session(self, student_id: str, session_id: str) -> None:
        """
        Đăng ký một Chat Session UUID mới cho student_id.
        Gọi sau khi student đã login thành công và muốn bắt đầu hội thoại mới.
        """
        self._session_map[session_id] = student_id
        # Pre-warm the student session in memory
        self._get_or_create_student_session(student_id, session_id)

    def get_student_id_by_session(self, session_id: str) -> Optional[str]:
        """Trả về student_id tương ứng với session_id, hoặc None nếu không tồn tại."""
        return self._session_map.get(session_id)

    def drop_session(self, session_id: str) -> None:
        """
        Hủy Chat Session: xóa mapping và xóa state khỏi bộ nhớ.
        Student_id vẫn giữ nguyên trong DB.
        """
        self._session_map.pop(session_id, None)
        self._sessions.pop(session_id, None)

    def delete_student(self, student_id: str) -> None:
        sessions = [sid for sid, owner in self._session_map.items() if owner == student_id]
        for session_id in sessions:
            self.drop_session(session_id)
        self._student_states.pop(student_id, None)

        errors = []
        for repository in (self.graphdb, self.mongodb, self.sqldb):
            try:
                repository.delete_student(student_id)
            except Exception as exc:
                errors.append(f"{type(repository).__name__}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def get_session(self, session_id: str) -> "StudentSession":
        """
        Lấy StudentSession theo Chat Session ID.
        Dùng bởi module TA — không yêu cầu student_id.
        """
        student_id = self._resolve(session_id)
        return self._get_or_create_student_session(student_id, session_id)

    def get_student_state(self, session_id: str) -> StudentState:
        return self.get_session(session_id).student_state

    def get_chat_history(self, session_id: str, mode: str = "full", recent_turns: Optional[int] = None) -> str:
        return self.get_session(session_id).memo.get_formatted_history(mode=mode, recent_turns=recent_turns)

    def get_mastery(self, session_id: str, node_name: str) -> int:
        student_id = self._resolve(session_id)
        return self.graphdb.get_mastery(student_id, node_name)

    def save_state(self, session_id: str) -> None:
        """Lưu toàn bộ state vào DB. Dùng update_learning_position để update từng phần."""
        student_id = self._resolve(session_id)
        state = self._student_states.get(student_id)
        if state:
            self.mongodb.update_student_state(student_id, state)

    def get_road_map(self, session_id: str) -> Dict:
        student_id = self._resolve(session_id)
        return self.graphdb.get_learning_graph(student_id)

    def update_learning_position(
        self, session_id: str, new_node: ConceptNode, bridge_nodes: list = None
    ):
        student_id = self._resolve(session_id)
        session = self.get_session(session_id)
        state = session.student_state

        if bridge_nodes:
            valid_bridge = bridge_nodes[:3]
            current_upcoming = state.get("upcoming_nodes", [])
            new_upcoming = valid_bridge + current_upcoming
            state["upcoming_nodes"] = new_upcoming
            self.mongodb.update_student_field(student_id, "state.upcoming_nodes", new_upcoming)

        current_pos = state.get("current_pos")

        if current_pos and hasattr(current_pos, 'name') and current_pos.name == new_node.name:
            return

        if current_pos is not None:
            prev = state.get("previous_nodes", [])
            prev.insert(0, current_pos)
            trimmed_prev = prev[:NONE_LENGTH]
            state["previous_nodes"] = trimmed_prev
            self.mongodb.update_student_field(student_id, "state.previous_nodes", trimmed_prev)

        self.graphdb.update_learn(student_id, current_pos, new_node)
        state["current_pos"] = new_node
        self.mongodb.update_student_field(
            student_id, 
            "state.current_pos", 
            new_node.model_dump() if hasattr(new_node, 'model_dump') else new_node
        )
        
        self._on_node_change(session)

    def current_resource(self, session_id: str, n: int = 3) -> List[Page]:
        return list(self.get_session(session_id).recent_pages)[-n:]

    def add_recent_page(self, session_id: str, page: Page) -> None:
        self.get_session(session_id).recent_pages.append(page)

    def add_finished_community(self, session_id: str, node: ConceptNode):
        student_id = self._resolve(session_id)
        state = self.get_student_state(session_id)
        if "finished_communities" not in state:
            state["finished_communities"] = []
        
        # In-memory update
        if node.name not in [n.name for n in state["finished_communities"]]:
            state["finished_communities"].append(node)
            # Atomic DB update
            self.mongodb.students.update_one(
                {"_id": student_id},
                {"$addToSet": {"state.finished_communities": node.model_dump() if hasattr(node, 'model_dump') else node}}
            )

    def apply_proposal(self, session_id: str, proposal_dict: Dict[str, Any]) -> None:
        proposal = LearningProposal(**proposal_dict)
        session = self.get_session(session_id)
        student_id = session.student_id
        state = session.student_state

        if proposal.new_upcoming:
            state["upcoming_nodes"] = proposal.new_upcoming
            self.mongodb.update_student_field(student_id, "state.upcoming_nodes", proposal.new_upcoming)

        if proposal.new_current:
            self.update_learning_position(session_id, proposal.new_current)

        self._rebuild_context(session, proposal)
        state["pending_proposal"] = None
        self.mongodb.update_student_field(student_id, "state.pending_proposal", None)
        self.mongodb.update_student_field(student_id, "state.summary", state["summary"])

    # ──────────────────────────────────────────────────────────────────────────
    # Admin / internal helpers (dùng trực tiếp student_id)
    # ──────────────────────────────────────────────────────────────────────────

    def get_student_profile(self, student_id: str):
        """Dùng từ student/api.py, không dùng từ TA."""
        return self.sqldb.get_student_by_id(student_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve(self, session_id: str) -> str:
        """Map session_id → student_id. Raise ValueError nếu không tồn tại."""
        student_id = self._session_map.get(session_id)
        if not student_id:
            raise ValueError(
                f"Unknown or expired chat session: '{session_id}'. "
                "Client must call POST /student/session/start first."
            )
        return student_id

    def _get_or_create_student_session(
        self, student_id: str, session_id: str
    ) -> "StudentSession":
        # 1. Đảm bảo learning state được load (dùng chung giữa các session)
        if student_id not in self._student_states:
            self._ensure_student_exists(student_id)
            self._student_states[student_id] = self.mongodb.get_student_state(student_id)
        
        state = self._student_states[student_id]

        # 2. Lấy hoặc tạo session-specific object
        if session_id not in self._sessions:
            session_data = self.mongodb.get_session_data(student_id, session_id)
            if not session_data:
                self.mongodb.create_session(student_id, session_id)
                session_data = self.mongodb.get_session_data(student_id, session_id)

            memo = Memo(
                session_id=session_id,
                session_data=session_data
            )
            # Load persisted SessionContext, or create fresh
            ctx_data = self.mongodb.get_session_context(student_id, session_id)
            context = (
                SessionContext.from_dict(session_id, ctx_data)
                if ctx_data
                else SessionContext(session_id=session_id)
            )
            self._sessions[session_id] = StudentSession(student_id, state, memo, context)
        
        return self._sessions[session_id]

    def _ensure_student_exists(self, student_id: str):
        if self.sqldb.get_student_by_id(student_id) is None:
            self.sqldb.create_student(student_id)
            self.mongodb.create_student(student_id)

    def _on_node_change(self, session: "StudentSession"):
        session.recent_pages.clear()
        session.student_state["active_resource"] = None

    def _rebuild_context(self, session: "StudentSession", proposal: LearningProposal):
        session.recent_pages.clear()
        session.student_state["active_resource"] = None
        existing = session.student_state.get("summary", "") or ""
        marker = f"\n[TRANSITION] {proposal.source_wf}: {proposal.reason}"
        session.student_state["summary"] = (existing + marker).strip()
