import json
from typing import Type, List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from TA.tools.neo.base import NeoTool
from TA.tools.neo.schema import CourseTreeInput
from TA.tools.tool_config import PREREQUISITE_WEIGHT
from TA.edu.helper.tree_order import topo_order
from core.repo.graph.cypher.tools.course import (
    CYPHER_course_tree_concept, CYPHER_course_tree_prereq,
)


def _assemble_tree(course: str, concepts: List[dict], prereqs: List[Tuple[str, str]],
                   max_per_topic: int, max_orphans: int) -> dict:
    ## backbone per topic, topo-order siblings
    by_topic: Dict[Optional[str], List[dict]] = {}
    for c in concepts:
        by_topic.setdefault(c.get("topic"), []).append(c)

    selected: Dict[str, dict] = {}
    topics_out = []
    for topic, members in by_topic.items():
        if topic is None:
            continue
        members.sort(key=lambda c: -c["score"])
        keep = members[:max_per_topic]
        for c in keep:
            selected[c["name"]] = c
        topics_out.append({"name": topic, "members": keep})

    orphans = sorted(by_topic.get(None, []), key=lambda c: -c["score"])[:max_orphans]
    for c in orphans:
        selected[c["name"]] = c

    requires: Dict[str, List[str]] = {n: [] for n in selected}
    edges_in = [(p, q) for p, q in prereqs if p in selected and q in selected]
    for pre, post in edges_in:
        requires[post].append(pre)

    def _node(c: dict) -> dict:
        return {"name": c["name"], "type": c.get("type", ""),
                "score": c["score"], "requires": requires.get(c["name"], []),
                "description": (c.get("description") or "")[:80]}

    scores = {n: c["score"] for n, c in selected.items()}
    final_topics = []
    for t in topics_out:
        names = [c["name"] for c in t["members"]]
        order = topo_order(names, edges_in, scores)
        by_name = {c["name"]: c for c in t["members"]}
        final_topics.append({"name": t["name"],
                             "score": sum(scores[n] for n in names),
                             "concepts": [_node(by_name[n]) for n in order]})

    topic_of = {c["name"]: t["name"] for t in topics_out for c in t["members"]}
    t_edges = list({(topic_of[p], topic_of[q]) for p, q in edges_in
                    if topic_of.get(p) and topic_of.get(q) and topic_of[p] != topic_of[q]})
    t_scores = {t["name"]: t["score"] for t in final_topics}
    t_order = topo_order([t["name"] for t in final_topics], t_edges, t_scores)
    final_topics.sort(key=lambda t: t_order.index(t["name"]))

    orphan_names = topo_order([c["name"] for c in orphans], edges_in, scores)
    orphan_by_name = {c["name"]: c for c in orphans}
    return {"course": course, "topics": final_topics,
            "orphan_concepts": [_node(orphan_by_name[n]) for n in orphan_names]}


class CourseTree(NeoTool):
    name: str = "course_tree"
    description: str = (
        "Build the backbone tree of a course: Comm -> Topic -> main concepts, ordered "
        "left-to-right by learning order (prerequisites first). Returns the FULL backbone "
        "as JSON. Low-relevance concepts are excluded — discover them later via "
        "recommend_new during teaching."
    )
    args_schema: Type[BaseModel] = CourseTreeInput
    mongo: Optional[Any] = Field(default=None, exclude=True)

    def _run(self, course_name: str, max_per_topic: int = 5, max_orphans: int = 5,
             force_rebuild: bool = False, config: RunnableConfig = None):
        course_name = self._norm_name(course_name)
        print(f"Run {self.name} | course={course_name}, max_per_topic={max_per_topic}")

        if self.mongo is not None and not force_rebuild:
            cached = self.mongo.get_course_tree(course_name)
            if cached.get("tree"):
                return json.dumps(cached["tree"], ensure_ascii=False)

        rows = self.run_query(CYPHER_course_tree_concept, {"course_name": course_name, "w": PREREQUISITE_WEIGHT})
        if not rows:
            return f"INFO: No concept nodes found for course '{course_name}'."

        concepts = [{"name": r["name"], "topic": r.get("topic"), "type": r.get("type", ""),
                     "score": float(r.get("score") or 0.0),
                     "description": r.get("content") or ""} for r in rows]

        names = [c["name"] for c in concepts]
        prereq_rows = self.run_query(CYPHER_course_tree_prereq, {"names": names}) or []
        prereqs = [(r["pre"], r["post"]) for r in prereq_rows]

        tree = _assemble_tree(course_name, concepts, prereqs, max_per_topic, max_orphans)
        if self.mongo is not None:
            self.mongo.put_course_tree(course_name, tree)
        return json.dumps(tree, ensure_ascii=False)

    async def _arun(self, course_name: str, max_per_topic: int = 5, max_orphans: int = 5,
                    force_rebuild: bool = False, config: RunnableConfig = None):
        return self._run(course_name, max_per_topic, max_orphans, force_rebuild, config)
