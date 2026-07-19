from typing import Type, Optional
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from TA.tools.neo.base import NeoTool
from TA.tools.neo.schema import (
    RecommendInput, BackboneInput, RelevanceInput, OptimalPathInput,
    BackboneOutput, HubConnection, RelevanceOutput, ConceptNode
)
from TA.tools.tool_config import PREREQUISITE_WEIGHT
from core.repo.graph.cypher.tools.course import (
    CYPHER_recommend_new, CYPHER_recommend_new_course, CYPHER_recommend_new_from_node,
    CYPHER_course_backbone_hub, CYPHER_course_backbone_rel, CYPHER_course_relevance,
)
from core.repo.graph.cypher.tools.algoritms import (
    CYPHER_optimal_path, CYPHER_optimal_path_fallback,
)


class RecommendNew(NeoTool):
    name: str = "recommend_new"
    description: str = (
        "Find the most connected concept nodes in the knowledge graph. "
        "Returns top hub nodes ranked by semantic out-degree (number of non-CONTENT relationships). "
        "Use this when a student has no current position and needs a starting point."
    )
    args_schema: Type[BaseModel] = RecommendInput

    def _run(self, course_filter: str = None, from_node: str = None, max_results: int = 10, config: RunnableConfig = None):
        session_id = (config or {}).get("configurable", {}).get("session_id", "")
        if course_filter:
            course_filter = self._norm_name(course_filter)
        if from_node:
            from_node = self._norm_name(from_node)
        print(f"Run {self.name} | course_filter={course_filter}, from_node={from_node}, max={max_results}")

        if from_node:
            cypher = CYPHER_recommend_new_from_node
            params = {"from_node": from_node, "max_results": max_results, "prereq_weight": PREREQUISITE_WEIGHT}
        elif course_filter:
            cypher = CYPHER_recommend_new_course
            params = {"course_filter": course_filter, "max_results": max_results, "prereq_weight": PREREQUISITE_WEIGHT}
        else:
            cypher = CYPHER_recommend_new
            params = {"max_results": max_results, "prereq_weight": PREREQUISITE_WEIGHT}

        results = self.run_query(query=cypher, params=params)
        print(f"RecommendNew results: {len(results)} nodes")

        if not results:
            return "INFO: No concept nodes found in knowledge graph."

        nodes = []
        for r in results:
            mastery = 0
            if self.tracker and session_id:
                mastery = self.tracker.get_mastery(session_id, r['name'])
            node = ConceptNode(
                name=r['name'],
                type=r.get('type', ''),
                course_name=r.get('course_name', ''),
                out_degree=r.get('out_degree', 0),
                description=r['content'][:50] if r.get('content') else '', 
                mastery=mastery
            )
            nodes.append(node)

        mode_label = f"NEIGHBORS of '{from_node}'" if from_node else f"TOP {len(nodes)} HUB CONCEPTS"
        lines = [f"{mode_label} (by connectivity):"]
        for i, n in enumerate(nodes, 1):
            lines.append(
                f"{i}. [{n.type}] {n.name} "
                f"| course: {n.course_name or 'N/A'} "
                f"| connections: {n.out_degree} "
                f"| mastery: {n.mastery} "
                f"| {n.description or 'N/A'}"
            )
        return "\n".join(lines)

    async def _arun(self, course_filter: str = None, from_node: str = None, max_results: int = 10, config: RunnableConfig = None):
        return self._run(course_filter, from_node, max_results, config)


class CourseBackbone(NeoTool):
    name: str = "course_backbone"
    description: str = (
        "Extract the 'backbone' (skeletal structure) of a course by finding its top hub concept nodes "
        "and the relationships between them. Returns hub nodes ranked by out-degree plus a connectivity map "
        "showing how hubs relate to each other. Use this to understand a course's core structure for roadmap planning."
    )
    args_schema: Type[BaseModel] = BackboneInput

    def _run(self, course_name: str, max_hubs: int = 15, config: RunnableConfig = None):
        session_id = (config or {}).get("configurable", {}).get("session_id", "")
        course_name = self._norm_name(course_name)
        print(f"Run {self.name} | course={course_name}, max_hubs={max_hubs}")

        hubs = self.run_query(query=CYPHER_course_backbone_hub, params={
            "course_name": course_name, "max_hubs": max_hubs, "prereq_weight": PREREQUISITE_WEIGHT
        })
        
        if not hubs:
            return f"INFO: No concept nodes found for course '{course_name}'."

        hub_ids = [h['id'] for h in hubs]
        hub_nodes = []
        for h in hubs:
            mastery = 0
            if self.tracker and session_id:
                mastery = self.tracker.get_mastery(session_id, h['name'])
            hub_nodes.append(ConceptNode(
                name=h['name'],
                type=h.get('type', ''),
                course_name=h.get('course_name', course_name),
                out_degree=h.get('out_degree', 0),
                description=f"{h['content'][:100]}..." if h.get('content') else '',
                mastery=mastery
            ))

        rels = self.run_query(query=CYPHER_course_backbone_rel, params={"hub_ids": hub_ids})

        connections = [HubConnection(**r) for r in rels] if rels else []

        lines = [f"BACKBONE of '{course_name}' ({len(hub_nodes)} hub nodes):"]
        lines.append("")
        lines.append("HUB NODES:")
        for i, h in enumerate(hub_nodes, 1):
            lines.append(f"  {i}. [{h.type}] {h.name} | degree: {h.out_degree} | mastery: {h.mastery} | {h.description or 'N/A'}")

        if connections:
            lines.append("")
            lines.append("CONNECTIONS BETWEEN HUBS:")
            for c in connections:
                arrow = "->" if c.direction == 'FORWARD' else "<-"
                lines.append(f"  {c.from_node} {arrow}[{c.relationship}]{arrow} {c.to_node}")

        return "\n".join(lines)

    async def _arun(self, course_name: str, max_hubs: int = 10, config: RunnableConfig = None):
        return self._run(course_name, max_hubs, config)


class CourseRelevance(NeoTool):
    name: str = "course_relevance"
    description: str = (
        "Find which other courses are most relevant to a target course by counting how many "
        "hub nodes (high out-degree concepts) from other courses point into the target course's hub nodes. "
        "Returns a compact summary of related courses ranked by overlap strength. "
        "Use this to identify prerequisite courses or cross-course dependencies."
    )
    args_schema: Type[BaseModel] = RelevanceInput

    def _run(self, target_course: str, min_degree: int = 3):
        target_course = self._norm_name(target_course)
        print(f"Run {self.name} | target={target_course}, min_degree={min_degree}")

        results = self.run_query(query=CYPHER_course_relevance, params={
            "target_course": target_course, 
            "min_degree": min_degree
        })

        if not results or not any(r['related_course'] for r in results):
            return f"INFO: No significant cross-course dependencies found for '{target_course}'."

        lines = [f"COURSE DEPENDENCIES for '{target_course}':"]
        for r in results:
            if not r['related_course']:
                continue
            concepts = ", ".join(r['key_concepts']) if r['key_concepts'] else "N/A"
            lines.append(f"  {r['related_course']}: {r['hub_overlap']} hub overlaps (via: {concepts})")

        return "\n".join(lines)

    async def _arun(self, target_course: str, min_degree: int = 3):
        return self._run(target_course, min_degree)


class OptimalPath(NeoTool):
    name: str = "optimal_path"
    description: str = (
        "Find the shortest weighted path between two concept nodes in the knowledge graph. "
        "Uses edge weights for optimal path calculation. Returns the ordered sequence of "
        "nodes along the path. Use this to understand the learning distance between concepts."
    )
    args_schema: Type[BaseModel] = OptimalPathInput

    def _run(self, start_node: str, end_node: str, config: RunnableConfig = None):
        session_id = (config or {}).get("configurable", {}).get("session_id", "")
        start_node = self._norm_name(start_node)
        end_node = self._norm_name(end_node)
        print(f"Run {self.name} | start={start_node}, end={end_node}")

        results = self.run_query(query=CYPHER_optimal_path, params={
            "start_node": start_node,
            "end_node": end_node,
        })

        if not results:
            results = self.run_query(query=CYPHER_optimal_path_fallback, params={
                "start_node": start_node,
                "end_node": end_node,
            })

        if not results:
            return f"INFO: No path found between '{start_node}' and '{end_node}'."

        lines = [f"OPTIMAL PATH: {start_node} → {end_node} ({len(results)} steps):"]
        for i, r in enumerate(results):
            mastery = 0
            if self.tracker and session_id:
                mastery = self.tracker.get_mastery(session_id, r['name'])
            desc = r.get('content', '')[:50] if r.get('content') else 'N/A'
            lines.append(
                f"  {i+1}. [{r.get('type', '')}] {r['name']} "
                f"| mastery: {mastery} | {desc}"
            )
        return "\n".join(lines)

    async def _arun(self, start_node: str, end_node: str, config: RunnableConfig = None):
        return self._run(start_node, end_node, config)
