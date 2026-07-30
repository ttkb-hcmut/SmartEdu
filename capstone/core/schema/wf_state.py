from collections import deque
import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ConceptNode(BaseModel):
    name: str = Field(description="Concept name")
    type: str = Field(default="", description="Node type (Concept, Method, etc)")
    course_name: str = Field(default="", description="Course this concept belongs to")
    out_degree: int = Field(default=0, description="Semantic out-degree (connection count)")
    description: str = Field(default="", description="First 50 chars of node content")
    mastery: int = Field(default=0, description="Student mastery level (0-6, Bloom's)")


class LearningProposal(BaseModel):
    type: Literal["roadmap", "advance", "bridge"] = "advance"
    new_current: Optional[ConceptNode] = None
    new_upcoming: List[ConceptNode] = Field(default_factory=list)
    reason: str = ""
    source_wf: str = ""
    auto_apply: bool = False


class StudentState(TypedDict):
    finished_communities: List[ConceptNode]
    current_pos: ConceptNode
    summary: str
    mastery_map: Dict[str, float]
    previous_nodes: List[ConceptNode]
    upcoming_nodes: List[ConceptNode]
    active_resource: Optional[str]
    pending_proposal: Optional[Dict[str, Any]]

class AgentOutput(BaseModel):
    thought: str = Field(description="Summary of agent thought when deciding on the stratergy")
    status: str = Field(default="SUCCESS", description="SUCCESS | FAIL | NEED_INFO")

class TAOutput(BaseModel):
    summary: str = Field(description="Short summary for memo heading")
    message: str = Field(description="Full answer to user query")
    ui_action: Optional[Dict[str, Any]] = Field(default=None, description="Frontend action, e.g. {'navigate_page': 5, 'document': '...'}")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    student_state: StudentState 
    intent: str
    thought: str
    status_flag: str
    worker_results: Dict[str, AgentOutput]
    user_query: str
    language: str                             # "vn" | "eng" — bilingual toggle from API
    pending_proposal: Optional[Dict[str, Any]]
    ui_action: Optional[Dict[str, Any]]
    _teach_mode: Literal["LECTURE", "REVIEW", "QUIZ", "IDLE"]
    _teach_context: Dict[str, Any]
    # retrieve fan-out
    _retrieve_flags: Dict[str, bool]
    _retrieve_harness: str
    ## or_ reducer -> parallel component writes merge instead of clash
    retrieval_pool: Annotated[Dict[str, Any], operator.or_]
    retrieval_artifacts: Annotated[Dict[str, Any], operator.or_]
    # middleware
    current_node: str
    # roadmap workflow control
    error: Optional[str]            # set when a node bails out (e.g. empty explore) so END isn't silent
    roadmap_attempts: int           # explore→evaluate retry counter, capped to bound the loop



