from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from core.schema.wf_state import AgentOutput, AgentState, ConceptNode

class BaseResponse(BaseModel):
    thought: str = Field(description="Brief reasoning for your stratergy, action and answer ")

class RAGCore(BaseResponse):
    entity_ids: List[str] = Field(default_factory=list, description="Resolved entity IDs from the KG")
    content: str = Field(default="", description="Synthesized content from tools")
    status: str = Field(default="SUCCESS", description="SUCCESS or FAIL")

class RAGDeep(BaseResponse):
    is_deep: bool = Field(description="Whether query requires deep analysis")
    bridge_concepts: List[Dict[str, Any]] = Field(default_factory=list, description="Prerequisite concepts the student needs")
    knowledge_gap_score: float = Field(default=0.0, description="0.0 = no gap, 1.0 = huge gap")

class DeepDecision(BaseResponse):
    decision: Literal["DEEP", "SKIP"] = Field(description="DEEP if deep analysis needed, SKIP otherwise")

class RouterDecision(BaseModel):
    intent: Literal["retrieve", "roadmap", "teaching", "confirm", "unknown"] = Field(
        description="Classified intent of the student query"
    )

## -- Roadmap schemas: UI-facing, need ai_message

class RoadmapExplore(BaseResponse):
    start_node: Optional[ConceptNode] = None
    goal: str = Field(description="Learning goal extracted from query")
    steps: List[ConceptNode] = Field(default_factory=list, description="Ordered hub nodes for the path")
    ai_message: str = Field(default="", description="Natural language narrative for the student about this exploration")
    raw_context: str = Field(default="", description="Captured KG tool output (course_relevance/backbone) for downstream prereq evaluation; set programmatically, not by the LLM")

class RoadmapCritique(BaseResponse):
    is_feasible: bool = Field(description="Whether the proposed path is feasible for this student")
    ai_message: str = Field(default="", description="Natural language critique summary for downstream synthesis")

class RoadmapFinal(BaseResponse):
    final_steps: List[ConceptNode] = Field(default_factory=list, description="Finalized ordered steps")
    pedagogical_advice: str = Field(description="Actionable advice for the student")
    ai_message: str = Field(default="", description="Natural language roadmap narrative for the student")

## -- Teach schemas

class TeachLectureOutput(BaseResponse):
    """UI-facing: lecture field IS the narrative."""
    lecture: str = Field(description="The lecture content to present to the student")
    challenge_question: str = Field(default="", description="A Socratic question to test understanding")

class TeachEvalOutput(BaseResponse):
    """UI-facing via finish node."""
    criteria: str = Field(description="Evaluation criteria derived from session questions")
    user_eval: str = Field(description="Assessment of the student's responses")
    passed: bool = Field(description="Whether the student demonstrated sufficient mastery")
    ai_message: str = Field(default="", description="Natural language evaluation feedback for the student")

class NextTopicOutput(BaseResponse):
    """Structured output for next topic selection."""
    selected_nodes: List[str] = Field(default_factory=list, description="Ordered list of next topic names, 1-3 max")