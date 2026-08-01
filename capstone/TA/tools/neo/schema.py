from pydantic import BaseModel, Field
from typing import Optional, List, Union

from core.schema.wf_state import ConceptNode
# --- TOOL INPUT SCHEMAS ---

class EntityInput(BaseModel):
    entity_name: str = Field(description="Name of entity")

class ContentInput(BaseModel):
    node_id: str = Field(description="Internal Graph ID of the concept")
    limit: int = Field(default=10, description="Limit of results")
    
class ExplorerInput(BaseModel):
    node_id: str = Field(description="ID của node gốc để tìm kiếm các thực thể liên quan qua các cạnh.")

class RecommendInput(BaseModel):
    course_filter: Optional[str] = Field(default=None, description="Course/community name to filter recommendations. None = global.")
    from_node: Optional[str] = Field(default=None, description="Current node name to find neighbors from. None = global hub ranking.")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of hub nodes to return (capped at 50).")

class BackboneInput(BaseModel):
    course_name: str = Field(description="Name of the course/community to extract backbone from.")
    max_hubs: int = Field(default=10, ge=1, le=50, description="Maximum number of hub nodes to include in the backbone (capped at 50).")

class RelevanceInput(BaseModel):
    target_course: str = Field(description="Name of the target course to find dependencies for.")
    min_degree: int = Field(default=3, description="Minimum out-degree threshold to qualify as a hub node.")

class CourseTreeInput(BaseModel):
    course_name: str = Field(description="Course/community name to build the backbone tree for.")
    max_per_topic: int = Field(default=5, description="Main concepts kept per topic.")
    max_orphans: int = Field(default=5, description="Topic-less concepts attached directly to the course root.")
    force_rebuild: bool = Field(default=False, description="Bypass the cached tree.")

# --- TOOL OUTPUT SCHEMAS ---



class HubConnection(BaseModel):
    """A relationship between two hub nodes."""
    from_node: str
    to_node: str
    relationship: str
    direction: str = Field(description="FORWARD or REVERSE")

class BackboneOutput(BaseModel):
    """Output of CourseBackbone tool."""
    course_name: str
    hubs: List[ConceptNode]
    connections: List[HubConnection] = []

class RelevanceOutput(BaseModel):
    """One related course entry."""
    related_course: str
    hub_overlap: int
    key_concepts: List[str] = []

class OptimalPathInput(BaseModel):
    start_node: str = Field(description="Name of the starting concept node.")
    end_node: str = Field(description="Name of the target concept node.")
