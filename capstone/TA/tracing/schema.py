"""
Trace log schema — SmartEdu TA Agent
Version: 1.1

Output format:
{
  "schema_version": "1.1",
  "session_id": "student_123",
  "chat": [
    {
      "chat_id": "20260507_191233",
      "query": "...",
      "agent": [
        {
          "node": "TA_Router",
          "prompt": "...",
          "state": { ... },
          "tool_result": {},
          "output": "..."
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StepTrace(BaseModel):
    """Trace logical step: tool call, textual, though"""

    node: str = Field(description="Node name (TA_Router, RAG_Core, ...)")
    prompt: str = Field(default="", description="Prompt fed to LLM at this step")
    state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot of StudentState (serialized, stringified)"
    )
    tool_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="tools (worker_results snapshot)"
    )
    output: str = Field(default="", description="Raw LLM output or node result")
    ## uri matched vs gold set in eval
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved chunks at this step")
    latency_ms: float = Field(default=0.0, description="Node wall time")
    tokens: Dict[str, int] = Field(default_factory=dict, description="LLM usage if available")


class ChatTrace(BaseModel):
    """Trace of chat turn: query → full agent execution (divided into steps) --> output --> serialized response"""

    chat_id: str = Field(description="Timestamp YYYYMMDD_HHMMSS")
    query: str = Field(description="User query")
    intent: str = Field(default="", description="Routed intent")
    agent: List[StepTrace] = Field(default_factory=list, description="Agentic steps")
    final_output: str = Field(default="", description="Serialized response returned to user")
    status: str = Field(default="SUCCESS", description="SUCCESS | FAIL")
    retrieve_flags: Dict[str, bool] = Field(default_factory=dict, description="Ablation component flags")
    preset: str = Field(default="", description="PLAIN | RAG | FULL | CUSTOM")
    policy_id: str = ""
    policy_digest: str = ""
    harness_id: str = ""
    run_id: str = ""
    question_id: str = ""
    warmup: bool = False
    model_profile: str = ""
    model: str = ""
    temperature: float = 0.0
    code_revision: str = ""
    dirty: bool = False
    errors: List[str] = Field(default_factory=list)


class TraceSession(BaseModel):
    """full state of a session"""

    schema_version: str = Field(default="1.2", description="Migration marker on schema change")
    session_id: str = Field(default="default")
    chat: List[ChatTrace] = Field(default_factory=list)
