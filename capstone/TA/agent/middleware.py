"""
## -- NodeSchemaMiddleware: dynamically inject Pydantic schema per node via awrap_model_call
Reference from:
https://docs.langchain.com/oss/python/langchain/middleware/overview

This design is heavily inspired from this git issue discussion:
https://github.com/langchain-ai/langchain/issues/34239#issuecomment-3630254458
"""
from __future__ import annotations
from typing import Any, Dict, Type, Awaitable, Callable

from typing_extensions import NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState as _MiddlewareAgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from TA.helper.schema import (
    RAGCore,
    RAGDeep,
    RoadmapExplore,
    TeachLectureOutput,
)

class _NodeSchemaState(_MiddlewareAgentState):
    """ 
    Extend agent state with current_node for middleware routing
    This is mandatorily not required for init state (since no node available for starter node)
    And can only be added during runtime
    """
    current_node: NotRequired[str]


class NodeMiddle(AgentMiddleware):
    """ Override response_format per node based on state['current_node']
    
    Hybrid strategy:
    - Tool nodes (RAG_Core, RAG_Deep, Roadmap_Explore): ToolStrategy
      → Allows ReAct loop to call intermediate tools, final schema submitted as tool call
    - Non-tool nodes (future): ProviderStrategy
      → Native Ollama JSON mode for maximum output stability
    """

    state_schema = _NodeSchemaState

    ## TA_*_Finish absent on purpose, stream plain text not structured
    TOOL_NODE_SCHEMAS = {
        # RAG agent nodes
        "RAG_Core": RAGCore,
        "RAG_Deep": RAGDeep,
        "Roadmap_Explore": RoadmapExplore,
        # TA agent nodes
        "Teach_Lecture": TeachLectureOutput,
    }

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """ Hybrid schema injection based on current_node type """
        node_name = request.state.get("current_node", "")
        target_schema = self.TOOL_NODE_SCHEMAS.get(node_name)

        if target_schema:
            ## -- Tool node: ToolStrategy lets the model call intermediate tools freely
            ## -- and submit the final output by calling the schema tool at the end
            new_request = request.override(
                response_format=ToolStrategy(schema=target_schema)
            )
            return await handler(new_request)

        return await handler(request)
