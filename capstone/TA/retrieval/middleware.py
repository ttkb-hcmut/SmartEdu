from __future__ import annotations

from typing import Annotated, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

from core.schema.retrieval import RetrievalRunContext
from TA.retrieval.policy import get_tool_spec


class RetrievalPolicyState(AgentState):
    current_node: NotRequired[str]
    retrieval_call_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


class RetrievalPolicyMiddleware(AgentMiddleware):
    state_schema = RetrievalPolicyState

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        context = getattr(request.runtime, "context", None)
        if request.state.get("current_node") != "RAG_Core" or not isinstance(
            context, RetrievalRunContext
        ):
            return await handler(request)

        names = {
            get_tool_spec(tool_id).name for tool_id in context.policy.allowed_tools
        }
        count = request.state.get("retrieval_call_count", 0)
        remain = max(context.harness.max_tool_calls - count, 0)
        tools = [tool for tool in request.tools if tool.name in names] if remain else []
        roster = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in tools
        ) or "- none"
        content = (
            f"{context.policy.prompt}\n\n"
            f"Retrieval tools available for this run:\n{roster}"
        )
        response = await handler(
            request.override(
                tools=tools,
                system_message=SystemMessage(content=content),
            )
        )
        if not isinstance(response, ModelResponse):
            return response

        capped = []
        for message in response.result:
            if not isinstance(message, AIMessage) or not message.tool_calls:
                capped.append(message)
                continue
            accepted = 0
            calls = []
            for call in message.tool_calls:
                if call["name"] not in names:
                    calls.append(call)
                elif accepted < remain:
                    calls.append(call)
                    accepted += 1
            capped.append(message.model_copy(update={"tool_calls": calls}))
        return ModelResponse(
            result=capped,
            structured_response=response.structured_response,
        )

    async def aafter_model(
        self,
        state: RetrievalPolicyState,
        runtime: Runtime[RetrievalRunContext],
    ) -> dict[str, Any] | None:
        context = runtime.context
        if state.get("current_node") != "RAG_Core" or not isinstance(
            context, RetrievalRunContext
        ):
            return None

        message = next(
            (
                item
                for item in reversed(state.get("messages", []))
                if isinstance(item, AIMessage)
            ),
            None,
        )
        if not message or not message.tool_calls:
            return None

        names = {
            get_tool_spec(tool_id).name for tool_id in context.policy.allowed_tools
        }
        calls = [call for call in message.tool_calls if call["name"] in names]
        if not calls:
            return None

        count = state.get("retrieval_call_count", 0)
        return {"retrieval_call_count": count + len(calls)}
