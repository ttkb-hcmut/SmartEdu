from __future__ import annotations

from typing import Annotated, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

from core.schema.retrieval import RetrievalHarnessId, RetrievalRunContext
from TA.retrieval.policy import get_tool_spec
from TA.tools.retrieval import RetrieveMore, SemanticSearch, TextbookSearch


_RETRIEVAL_NODES = {"RAG_Core", "Retrieval_Aggregator", "Retrieval_Ledger_Aggregator"}


def _tool_names(context: RetrievalRunContext) -> set[str]:
    if context.harness.id is RetrievalHarnessId.AGENTIC_V3:
        return {"retrieve_more"}
    return {get_tool_spec(tool_id).name for tool_id in context.policy.allowed_tools}


def _normalized_call(query: str, sources: list[str]) -> tuple[str, tuple[str, ...]]:
    return " ".join(query.casefold().split()), tuple(sorted(set(sources)))


class RetrievalPolicyState(AgentState):
    current_node: NotRequired[str]
    retrieval_call_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
    blocked_retrieval_calls: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


class RetrievalPolicyMiddleware(AgentMiddleware):
    state_schema = RetrievalPolicyState

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        context = getattr(request.runtime, "context", None)
        if request.state.get("current_node") not in _RETRIEVAL_NODES or not isinstance(
            context, RetrievalRunContext
        ):
            return await handler(request)

        names = _tool_names(context)
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
            per_round = context.harness.max_calls_per_round or remain
            calls = []
            retrieval_calls = 0
            for call in message.tool_calls:
                if call["name"] not in names:
                    calls.append(call)
                elif accepted < min(remain, per_round):
                    retrieval_calls += 1
                    calls.append(call)
                    accepted += 1
                else:
                    retrieval_calls += 1
            blocked = retrieval_calls - accepted
            additional_kwargs = {
                **message.additional_kwargs,
                "blocked_retrieval_calls": blocked,
            }
            capped.append(
                message.model_copy(
                    update={
                        "tool_calls": calls,
                        "additional_kwargs": additional_kwargs,
                    }
                )
            )
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
        if state.get("current_node") not in _RETRIEVAL_NODES or not isinstance(
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
        if not message:
            return None

        names = _tool_names(context)
        calls = [call for call in message.tool_calls if call["name"] in names]
        blocked = int(message.additional_kwargs.get("blocked_retrieval_calls", 0))
        if not calls and not blocked:
            return None

        count = state.get("retrieval_call_count", 0)
        blocked_count = state.get("blocked_retrieval_calls", 0)
        return {
            "retrieval_call_count": count + len(calls),
            "blocked_retrieval_calls": blocked_count + blocked,
        }

    async def awrap_tool_call(self, request, handler):
        context = getattr(request.runtime, "context", None)
        tool = request.tool
        if (
            request.state.get("current_node") not in _RETRIEVAL_NODES
            or not isinstance(context, RetrievalRunContext)
            or not isinstance(tool, (RetrieveMore, SemanticSearch, TextbookSearch))
        ):
            return await handler(request)

        args = tool.args_schema.model_validate(request.tool_call["args"])
        if isinstance(tool, RetrieveMore):
            if not args.sources:
                selected = [tool_id.value for tool_id in context.policy.allowed_tools]
                content, artifact = await tool._arun(
                    args.query, selected, runtime=request.runtime
                )
                artifact.update(
                    args={"query": args.query, "sources": []},
                    requested_sources=[],
                    invalid_sources=["<none>"],
                    defaulted_sources=True,
                    repeated_query=False,
                    context_limit=False,
                )
            elif _normalized_call(args.query, args.sources) in {
                _normalized_call(
                    str(message.artifact.get("args", {}).get("query", "")),
                    list(message.artifact.get("args", {}).get("sources", [])),
                )
                for message in request.state.get("messages", [])
                if isinstance(message, ToolMessage)
                and message.name == tool.name
                and isinstance(message.artifact, dict)
            }:
                content = "This query and source set were already searched; use prior evidence or reformulate."
                artifact = {
                    "tool": tool.name,
                    "source": "parallel",
                    "args": {"query": args.query, "sources": args.sources},
                    "requested_sources": args.sources,
                    "executed_sources": [],
                    "invalid_sources": [],
                    "source_artifacts": [],
                    "chunks": [],
                    "latency_ms": 0.0,
                    "error": "",
                    "repeated_query": True,
                    "context_limit": False,
                    "defaulted_sources": False,
                }
            else:
                current_chars = sum(
                    len(str(message.content)) for message in request.state.get("messages", [])
                )
                worst_round = (
                    len(set(args.sources))
                    * context.harness.per_source_k
                    * context.harness.evidence_excerpt_chars
                    + 2_000
                )
                if current_chars + worst_round > context.harness.max_context_chars:
                    content = "Context limit reached; conclude from the accepted evidence ledger."
                    artifact = {
                        "tool": tool.name,
                        "source": "parallel",
                        "args": {"query": args.query, "sources": args.sources},
                        "requested_sources": args.sources,
                        "executed_sources": [],
                        "invalid_sources": [],
                        "source_artifacts": [],
                        "chunks": [],
                        "latency_ms": 0.0,
                        "error": "",
                        "repeated_query": False,
                        "context_limit": True,
                        "defaulted_sources": False,
                    }
                else:
                    content, artifact = await tool._arun(
                        args.query, args.sources, runtime=request.runtime
                    )
                    artifact.update(
                        repeated_query=False,
                        context_limit=False,
                        defaulted_sources=False,
                    )
        else:
            content, artifact = await tool._arun(args.query, runtime=request.runtime)
        return ToolMessage(
            content=content,
            artifact=artifact,
            name=tool.name,
            tool_call_id=request.tool_call["id"],
        )
