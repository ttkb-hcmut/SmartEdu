import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from core.schema.retrieval import (
    RetrievalRunContext,
    RetrievalValidation,
    RetrievalValidity,
)
from core.schema.wf_state import AgentState
from TA.retrieval.policy import validate_retrieval_artifacts
from TA.tools.retrieval import RetrieveMore, SemanticSearch, TextbookSearch, append_ledger

logger = logging.getLogger(__name__)


def _model_contract_error(agent, context: RetrievalRunContext) -> str:
    model = getattr(agent, "model", None)
    name = getattr(model, "model", None) or getattr(model, "model_name", None)
    temperature = getattr(model, "temperature", None)
    if name != context.policy.model_name:
        return f"model mismatch: expected {context.policy.model_name}, got {name or 'unknown'}"
    if temperature is None or float(temperature) != context.policy.temperature:
        return (
            f"temperature mismatch: expected {context.policy.temperature}, "
            f"got {temperature if temperature is not None else 'unknown'}"
        )
    return ""


def _followup_artifacts(messages) -> List[Dict[str, Any]]:
    return [
        message.artifact
        for message in messages
        if isinstance(message, ToolMessage)
        and message.name == "retrieve_more"
        and isinstance(message.artifact, dict)
    ]


def _rag_envelope(
    ledger: List[Dict[str, Any]],
    synthesis: str,
    validation: RetrievalValidation,
    excerpt_chars: int,
    execution_error: str,
) -> Dict[str, Any]:
    return {
        "thought": synthesis,
        "entity_ids": [str(item.get("id") or item["uri"]) for item in ledger],
        "content": "\n".join(
            f"- [{item['uri']}] {item.get('text', '')[:excerpt_chars]}" for item in ledger
        ),
        "status": "SUCCESS" if validation.validity is RetrievalValidity.VALID else "FAIL",
        "validity": validation.validity.value,
        "retrieval_attempts": validation.attempted_calls,
        "errors": list(validation.errors),
        "agent_status": "completed" if not execution_error else "failed",
    }


def build_agentic_v3_retrieve_wf(agents, resources: Optional[Dict] = None):
    builder = StateGraph(AgentState, context_schema=RetrievalRunContext)
    resources = resources or {}
    aggregator = agents.get("RAG_LEDGER_AGGREGATOR")
    retrieve_more = RetrieveMore(
        semantic=SemanticSearch(
            milvus_db=resources.get("milvus_db"),
            embedder=resources.get("embedder"),
        ),
        textbook=TextbookSearch(
            graph_db=resources.get("graph_db"),
            embedder=resources.get("embedder"),
        ),
    )

    async def run_ledger(state, config, runtime: Runtime[RetrievalRunContext]):
        started = time.perf_counter()
        context = runtime.context
        query = state.get("user_query", state["messages"][-1].content)
        seed_content, seed_artifact = await retrieve_more._arun(
            query,
            [tool.value for tool in context.policy.seed_tools],
            runtime=runtime,
        )
        ledger, _, _ = append_ledger(
            [], seed_artifact.get("chunks", []), round_index=0, query=query
        )
        followups: List[Dict[str, Any]] = []
        aggregator_messages = []
        execution_error = ""
        synthesis = ""

        if aggregator is None:
            execution_error = "RAG ledger aggregator unavailable"
        elif model_error := _model_contract_error(aggregator, context):
            execution_error = model_error
        else:
            try:
                result = await aggregator.ainvoke(
                    {
                        "messages": [(
                            "user",
                            f"Question: {query}\n\nSeed evidence ledger:\n{seed_content}",
                        )],
                        "current_node": "Retrieval_Ledger_Aggregator",
                        "retrieval_call_count": 0,
                    },
                    config={**config, "recursion_limit": context.harness.recursion_limit},
                    context=context,
                )
                aggregator_messages = result.get("messages", [])
                followups = _followup_artifacts(aggregator_messages)
                for round_index, artifact in enumerate(followups, start=1):
                    ledger, _, _ = append_ledger(
                        ledger,
                        artifact.get("chunks", []),
                        round_index=round_index,
                        query=str(artifact.get("args", {}).get("query", "")),
                    )
                synthesis = next(
                    (
                        str(message.content)
                        for message in reversed(aggregator_messages)
                        if isinstance(message, AIMessage) and message.content
                    ),
                    "",
                )
                if not synthesis and hasattr(aggregator.model, "ainvoke"):
                    forced = await aggregator.model.ainvoke(
                        [
                            ("system", context.policy.prompt),
                            (
                                "user",
                                "Conclude now from accepted evidence. State supported chain and uncertainty.",
                            ),
                        ],
                        config=config,
                    )
                    synthesis = str(getattr(forced, "content", ""))
            except Exception as exc:
                execution_error = f"{type(exc).__name__}: {exc}"
                logger.warning("[agentic_v3] aggregator failed: %s", execution_error)

        deep_artifacts = [seed_artifact, *followups]
        blocked_tool_calls = sum(
            int(message.additional_kwargs.get("blocked_retrieval_calls", 0))
            for message in aggregator_messages
            if isinstance(message, AIMessage)
        )
        source_artifacts = [
            source_artifact
            for artifact in deep_artifacts
            for source_artifact in artifact.get("source_artifacts", [])
        ]
        validation = validate_retrieval_artifacts(context, source_artifacts)
        deep_errors = [artifact["error"] for artifact in deep_artifacts if artifact.get("error")]
        if execution_error or deep_errors:
            validation = RetrievalValidation(
                RetrievalValidity.INVALID,
                len(source_artifacts),
                (*validation.errors, *deep_errors, *([execution_error] if execution_error else [])),
            )
        rag_result = _rag_envelope(
            ledger,
            synthesis,
            validation,
            context.harness.evidence_excerpt_chars,
            execution_error,
        )

        configurable = config.get("configurable", {})
        tracer, chat_id = configurable.get("tracer"), configurable.get("chat_id", "")
        aggregator_latency_ms = (time.perf_counter() - started) * 1000
        if tracer and chat_id:
            for round_index, artifact in enumerate(deep_artifacts):
                for source_artifact in artifact.get("source_artifacts", []):
                    tracer.log_step(
                        chat_id=chat_id,
                        node=f"Comp_{source_artifact.get('source', 'Unknown').title()}",
                        tool_result={
                            "round": round_index,
                            "tool": source_artifact.get("tool", ""),
                            "args": source_artifact.get("args", {}),
                            "error": source_artifact.get("error", ""),
                        },
                        chunks=source_artifact.get("chunks", []),
                        latency_ms=source_artifact.get("latency_ms", 0.0),
                    )
                tracer.log_step(
                    chat_id=chat_id,
                    node="Retrieval_Round",
                    tool_result={
                        "round": round_index,
                        "query": artifact.get("args", {}).get("query", ""),
                        "requested_sources": artifact.get("requested_sources", []),
                        "executed_sources": artifact.get("executed_sources", []),
                        "invalid_sources": artifact.get("invalid_sources", []),
                        "repeated_query": artifact.get("repeated_query", False),
                        "context_limit": artifact.get("context_limit", False),
                        "defaulted_sources": artifact.get("defaulted_sources", False),
                        "error": artifact.get("error", ""),
                    },
                    chunks=artifact.get("chunks", []),
                    latency_ms=artifact.get("latency_ms", 0.0),
                )
            tracer.log_step(
                chat_id=chat_id,
                node="Retrieval_Ledger_Aggregator",
                tool_result={
                    "seed_calls": len(seed_artifact.get("source_artifacts", [])),
                    "aggregator_calls": len(followups),
                    "blocked_tool_calls": blocked_tool_calls,
                    "aggregator_latency_ms": aggregator_latency_ms,
                    "error": execution_error,
                },
                output=synthesis,
                latency_ms=aggregator_latency_ms,
            )
            tracer.log_step(
                chat_id=chat_id,
                node="Agentic_Retrieve",
                tool_result={
                    "status": rag_result["status"],
                    "validity": rag_result["validity"],
                    "attempts": rag_result["retrieval_attempts"],
                    "seed_calls": len(seed_artifact.get("source_artifacts", [])),
                    "aggregator_calls": len(followups),
                    "blocked_tool_calls": blocked_tool_calls,
                    "rounds": len(deep_artifacts),
                    "aggregator_latency_ms": aggregator_latency_ms,
                    "invalid_source_requests": sum(len(a.get("invalid_sources", [])) for a in followups),
                    "repeated_queries": sum(bool(a.get("repeated_query")) for a in followups),
                    "context_limit_stops": sum(bool(a.get("context_limit")) for a in followups),
                    "errors": rag_result["errors"],
                },
                chunks=ledger,
            )

        current = state.get("worker_results", {})
        return {
            "worker_results": {**current, "RAG": rag_result},
            "status_flag": "FAIL" if validation.validity is RetrievalValidity.INVALID else "SUCCESS",
        }

    builder.add_node("Agentic_Ledger_Retrieve", run_ledger)
    builder.set_entry_point("Agentic_Ledger_Retrieve")
    builder.add_edge("Agentic_Ledger_Retrieve", END)
    return builder.compile()
