import json
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from core.schema.retrieval import (
    RetrievalRunContext,
    RetrievalValidation,
    RetrievalValidity,
)
from TA.retrieval.policy import validate_retrieval_artifacts
from TA.retrieval.schema import FinalChain, HopAction, HopDecision, StopReason
from TA.helper.model_call import is_transport_error, typed_call
from TA.tools.retrieval import RetrieveMore, SemanticSearch, TextbookSearch, append_ledger


def _tracer_ctx(config):
    configurable = config.get("configurable", {})
    return configurable.get("tracer"), configurable.get("chat_id", "")


def _model_contract_error(model, context: RetrievalRunContext) -> str:
    name = getattr(model, "model", None) or getattr(model, "model_name", None)
    temperature = getattr(model, "temperature", None)
    if name != context.policy.model_name:
        return f"model mismatch: expected {context.policy.model_name}, got {name or 'unknown'}"
    if temperature is None or float(temperature) != context.policy.temperature:
        return f"temperature mismatch: expected {context.policy.temperature}, got {temperature}"
    return ""


def _ledger_text(ledger: List[Dict[str, Any]], excerpt_chars: int) -> str:
    return "\n".join(
        f"- [{item['uri']}] {item.get('text', '')[:excerpt_chars]}" for item in ledger
    )


def _decision_error(
    decision: HopDecision,
    context: RetrievalRunContext,
    ledger: List[Dict[str, Any]],
) -> str:
    if decision.action is HopAction.STOP:
        return ""
    if not decision.basis_uris:
        return "retrieve decision requires ledger basis URIs"
    allowed = set(context.policy.allowed_tools)
    invalid_sources = [source.value for source in decision.sources if source not in allowed]
    if invalid_sources:
        return f"sources not allowed for arm: {', '.join(invalid_sources)}"
    accepted = {str(item["uri"]) for item in ledger}
    missing = [uri for uri in decision.basis_uris if uri not in accepted]
    if missing:
        return f"basis URIs not in ledger: {', '.join(missing)}"
    return ""


def _normalized_call(query: str, sources) -> tuple[str, tuple[str, ...]]:
    return " ".join(query.casefold().split()), tuple(sorted(source.value for source in sources))


def _render_chain(chain: FinalChain) -> str:
    lines = [
        f"{index}. {claim.claim} [{', '.join(claim.evidence_uris)}]"
        for index, claim in enumerate(chain.claims, start=1)
    ]
    if chain.remaining_uncertainty:
        lines.append(f"Uncertainty: {chain.remaining_uncertainty}")
    if not lines:
        lines.append("No supported chain; evidence is insufficient.")
    return "\n".join(lines)


def _envelope(ledger, chain, validation, context, diagnostics):
    return {
        "thought": _render_chain(chain),
        "entity_ids": [str(item.get("id") or item["uri"]) for item in ledger],
        "content": _ledger_text(ledger, context.harness.evidence_excerpt_chars),
        "status": "SUCCESS" if validation.validity is RetrievalValidity.VALID else "FAIL",
        "validity": validation.validity.value,
        "retrieval_attempts": validation.attempted_calls,
        "errors": list(validation.errors),
        "agent_status": "completed" if validation.validity is RetrievalValidity.VALID else "failed",
        **diagnostics,
    }


def build_agentic_v4_retrieve_wf(agents, resources: Optional[Dict] = None):
    builder = StateGraph(dict, context_schema=RetrievalRunContext)
    resources = resources or {}
    planner = agents.get("RETRIEVAL_PLANNER")
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

    async def seed_retrieve(state, config, runtime: Runtime[RetrievalRunContext]):
        started = time.perf_counter()
        context = runtime.context
        query = state.get("user_query", state["messages"][-1].content)
        _, artifact = await retrieve_more._arun(
            query,
            [tool.value for tool in context.policy.seed_tools],
            runtime=runtime,
        )
        ledger, _, _ = append_ledger(
            [], artifact.get("chunks", []), round_index=0, query=query
        )
        tracer, chat_id = _tracer_ctx(config)
        if tracer and chat_id:
            tracer.log_step(
                chat_id=chat_id,
                node="Seed_Retrieve",
                tool_result={"query": query, "error": artifact.get("error", "")},
                chunks=artifact.get("chunks", []),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            tracer.log_step(
                chat_id=chat_id,
                node="Retrieval_Round",
                tool_result={"round": 0, "query": query, "error": artifact.get("error", "")},
                chunks=artifact.get("chunks", []),
                latency_ms=artifact.get("latency_ms", 0.0),
            )
            for source in artifact.get("source_artifacts", []):
                tracer.log_step(
                    chat_id=chat_id,
                    node=f"Comp_{source.get('source', 'Unknown').title()}",
                    tool_result={"round": 0, "error": source.get("error", "")},
                    chunks=source.get("chunks", []),
                    latency_ms=source.get("latency_ms", 0.0),
                )
        return {
            **state,
            "_v4_query": query,
            "_v4_ledger": ledger,
            "_v4_seed": artifact,
            "_v4_followups": [],
            "_v4_decisions": [],
            "_v4_followup_count": 0,
            "_v4_schema_repairs": 0,
            "_v4_transport_retries": 0,
            "_v4_planner_calls": 0,
            "_v4_planner_latency_ms": 0.0,
            "_v4_errors": [],
            "_v4_normalizations": [],
            "_v4_searched": [
                _normalized_call(query, context.policy.seed_tools)
            ],
        }

    async def plan_hop(state, config, runtime: Runtime[RetrievalRunContext]):
        context = runtime.context
        started = time.perf_counter()
        errors = list(state.get("_v4_errors", []))
        repairs = int(state.get("_v4_schema_repairs", 0))
        retries = int(state.get("_v4_transport_retries", 0))
        calls = int(state.get("_v4_planner_calls", 0))
        normalizations = list(state.get("_v4_normalizations", []))
        hop_notes: list[str] = []
        decision = None
        decision_error = ""
        fatal_error = ""
        if state["_v4_followup_count"] >= context.harness.max_tool_calls:
            decision = HopDecision(
                action=HopAction.STOP,
                stop_reason=StopReason.BUDGET_EXHAUSTED,
            )
        elif planner is None:
            errors.append("retrieval planner unavailable")
        elif model_error := _model_contract_error(planner, context):
            errors.append(model_error)
        else:
            repair_note = ""
            for repair_index in range(context.policy.schema_repair_attempts + 1):
                prompt = (
                    f"Question: {state['_v4_query']}\n"
                    f"Allowed sources: {[tool.value for tool in context.policy.allowed_tools]}\n"
                    f"Previous decisions: {json.dumps(state['_v4_decisions'], default=str)}\n"
                    f"Evidence ledger:\n{_ledger_text(state['_v4_ledger'], context.harness.evidence_excerpt_chars)}"
                    f"{repair_note}"
                )
                messages = [("system", context.policy.prompt), ("user", prompt)]
                try:
                    decision, notes, used_retries = await typed_call(
                        planner,
                        HopDecision,
                        messages,
                        config=config,
                        output_mode=context.policy.model_output_mode,
                        timeout_s=context.policy.model_timeout_s,
                        retries=context.policy.model_transport_retries,
                    )
                    retries += used_retries
                    calls += 1
                    hop_notes.extend(notes)
                    decision_error = _decision_error(decision, context, state["_v4_ledger"])
                except Exception as exc:
                    retries += int(getattr(exc, "transport_retries_used", 0))
                    if is_transport_error(exc):
                        fatal_error = f"{type(exc).__name__}: {exc}"
                        break
                    decision_error = f"{type(exc).__name__}: {exc}"
                if not decision_error:
                    break
                if repair_index < context.policy.schema_repair_attempts:
                    repairs += 1
                    repair_note = f"\nPrevious decision invalid: {decision_error}. Return a corrected decision."
            if fatal_error:
                errors.append(fatal_error)
                decision = None
            elif decision_error:
                errors.append(decision_error)
                decision = None
        normalizations.extend(hop_notes)

        if decision is None:
            decision = HopDecision(
                action=HopAction.STOP,
                stop_reason=StopReason.INSUFFICIENT_EVIDENCE,
            )
        if decision.action is HopAction.RETRIEVE:
            current_chars = len(_ledger_text(
                state["_v4_ledger"], context.harness.evidence_excerpt_chars
            ))
            worst_round = (
                len(set(decision.sources))
                * context.harness.per_source_k
                * context.harness.evidence_excerpt_chars
                + 2_000
            )
            if current_chars + worst_round > context.harness.max_context_chars:
                decision = HopDecision(
                    action=HopAction.STOP,
                    supported_claims=decision.supported_claims,
                    stop_reason=StopReason.CONTEXT_LIMIT,
                )
        elapsed = (time.perf_counter() - started) * 1000
        tracer, chat_id = _tracer_ctx(config)
        if tracer and chat_id:
            tracer.log_step(
                chat_id=chat_id,
                node="Plan_Hop",
                tool_result={
                    "action": decision.action.value,
                    "query": decision.query,
                    "sources": [source.value for source in decision.sources],
                    "basis_uris": decision.basis_uris,
                    "stop_reason": decision.stop_reason.value if decision.stop_reason else "",
                    "schema_repairs": repairs,
                    "error": decision_error if errors else "",
                    "provider_normalizations": hop_notes,
                },
                latency_ms=elapsed,
            )
        return {
            **state,
            "_v4_decision": decision,
            "_v4_schema_repairs": repairs,
            "_v4_transport_retries": retries,
            "_v4_planner_calls": calls,
            "_v4_planner_latency_ms": state["_v4_planner_latency_ms"] + elapsed,
            "_v4_errors": errors,
            "_v4_normalizations": normalizations,
        }

    def route_decision(state):
        return "retrieve" if state["_v4_decision"].action is HopAction.RETRIEVE else "finalize"

    async def retrieve_hop(state, config, runtime: Runtime[RetrievalRunContext]):
        context = runtime.context
        decision = state["_v4_decision"]
        round_index = state["_v4_followup_count"] + 1
        key = _normalized_call(decision.query, decision.sources)
        if key in state["_v4_searched"]:
            artifact = {
                "tool": "retrieve_more",
                "source": "parallel",
                "args": {"query": decision.query, "sources": [s.value for s in decision.sources]},
                "requested_sources": [s.value for s in decision.sources],
                "executed_sources": [],
                "invalid_sources": [],
                "source_artifacts": [],
                "chunks": [],
                "latency_ms": 0.0,
                "error": "",
                "repeated_query": True,
                "context_limit": False,
            }
        else:
            _, artifact = await retrieve_more._arun(
                decision.query,
                [source.value for source in decision.sources],
                runtime=runtime,
            )
            artifact.update(repeated_query=False, context_limit=False)
        ledger, new_count, duplicate_count = append_ledger(
            state["_v4_ledger"],
            artifact.get("chunks", []),
            round_index=round_index,
            query=decision.query,
        )
        decision_payload = decision.model_dump(mode="json")
        tracer, chat_id = _tracer_ctx(config)
        if tracer and chat_id:
            for source in artifact.get("source_artifacts", []):
                tracer.log_step(
                    chat_id=chat_id,
                    node=f"Comp_{source.get('source', 'Unknown').title()}",
                    tool_result={"round": round_index, "error": source.get("error", "")},
                    chunks=source.get("chunks", []),
                    latency_ms=source.get("latency_ms", 0.0),
                )
            tracer.log_step(
                chat_id=chat_id,
                node="Retrieve_Hop",
                tool_result={
                    "round": round_index,
                    "query": decision.query,
                    "sources": [source.value for source in decision.sources],
                    "new_evidence": new_count,
                    "duplicate_evidence": duplicate_count,
                    "repeated_query": artifact.get("repeated_query", False),
                    "error": artifact.get("error", ""),
                },
                chunks=artifact.get("chunks", []),
                latency_ms=artifact.get("latency_ms", 0.0),
            )
            tracer.log_step(
                chat_id=chat_id,
                node="Retrieval_Round",
                tool_result={
                    "round": round_index,
                    "query": decision.query,
                    "repeated_query": artifact.get("repeated_query", False),
                    "error": artifact.get("error", ""),
                },
                chunks=artifact.get("chunks", []),
                latency_ms=artifact.get("latency_ms", 0.0),
            )
        return {
            **state,
            "_v4_ledger": ledger,
            "_v4_followups": [*state["_v4_followups"], artifact],
            "_v4_decisions": [*state["_v4_decisions"], decision_payload],
            "_v4_followup_count": round_index,
            "_v4_searched": [*state["_v4_searched"], key],
        }

    async def finalize_chain(state, config, runtime: Runtime[RetrievalRunContext]):
        context = runtime.context
        started = time.perf_counter()
        errors = list(state.get("_v4_errors", []))
        repairs = int(state.get("_v4_schema_repairs", 0))
        retries = int(state.get("_v4_transport_retries", 0))
        calls = int(state.get("_v4_planner_calls", 0))
        normalizations = list(state.get("_v4_normalizations", []))
        chain = None
        if not errors and planner is not None and not _model_contract_error(planner, context):
            repair_note = ""
            for repair_index in range(context.policy.schema_repair_attempts + 1):
                prompt = (
                    f"Question: {state['_v4_query']}\n"
                    f"Decisions: {json.dumps(state['_v4_decisions'], default=str)}\n"
                    f"Complete evidence ledger:\n{_ledger_text(state['_v4_ledger'], context.harness.evidence_excerpt_chars)}"
                    f"{repair_note}\nReturn the final cited reasoning chain."
                )
                messages = [("system", context.policy.finalizer_prompt), ("user", prompt)]
                try:
                    chain, notes, used_retries = await typed_call(
                        planner,
                        FinalChain,
                        messages,
                        config=config,
                        output_mode=context.policy.model_output_mode,
                        timeout_s=context.policy.model_timeout_s,
                        retries=context.policy.model_transport_retries,
                    )
                    retries += used_retries
                    calls += 1
                    normalizations.extend(notes)
                    accepted = {str(item["uri"]) for item in state["_v4_ledger"]}
                    stray = [
                        uri
                        for claim in chain.claims
                        for uri in claim.evidence_uris
                        if uri not in accepted
                    ]
                    if stray:
                        raise ValueError(f"final chain cites unknown URIs: {', '.join(stray)}")
                    break
                except Exception as exc:
                    retries += int(getattr(exc, "transport_retries_used", 0))
                    chain = None
                    final_error = f"{type(exc).__name__}: {exc}"
                    if is_transport_error(exc):
                        errors.append(final_error)
                        break
                    if repair_index < context.policy.schema_repair_attempts:
                        repairs += 1
                        repair_note = f"\nPrevious final chain invalid: {final_error}. Correct it."
                    else:
                        errors.append(final_error)
        if chain is None:
            chain = FinalChain(
                answerable=False,
                remaining_uncertainty="; ".join(errors) or "evidence is insufficient",
            )

        artifacts = [state["_v4_seed"], *state["_v4_followups"]]
        source_artifacts = [
            source
            for artifact in artifacts
            for source in artifact.get("source_artifacts", [])
        ]
        validation = validate_retrieval_artifacts(context, source_artifacts)
        artifact_errors = [artifact["error"] for artifact in artifacts if artifact.get("error")]
        if errors or artifact_errors:
            validation = RetrievalValidation(
                RetrievalValidity.INVALID,
                len(source_artifacts),
                (*validation.errors, *artifact_errors, *errors),
            )
        elapsed = (time.perf_counter() - started) * 1000
        stop_reason = state["_v4_decision"].stop_reason
        diagnostics = {
            "schema_repairs": repairs,
            "transport_retries": retries,
            "stop_reason": stop_reason.value if stop_reason else "",
            "planner_calls": calls,
            "provider_normalizations": normalizations,
        }
        rag_result = _envelope(
            state["_v4_ledger"], chain, validation, context, diagnostics
        )
        tracer, chat_id = _tracer_ctx(config)
        if tracer and chat_id:
            tracer.log_step(
                chat_id=chat_id,
                node="Finalize_Chain",
                tool_result={**diagnostics, "error": "; ".join(errors)},
                output=rag_result["thought"],
                latency_ms=elapsed,
            )
            tracer.log_step(
                chat_id=chat_id,
                node="Agentic_Retrieve",
                tool_result={
                    "status": rag_result["status"],
                    "validity": rag_result["validity"],
                    "attempts": rag_result["retrieval_attempts"],
                    "seed_calls": len(state["_v4_seed"].get("source_artifacts", [])),
                    "aggregator_calls": calls,
                    "blocked_tool_calls": 0,
                    "rounds": len(artifacts),
                    "aggregator_latency_ms": state["_v4_planner_latency_ms"] + elapsed,
                    "invalid_source_requests": 0,
                    "repeated_queries": sum(bool(a.get("repeated_query")) for a in state["_v4_followups"]),
                    "context_limit_stops": int(stop_reason is StopReason.CONTEXT_LIMIT),
                    **diagnostics,
                    "errors": rag_result["errors"],
                },
                chunks=state["_v4_ledger"],
            )
        current = state.get("worker_results", {})
        return {
            **state,
            "worker_results": {**current, "RAG": rag_result},
            "status_flag": "FAIL" if validation.validity is RetrievalValidity.INVALID else "SUCCESS",
            "_error": "; ".join(rag_result["errors"]),
            "_v4_normalizations": normalizations,
        }

    builder.add_node("Seed_Retrieve", seed_retrieve)
    builder.add_node("Plan_Hop", plan_hop)
    builder.add_node("Retrieve_Hop", retrieve_hop)
    builder.add_node("Finalize_Chain", finalize_chain)
    builder.set_entry_point("Seed_Retrieve")
    builder.add_edge("Seed_Retrieve", "Plan_Hop")
    builder.add_conditional_edges(
        "Plan_Hop",
        route_decision,
        {"retrieve": "Retrieve_Hop", "finalize": "Finalize_Chain"},
    )
    builder.add_edge("Retrieve_Hop", "Plan_Hop")
    builder.add_edge("Finalize_Chain", END)
    return builder.compile()
