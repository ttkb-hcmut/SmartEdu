from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.schema.retrieval import RetrievalRunContext


_TA_NODES = {
    "TA_Retrieve_Finish",
    "TA_Roadmap_Finish",
    "TA_Teach_Finish",
    "TA_Unknown_Finish",
    "Roadmap_Evaluator",
    "TA_Advice",
    "Teach_Understand",
    "Teach_Lecture",
    "Teach_Evaluate",
    "Next_Topic",
}
_RAG_NODES = {
    "RAG_Core",
    "Retrieval_Aggregator",
    "Retrieval_Ledger_Aggregator",
    "Roadmap_Explore",
    "Teach_RAG",
}
_RETRIEVAL_NODES = {
    "WF_Retrieve",
    "Agentic_Retrieve",
    "Retrieve_Harness",
    "agentic-v1",
    "agentic-v2",
    "agentic-v3",
    "agentic-v4",
    "Agentic_Ledger_Retrieve",
    "Retrieval_Round",
    "Seed_Retrieval",
    "Seed_Retrieve",
    "Retrieve_Hop",
    "fanout-v1",
    "Retrieve_Dispatch",
    "Comp_Semantic",
    "Comp_Textbook",
    "Fusion",
}
_DETERMINISTIC_NODES = {
    "WF_Roadmap",
    "WF_Teach",
    "Apply_Proposal",
    "__error__",
}


def _model_config(agent: Any, profile: str) -> dict[str, Any]:
    model = getattr(agent, "model", None)
    if model is None or isinstance(model, str):
        model = agent
    return {
        "kind": "llm",
        "profile": profile,
        "model": getattr(model, "model", ""),
        "temperature": getattr(model, "temperature", None),
        "num_ctx": getattr(model, "num_ctx", None),
        "num_predict": getattr(model, "num_predict", None),
        "keep_alive": getattr(model, "keep_alive", None),
        "reasoning": getattr(model, "reasoning", None),
        "timeout_s": getattr(model, "timeout", None),
        "retry_layer": "none",
        "max_retries": 0,
    }


def build_node_config_manifest(
    agents: dict[str, Any],
    context: RetrievalRunContext,
) -> dict[str, dict[str, Any]]:
    ta = _model_config(agents["TA"], "TA")
    rag = _model_config(agents["RAG"], context.policy.model_profile)
    aggregator_key = {
        "agentic-v3": "RAG_LEDGER_AGGREGATOR",
        "agentic-v4": "RETRIEVAL_PLANNER",
    }.get(context.harness.id.value, "RAG_AGGREGATOR")
    aggregator = _model_config(agents.get(aggregator_key, agents["RAG"]), context.policy.model_profile)
    if context.harness.id.value == "agentic-v4":
        aggregator.update(
            model=context.policy.model_name,
            temperature=context.policy.temperature,
            timeout_s=context.policy.model_timeout_s,
            retry_layer="controller",
            max_retries=context.policy.model_transport_retries,
            schema_repair_attempts=context.policy.schema_repair_attempts,
        )
    answerer = _model_config(
        agents.get("RETRIEVAL_ANSWERER", agents["TA"]),
        context.policy.answer_model_profile,
    )
    answerer.update(
        model=context.policy.answer_model_name,
        temperature=context.policy.answer_temperature,
    )
    retrieval = {
        "policy_id": context.policy.id.value,
        "policy_digest": context.policy.digest,
        "harness_id": context.harness.id.value,
        "harness_digest": context.harness.digest,
        "recursion_limit": context.harness.recursion_limit,
        "top_k": context.harness.top_k,
        "per_source_k": context.harness.per_source_k,
        "rrf_k": context.harness.rrf_k,
        "max_tool_calls": context.harness.max_tool_calls,
        "max_calls_per_round": context.harness.max_calls_per_round,
        "max_context_chars": context.harness.max_context_chars,
        "evidence_excerpt_chars": context.harness.evidence_excerpt_chars,
        "answer_context_chars": context.policy.answer_context_chars,
        "tools": [tool.value for tool in context.policy.allowed_tools],
        "course_scope": context.scope.course,
    }
    deterministic = {
        "kind": "deterministic",
        "model": None,
        "retry_layer": "none",
        "max_retries": 0,
    }
    manifest = {
        node: deepcopy(ta) for node in _TA_NODES
    }
    manifest.update({node: {**deepcopy(rag), **retrieval} for node in _RAG_NODES})
    manifest["Retrieval_Aggregator"] = {**deepcopy(aggregator), **retrieval}
    manifest["Retrieval_Ledger_Aggregator"] = {**deepcopy(aggregator), **retrieval}
    manifest["Plan_Hop"] = {**deepcopy(aggregator), **retrieval}
    manifest["Finalize_Chain"] = {**deepcopy(aggregator), **retrieval}
    manifest["TA_Retrieve_Finish"] = {**deepcopy(answerer), **retrieval}
    manifest.update({node: {**deepcopy(deterministic), **retrieval} for node in _RETRIEVAL_NODES})
    manifest["TA_Router"] = {
        **deepcopy(ta),
        "temperature": 0,
        "num_predict": 256,
        "forced_route_bypasses_model": context.case.forced_route is not None,
    }
    manifest.update({node: deepcopy(deterministic) for node in _DETERMINISTIC_NODES})
    return manifest


def node_execution_config(
    manifest: dict[str, dict[str, Any]],
    node: str,
) -> dict[str, Any]:
    base = node.split(" (", 1)[0]
    if base.startswith("Comp_"):
        base = base if base in manifest else "Comp_Textbook"
    return deepcopy(manifest.get(base, {
        "kind": "unclassified",
        "model": None,
        "retry_layer": "unknown",
        "max_retries": None,
    }))
