from dataclasses import asdict, dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

from core.llm.prompt.agents import RAG_PROMPT
from core.schema.retrieval import (
    RetrievalCase,
    RetrievalCodeState,
    RetrievalHarnessContract,
    RetrievalHarnessId,
    RetrievalOutputMode,
    RetrievalPolicyContract,
    RetrievalPolicyId,
    RetrievalPreset,
    RetrievalPromptId,
    RetrievalRunContext,
    RetrievalScope,
    RetrievalToolId,
    RetrievalValidation,
    RetrievalValidity,
)

if TYPE_CHECKING:
    from core.config import Retrieve_param


MULTIHOP_PROMPT = RAG_PROMPT + "\n\n" + (
    "You are a retrieval aggregator, not an answerer. Review the supplied seed "
    "evidence, then use tools only when an additional focused lookup would improve "
    "the evidence. Do not produce a final answer."
)

EVIDENCE_LEDGER_PROMPT = (
    "You are a retrieval aggregator, not the final answerer. The seed evidence and every "
    "later retrieval result form an append-only evidence ledger: never filter, select, or "
    "discard it. Identify the facts already supported and the exact missing reasoning link. "
    "When evidence is missing, call retrieve_more once with a focused query built from entities "
    "discovered in prior evidence and choose the source or sources that best fit that query. "
    "Use retrieval to support a strong claim or perform another hop, not to repeat the original "
    "question. Every retrieve_more call must include non-empty sources. To stop, do not call "
    "retrieve_more. Stop when the chain is supported or no useful focused query remains. Your final "
    "plain-text response must summarize the supported chain and state any remaining uncertainty; "
    "it is advisory and never replaces ledger evidence."
)

TYPED_HOP_PROMPT = (
    "You are a retrieval planner, not an answerer. Inspect the complete evidence ledger and "
    "previous decisions, then return exactly one flat JSON object with these top-level fields: "
    "action, sub_question, query, sources, basis_uris. Never nest the fields under an action "
    "name.\n\n"
    "To retrieve: action must be \"retrieve\". sub_question is the single specific question "
    "this hop must answer -- an actual question, not a description of what is missing. query "
    "is one focused search string built from entities already in the ledger. sources lists "
    "which arms to search this hop. basis_uris lists the accepted evidence URIs that justify "
    "why this hop is needed. All five fields are required together. Example:\n"
    '{"action": "retrieve", "sub_question": "Who succeeded Adelphia as owner of Empire Sports '
    'Network?", "query": "Empire Sports Network owner successor company", '
    '"sources": ["semantic"], "basis_uris": ["Bench_MuSiQue/paragraph/abc123..."]}\n\n'
    "To stop: action must be \"stop\" and stop_reason must be exactly one of "
    "\"chain_complete\", \"insufficient_evidence\", \"context_limit\", \"budget_exhausted\" -- "
    "no other value is accepted. stop is how you say you are ready to answer -- there is no "
    "separate \"answer\" action, never use one. Example:\n"
    '{"action": "stop", "stop_reason": "chain_complete"}\n\n'
    "Ground every retrieval in accepted evidence URIs already in the ledger, use one focused "
    "query per hop, and select only sources listed for this arm. Stop when every sub_question "
    "needed to answer is supported or no grounded query remains. Never remove or hide accepted "
    "evidence."
)

TYPED_FINALIZER_PROMPT = (
    "You are the retrieval chain finalizer, not the answerer. Return exactly one flat JSON "
    "object with these top-level fields: claims, answerable, remaining_uncertainty.\n\n"
    "claims is a list. Each entry is an object with exactly two fields: claim, a sentence "
    "stating one supported fact -- the field is named \"claim\", never \"statement\", \"text\" "
    "or \"fact\"; and evidence_uris, a non-empty list of accepted ledger URIs supporting that "
    "claim. A claim with no evidence URI must be dropped, not emitted empty.\n"
    "answerable is a required boolean -- true only when the claims together answer the "
    "question, false otherwise. Never omit it.\n"
    "remaining_uncertainty is a string, empty when nothing is left open.\n\n"
    "Example:\n"
    '{"claims": [{"claim": "Empire Sports Network was owned by Adelphia Communications.", '
    '"evidence_uris": ["Bench_MuSiQue/paragraph/abc123"]}, {"claim": "Adelphia Communications '
    'was founded by John Rigas.", "evidence_uris": ["Bench_MuSiQue/paragraph/def456"]}], '
    '"answerable": true, "remaining_uncertainty": ""}\n\n'
    "Order claims into a causal chain, judge answerability honestly, and state remaining "
    "uncertainty. Never invent a URI and never remove evidence from the ledger passed to the "
    "answerer."
)

BENCHMARK_ANSWER_PROMPT = (
    "Answer the question using only the retrieved evidence. Return exactly one JSON object "
    "with an answer field holding a concise direct answer, not an explanation. If evidence "
    "is insufficient, use {{\"answer\": \"unknown\"}}.\n\n"
    "Question: {question}\n\n"
    "Aggregator synthesis:\n{synthesis}\n\n"
    "Retrieved evidence:\n{evidence}"
)


@dataclass(frozen=True)
class RetrievalToolSpec:
    name: str
    description: str


_TOOLS = {
    RetrievalToolId.SEMANTIC: RetrievalToolSpec(
        name="semantic_search",
        description="Search course concept content with semantic similarity.",
    ),
    RetrievalToolId.TEXTBOOK: RetrievalToolSpec(
        name="textbook_search",
        description="Search course textbook passages with hybrid semantic and lexical retrieval.",
    ),
}

_TOOLSETS = {
    RetrievalPreset.RAG: (RetrievalToolId.SEMANTIC,),
    RetrievalPreset.FULL: (RetrievalToolId.SEMANTIC, RetrievalToolId.TEXTBOOK),
}

_MINIMUM_CALLS = {
    RetrievalPreset.RAG: 1,
    RetrievalPreset.FULL: 2,
}


def _digest(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _policy_payload() -> dict:
    return {
        "id": RetrievalPolicyId.BASELINE_V3.value,
        "prompt_id": RetrievalPromptId.MULTIHOP_V1.value,
        "prompt": MULTIHOP_PROMPT,
        "model_profile": "RAG",
        "model_name": "qwen3:8b",
        "temperature": 0.0,
        "answer_model_profile": "TA",
        "answer_model_name": "gpt-oss:120b-cloud",
        "answer_temperature": 0.0,
        "toolsets": {
            preset.value: [tool.value for tool in tools]
            for preset, tools in _TOOLSETS.items()
        },
        "minimum_calls": {
            preset.value: minimum for preset, minimum in _MINIMUM_CALLS.items()
        },
        "empty_hits_valid": True,
        "tools": {
            tool.value: asdict(spec) for tool, spec in _TOOLS.items()
        },
    }


_POLICY_DIGEST = _digest(_policy_payload())

_HARNESSES = {
    RetrievalHarnessId.AGENTIC_V1: {
        "max_tool_calls": 6,
        "recursion_limit": 20,
        "top_k": 5,
        "per_source_k": 0,
        "rrf_k": 0,
    },
    RetrievalHarnessId.AGENTIC_V2: {
        "max_tool_calls": 6,
        "recursion_limit": 20,
        "top_k": 5,
        "per_source_k": 8,
        "rrf_k": 60,
    },
    RetrievalHarnessId.AGENTIC_V3: {
        "max_tool_calls": 4,
        "recursion_limit": 20,
        "top_k": 5,
        "per_source_k": 8,
        "rrf_k": 60,
        "max_calls_per_round": 1,
        "max_context_chars": 160_000,
        "evidence_excerpt_chars": 1_000,
    },
    RetrievalHarnessId.AGENTIC_V4: {
        "max_tool_calls": 4,
        "recursion_limit": 20,
        "top_k": 5,
        "per_source_k": 8,
        "rrf_k": 60,
        "max_calls_per_round": 1,
        "max_context_chars": 160_000,
        "evidence_excerpt_chars": 1_000,
    },
    RetrievalHarnessId.FANOUT_V1: {
        "max_tool_calls": 0,
        "recursion_limit": 0,
        "top_k": 5,
        "per_source_k": 8,
        "rrf_k": 60,
    },
}


def get_tool_spec(tool_id: RetrievalToolId) -> RetrievalToolSpec:
    return _TOOLS[tool_id]


def validate_retrieval_artifacts(
    context: RetrievalRunContext,
    artifacts: Sequence[Mapping[str, Any]],
) -> RetrievalValidation:
    errors = tuple(str(item.get("error")) for item in artifacts if item.get("error"))
    if errors:
        return RetrievalValidation(RetrievalValidity.INVALID, len(artifacts), errors)
    if len(artifacts) < context.policy.min_tool_calls:
        return RetrievalValidation(RetrievalValidity.POLICY_INVALID, len(artifacts))
    if artifacts and not context.policy.empty_hits_valid and not any(
        item.get("chunks") for item in artifacts
    ):
        return RetrievalValidation(RetrievalValidity.POLICY_INVALID, len(artifacts))
    return RetrievalValidation(RetrievalValidity.VALID, len(artifacts))


def resolve_retrieval_context(
    params: "Retrieve_param",
    case: RetrievalCase | None = None,
    code: RetrievalCodeState | None = None,
) -> RetrievalRunContext:
    try:
        preset = RetrievalPreset(params.preset)
    except ValueError as exc:
        raise ValueError(f"unknown preset: {params.preset}") from exc
    try:
        policy_id = RetrievalPolicyId(params.policy_id)
    except ValueError as exc:
        raise ValueError(f"unknown policy_id: {params.policy_id}") from exc
    try:
        harness_id = RetrievalHarnessId(params.harness_id)
    except ValueError as exc:
        raise ValueError(f"unknown harness_id: {params.harness_id}") from exc

    if policy_id not in {
        RetrievalPolicyId.BASELINE_V3,
        RetrievalPolicyId.BASELINE_V4,
        RetrievalPolicyId.BASELINE_V5,
    }:
        raise ValueError(f"unsupported policy_id: {policy_id.value}")

    if policy_id is RetrievalPolicyId.BASELINE_V5:
        from TA.retrieval.schema import FinalChain, HopDecision

        policy_values = {
            "prompt_id": RetrievalPromptId.MULTIHOP_V3,
            "prompt": TYPED_HOP_PROMPT,
            "model_profile": "retrieval_planner",
            "model_name": "gpt-oss:120b-cloud",
            "temperature": 0.0,
            "answer_model_profile": "retrieval_answerer",
            "answer_model_name": "gpt-oss:120b-cloud",
            "answer_temperature": 0.0,
            ## RETRIEVAL_PLANNER is a bare ChatOllama with no bound tools, so
            ## format+tools conflict does not apply here
            "model_output_mode": RetrievalOutputMode.STRUCTURED,
            "finalizer_prompt": TYPED_FINALIZER_PROMPT,
            "answer_prompt": BENCHMARK_ANSWER_PROMPT,
            "min_tool_calls": 0,
            "model_timeout_s": 120,
            "model_transport_retries": 1,
            "schema_repair_attempts": 1,
            "answer_timeout_s": 120,
            "answer_transport_retries": 1,
        }
        policy_digest = _digest({
            "id": policy_id.value,
            **policy_values,
            "hop_decision_schema": HopDecision.model_json_schema(),
            "final_chain_schema": FinalChain.model_json_schema(),
            "toolsets": {
                arm.value: [tool.value for tool in tools]
                for arm, tools in _TOOLSETS.items()
            },
            "empty_hits_valid": True,
        })
    elif policy_id is RetrievalPolicyId.BASELINE_V4:
        policy_values = {
            "prompt_id": RetrievalPromptId.MULTIHOP_V2,
            "prompt": EVIDENCE_LEDGER_PROMPT,
            "model_profile": "retrieval_aggregator",
            "model_name": "gpt-oss:120b-cloud",
            "temperature": 0.0,
            "answer_model_profile": "retrieval_answerer",
            "answer_model_name": "gpt-oss:120b-cloud",
            "answer_temperature": 0.0,
            "min_tool_calls": 0,
        }
        policy_digest = _digest({
            "id": policy_id.value,
            **policy_values,
            "toolsets": {
                arm.value: [tool.value for tool in tools]
                for arm, tools in _TOOLSETS.items()
            },
            "tools": {
                "retrieve_more": {
                    "arguments": ["query", "sources"],
                    "sources": [tool.value for tool in RetrievalToolId],
                },
                **{tool.value: asdict(spec) for tool, spec in _TOOLS.items()},
            },
            "empty_hits_valid": True,
        })
    else:
        policy_values = {
            "prompt_id": RetrievalPromptId.MULTIHOP_V1,
            "prompt": MULTIHOP_PROMPT,
            "model_profile": "RAG",
            "model_name": "qwen3:8b",
            "temperature": 0.0,
            "answer_model_profile": "TA",
            "answer_model_name": "gpt-oss:120b-cloud",
            "answer_temperature": 0.0,
            "min_tool_calls": _MINIMUM_CALLS[preset],
        }
        policy_digest = _POLICY_DIGEST

    policy = RetrievalPolicyContract(
        id=policy_id,
        digest=policy_digest,
        allowed_tools=_TOOLSETS[preset],
        seed_tools=_TOOLSETS[preset],
        empty_hits_valid=True,
        **policy_values,
    )
    harness_values = _HARNESSES[harness_id]
    harness = RetrievalHarnessContract(
        id=harness_id,
        digest=_digest({"id": harness_id.value, **harness_values}),
        **harness_values,
    )
    return RetrievalRunContext(
        preset=preset,
        policy=policy,
        harness=harness,
        scope=RetrievalScope(course=params.course_scope),
        case=case or RetrievalCase(),
        code=code or RetrievalCodeState(),
    )
