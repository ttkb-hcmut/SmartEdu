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
    "Classify the question as single-hop or multi-hop. For multi-hop questions, "
    "decompose it into focused sub-queries, retrieve evidence for each sub-query, "
    "then submit one grounded result. Use only the tools listed for this run. "
    "Do not answer from memory when retrieval is required."
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
    RetrievalPreset.PLAIN: (),
    RetrievalPreset.RAG: (RetrievalToolId.SEMANTIC,),
    RetrievalPreset.FULL: (RetrievalToolId.SEMANTIC, RetrievalToolId.TEXTBOOK),
}

_MINIMUM_CALLS = {
    RetrievalPreset.PLAIN: 0,
    RetrievalPreset.RAG: 1,
    RetrievalPreset.FULL: 1,
}


def _digest(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _policy_payload() -> dict:
    return {
        "id": RetrievalPolicyId.BASELINE_V1.value,
        "prompt_id": RetrievalPromptId.MULTIHOP_V1.value,
        "prompt": MULTIHOP_PROMPT,
        "model_profile": "RAG",
        "model_name": "qwen3:8b",
        "temperature": 0.0,
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

    if policy_id is not RetrievalPolicyId.BASELINE_V1:
        raise ValueError(f"unsupported policy_id: {policy_id.value}")

    policy = RetrievalPolicyContract(
        id=policy_id,
        digest=_POLICY_DIGEST,
        prompt_id=RetrievalPromptId.MULTIHOP_V1,
        prompt=MULTIHOP_PROMPT,
        model_profile="RAG",
        model_name="qwen3:8b",
        temperature=0.0,
        allowed_tools=_TOOLSETS[preset],
        min_tool_calls=_MINIMUM_CALLS[preset],
        empty_hits_valid=True,
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
