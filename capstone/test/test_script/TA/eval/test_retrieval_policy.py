from dataclasses import FrozenInstanceError, fields

import pytest


def test_retrieve_param_is_selection_only():
    from core.config import Retrieve_param

    assert {field.name for field in fields(Retrieve_param)} == {
        "preset",
        "policy_id",
        "harness_id",
        "course_scope",
    }


def test_policy_resolves_typed_arm_contract():
    from core.config import Retrieve_param
    from core.schema.retrieval import (
        RetrievalHarnessId,
        RetrievalPolicyId,
        RetrievalPreset,
        RetrievalPromptId,
        RetrievalToolId,
    )
    from TA.retrieval.policy import resolve_retrieval_context

    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            course_scope="Bench_MuSiQue",
            harness_id=RetrievalHarnessId.AGENTIC_V1,
        )
    )

    assert context.policy.id is RetrievalPolicyId.BASELINE_V3
    assert context.policy.prompt_id is RetrievalPromptId.MULTIHOP_V1
    assert context.policy.model_profile == "RAG"
    assert context.policy.model_name == "qwen3:8b"
    assert context.policy.temperature == 0.0
    assert context.policy.answer_model_profile == "TA"
    assert context.policy.answer_model_name == "gpt-oss:120b-cloud"
    assert context.policy.answer_temperature == 0.0
    assert context.policy.allowed_tools == (
        RetrievalToolId.SEMANTIC,
        RetrievalToolId.TEXTBOOK,
    )
    assert context.policy.seed_tools == (
        RetrievalToolId.SEMANTIC,
        RetrievalToolId.TEXTBOOK,
    )
    assert context.policy.min_tool_calls == 2
    assert context.policy.empty_hits_valid is True
    assert context.harness.id is RetrievalHarnessId.AGENTIC_V1
    assert context.harness.max_tool_calls == 6
    assert context.harness.recursion_limit == 20
    assert context.harness.top_k == 5
    assert context.preset is RetrievalPreset.FULL
    assert context.scope.course == "Bench_MuSiQue"


@pytest.mark.parametrize(
    ("preset", "tools", "minimum"),
    [
        ("RAG", ("semantic",), 1),
        ("FULL", ("semantic", "textbook"), 2),
    ],
)
def test_policy_owns_arm_mapping(preset, tools, minimum):
    from core.config import Retrieve_param
    from TA.retrieval.policy import resolve_retrieval_context

    context = resolve_retrieval_context(Retrieve_param.from_preset(preset))

    assert tuple(tool.value for tool in context.policy.allowed_tools) == tools
    assert tuple(tool.value for tool in context.policy.seed_tools) == tools
    assert context.policy.min_tool_calls == minimum


def test_fanout_rules_are_versioned_separately():
    from core.config import Retrieve_param
    from core.schema.retrieval import RetrievalHarnessId
    from TA.retrieval.policy import resolve_retrieval_context

    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            harness_id=RetrievalHarnessId.FANOUT_V1,
        )
    )

    assert context.harness.id is RetrievalHarnessId.FANOUT_V1
    assert context.harness.top_k == 5
    assert context.harness.per_source_k == 8
    assert context.harness.rrf_k == 60
    assert context.harness.max_tool_calls == 0


def test_agentic_v2_reuses_fanout_seed_limits_then_allows_expansion():
    from core.config import Retrieve_param
    from core.schema.retrieval import RetrievalHarnessId
    from TA.retrieval.policy import resolve_retrieval_context

    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            harness_id=RetrievalHarnessId.AGENTIC_V2,
        )
    )

    assert context.harness.per_source_k == 8
    assert context.harness.rrf_k == 60
    assert context.harness.max_tool_calls == 6


def test_agentic_v3_owns_bounded_ledger_rules_and_strong_models():
    from core.config import Retrieve_param
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId, RetrievalPromptId
    from TA.retrieval.policy import resolve_retrieval_context

    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )

    assert context.policy.prompt_id is RetrievalPromptId.MULTIHOP_V2
    assert context.policy.model_profile == "retrieval_aggregator"
    assert context.policy.model_name == "gpt-oss:120b-cloud"
    assert context.policy.answer_model_profile == "retrieval_answerer"
    assert context.policy.answer_model_name == "gpt-oss:120b-cloud"
    assert context.policy.min_tool_calls == 0
    assert "To stop, do not call retrieve_more" in context.policy.prompt
    assert "non-empty sources" in context.policy.prompt
    assert context.harness.max_tool_calls == 4
    assert context.harness.max_calls_per_round == 1
    assert context.harness.per_source_k == 8
    assert context.harness.rrf_k == 60
    assert context.harness.max_context_chars == 160_000
    assert context.harness.evidence_excerpt_chars == 1_000


def test_v4_policy_digest_is_arm_independent_and_pinned():
    from core.config import Retrieve_param
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId
    from TA.retrieval.policy import resolve_retrieval_context

    contexts = [
        resolve_retrieval_context(
            Retrieve_param.from_preset(
                preset,
                policy_id=RetrievalPolicyId.BASELINE_V4,
                harness_id=RetrievalHarnessId.AGENTIC_V3,
            )
        )
        for preset in ("RAG", "FULL")
    ]

    assert contexts[0].policy.digest == contexts[1].policy.digest
    assert contexts[0].policy.digest == "8a707ff60cd1f207"


def test_agentic_v4_resolves_typed_controller_contract():
    from core.config import Retrieve_param
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId, RetrievalPromptId
    from TA.retrieval.policy import resolve_retrieval_context

    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )

    assert context.policy.prompt_id is RetrievalPromptId.MULTIHOP_V3
    assert context.policy.model_profile == "retrieval_planner"
    assert context.policy.model_name == "gpt-oss:120b-cloud"
    assert context.policy.model_timeout_s == 120
    assert context.policy.model_transport_retries == 1
    assert context.policy.schema_repair_attempts == 1
    assert context.policy.answer_model_profile == "retrieval_answerer"
    assert context.policy.answer_timeout_s == 120
    assert context.policy.answer_transport_retries == 1
    assert context.harness.max_tool_calls == 4
    assert context.harness.max_calls_per_round == 1
    assert context.harness.per_source_k == 8
    assert context.harness.max_context_chars == 160_000


def test_v5_policy_digest_is_arm_independent_and_versioned():
    from core.config import Retrieve_param
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId
    from TA.retrieval.policy import resolve_retrieval_context

    contexts = [
        resolve_retrieval_context(
            Retrieve_param.from_preset(
                preset,
                policy_id=RetrievalPolicyId.BASELINE_V5,
                harness_id=RetrievalHarnessId.AGENTIC_V4,
            )
        )
        for preset in ("RAG", "FULL")
    ]

    assert contexts[0].policy.digest == contexts[1].policy.digest
    assert contexts[0].policy.digest == "100fcfe5e40adc59"


def test_policy_digest_is_pinned():
    from core.config import Retrieve_param
    from TA.retrieval.policy import resolve_retrieval_context

    context = resolve_retrieval_context(Retrieve_param.from_preset("FULL"))

    assert context.policy.digest == "305264a7f1fe7c5c"


def test_resolved_context_is_immutable():
    from core.config import Retrieve_param
    from TA.retrieval.policy import resolve_retrieval_context

    context = resolve_retrieval_context(Retrieve_param.from_preset("RAG"))

    with pytest.raises(FrozenInstanceError):
        context.scope.course = "other"


def test_unknown_contract_id_is_rejected():
    from core.config import Retrieve_param
    from TA.retrieval.policy import resolve_retrieval_context

    with pytest.raises(ValueError, match="policy_id"):
        resolve_retrieval_context(Retrieve_param(policy_id="missing"))


def test_removed_plain_preset_is_rejected():
    from core.config import Retrieve_param

    with pytest.raises(ValueError, match="plain"):
        Retrieve_param.from_preset("PLAIN")
