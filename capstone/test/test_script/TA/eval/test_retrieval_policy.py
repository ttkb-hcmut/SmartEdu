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

    assert context.policy.id is RetrievalPolicyId.BASELINE_V1
    assert context.policy.prompt_id is RetrievalPromptId.MULTIHOP_V1
    assert context.policy.model_profile == "RAG"
    assert context.policy.model_name == "qwen3:8b"
    assert context.policy.temperature == 0.0
    assert context.policy.allowed_tools == (
        RetrievalToolId.SEMANTIC,
        RetrievalToolId.TEXTBOOK,
    )
    assert context.policy.min_tool_calls == 1
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
        ("PLAIN", (), 0),
        ("RAG", ("semantic",), 1),
        ("FULL", ("semantic", "textbook"), 1),
    ],
)
def test_policy_owns_arm_mapping(preset, tools, minimum):
    from core.config import Retrieve_param
    from TA.retrieval.policy import resolve_retrieval_context

    context = resolve_retrieval_context(Retrieve_param.from_preset(preset))

    assert tuple(tool.value for tool in context.policy.allowed_tools) == tools
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


def test_policy_digest_is_pinned():
    from core.config import Retrieve_param
    from TA.retrieval.policy import resolve_retrieval_context

    context = resolve_retrieval_context(Retrieve_param.from_preset("FULL"))

    assert context.policy.digest == "653a634fae85be04"


def test_resolved_context_is_immutable():
    from core.config import Retrieve_param
    from TA.retrieval.policy import resolve_retrieval_context

    context = resolve_retrieval_context(Retrieve_param.from_preset("PLAIN"))

    with pytest.raises(FrozenInstanceError):
        context.scope.course = "other"


def test_unknown_contract_id_is_rejected():
    from core.config import Retrieve_param
    from TA.retrieval.policy import resolve_retrieval_context

    with pytest.raises(ValueError, match="policy_id"):
        resolve_retrieval_context(Retrieve_param(policy_id="missing"))
