import asyncio
from types import SimpleNamespace

from core.config import Retrieve_param
from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId
from TA.retrieval.policy import resolve_retrieval_context


def _chunk(uri, source, score=1.0):
    return {"id": uri, "uri": uri, "text": uri, "score": score, "source": source}


def _v3_context(preset="FULL"):
    return resolve_retrieval_context(
        Retrieve_param.from_preset(
            preset,
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )


def test_round_rrf_orders_but_never_truncates_unique_evidence():
    from TA.tools.retrieval import rank_round

    semantic = [_chunk(f"s{i}", "semantic") for i in range(8)]
    textbook = [_chunk("s7", "textbook"), *[_chunk(f"t{i}", "textbook") for i in range(7)]]

    ranked = rank_round({"semantic": semantic, "textbook": textbook}, rrf_k=60)

    assert len(ranked) == 15
    assert ranked[0]["uri"] == "s7"
    assert {item["uri"] for item in ranked} == {
        *(f"s{i}" for i in range(8)),
        *(f"t{i}" for i in range(7)),
    }
    assert ranked[0]["source_ranks"] == {"semantic": 8, "textbook": 1}


def test_ledger_keeps_first_seen_order_and_records_repeat_occurrences():
    from TA.tools.retrieval import append_ledger

    first = rank = [
        {**_chunk("gold-a", "semantic"), "rrf_score": 0.2, "source_ranks": {"semantic": 1}},
        {**_chunk("gold-b", "textbook"), "rrf_score": 0.1, "source_ranks": {"textbook": 1}},
    ]
    ledger, new_count, duplicate_count = append_ledger([], first, round_index=0, query="original")
    second = [
        {**_chunk("gold-a", "textbook"), "rrf_score": 0.3, "source_ranks": {"textbook": 2}},
        {**_chunk("new-c", "semantic"), "rrf_score": 0.2, "source_ranks": {"semantic": 1}},
    ]

    ledger, second_new, second_duplicates = append_ledger(
        ledger, second, round_index=1, query="focused"
    )

    assert (new_count, duplicate_count) == (2, 0)
    assert (second_new, second_duplicates) == (1, 1)
    assert [item["uri"] for item in ledger] == ["gold-a", "gold-b", "new-c"]
    assert ledger[0]["first_seen_round"] == 0
    assert [item["query"] for item in ledger[0]["occurrences"]] == ["original", "focused"]


class _Source:
    def __init__(self, name):
        self.name = name
        self.calls = []

    async def _arun(self, query, runtime):
        self.calls.append(query)
        chunk = _chunk(f"{self.name}-{query}", self.name)
        return self.name, {
            "tool": f"{self.name}_search",
            "source": self.name,
            "args": {"query": query},
            "chunks": [chunk],
            "latency_ms": 1.0,
            "error": "",
        }


def test_deep_tool_runs_model_selected_allowed_sources_only():
    from TA.tools.retrieval import RetrieveMore

    semantic = _Source("semantic")
    textbook = _Source("textbook")
    tool = RetrieveMore(semantic=semantic, textbook=textbook)

    _, artifact = asyncio.run(
        tool._arun(
            query="bridge",
            sources=["textbook"],
            runtime=SimpleNamespace(context=_v3_context("FULL")),
        )
    )

    assert semantic.calls == []
    assert textbook.calls == ["bridge"]
    assert artifact["requested_sources"] == ["textbook"]
    assert artifact["executed_sources"] == ["textbook"]
    assert artifact["invalid_sources"] == []


def test_deep_tool_reports_out_of_arm_source_without_repository_call():
    from TA.tools.retrieval import RetrieveMore

    semantic = _Source("semantic")
    textbook = _Source("textbook")
    tool = RetrieveMore(semantic=semantic, textbook=textbook)

    content, artifact = asyncio.run(
        tool._arun(
            query="bridge",
            sources=["textbook"],
            runtime=SimpleNamespace(context=_v3_context("RAG")),
        )
    )

    assert semantic.calls == [] and textbook.calls == []
    assert artifact["executed_sources"] == []
    assert artifact["invalid_sources"] == ["textbook"]
    assert artifact["error"] == ""
    assert "allowed" in content.lower()
