"""Fan-out retrieve workflow — routing, fusion, chunk normalization. No DBs."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.config import Retrieve_param
from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId, RetrievalPreset
from TA.helper.schema import RAGCore
from TA.retrieval.policy import resolve_retrieval_context
from TA.workflow.retrieve import build_retrieve_wf, rrf_merge, _norm_chunk
from TA.tracing.tracer import AgentTracer


# ── rrf_merge ─────────────────────────────────────────────────────────

def _c(uri, source="rag"):
    return {"id": uri, "uri": uri, "text": "t", "score": 1.0, "source": source}


def test_rrf_cross_pool_agreement_wins():
    merged = rrf_merge({"semantic": [_c("a"), _c("b")],
                        "textbook": [_c("b", "textbook"), _c("c", "textbook")]},
                       rrf_k=60, top_k=5)
    assert [c["uri"] for c in merged][0] == "b"
    assert len(merged) == 3  # deduped by uri


def test_rrf_empty_pools():
    assert rrf_merge({}, rrf_k=60, top_k=5) == []
    assert rrf_merge({"rag": []}, rrf_k=60, top_k=5) == []


def test_rrf_caps_at_top_k():
    pool = {"rag": [_c(f"u{i}") for i in range(10)]}
    assert len(rrf_merge(pool, rrf_k=60, top_k=3)) == 3


def test_norm_chunk_uri_falls_back_to_id():
    n = _norm_chunk({"id": "x1", "text": "t", "score": "0.5"}, "rag")
    assert n["uri"] == "x1" and n["score"] == 0.5 and n["source"] == "rag"


# ── graph routing per preset (components no-op without resources) ─────

def _invoke(preset):
    wf = build_retrieve_wf(agents={}, resources=None)
    tracer = AgentTracer(session_id=f"test_wf_{preset}")
    chat_id = tracer.begin_chat("q")
    state = {"messages": [HumanMessage(content="q")], "user_query": "q",
             "worker_results": {}, "language": "eng"}
    cfg = {"configurable": {"tracer": tracer, "chat_id": chat_id}}
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            preset,
            harness_id=RetrievalHarnessId.FANOUT_V1,
        )
    )
    out = asyncio.run(wf.ainvoke(state, config=cfg, context=context))
    nodes = [s.node for s in tracer._active_chats[chat_id].agent]
    return out, nodes


def test_rag_runs_single_component():
    _, nodes = _invoke("RAG")
    assert "Comp_Semantic" in nodes and "Comp_Textbook" not in nodes
    assert nodes.count("Fusion") == 1


def test_full_fans_out_both_components():
    _, nodes = _invoke("FULL")
    assert "Comp_Semantic" in nodes and "Comp_Textbook" in nodes
    assert nodes.count("Fusion") == 1  # fan-in runs fusion once


# ── preset semantics ──────────────────────────────────────────────────

def test_presets_round_trip():
    for name in ["RAG", "FULL"]:
        assert Retrieve_param.from_preset(name).preset is RetrievalPreset[name]


def test_preset_keeps_scope_selection_only():
    rp = Retrieve_param.from_preset("RAG", course_scope="Bench_MuSiQue")
    assert rp.preset is RetrievalPreset.RAG
    assert rp.course_scope == "Bench_MuSiQue"


class _FakeRagAgent:
    name = "RAG"
    model = SimpleNamespace(model="qwen3:8b", temperature=0.0)

    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    async def ainvoke(self, state, config, context):
        self.calls.append((state, config, context))
        messages = [
            ToolMessage(
                content="result",
                tool_call_id=f"call-{index}",
                name=artifact["tool"],
                artifact=artifact,
            )
            for index, artifact in enumerate(self.artifacts)
        ]
        return {
            "messages": messages,
            "structured_response": RAGCore(
                thought="agent thought",
                entity_ids=["entity"],
                content="agent content",
                status="FAIL",
            ),
        }


class _FakeAggregator:
    name = "RAG_Aggregator"
    model = SimpleNamespace(model="qwen3:8b", temperature=0.0)

    def __init__(self, artifacts=()):
        self.artifacts = artifacts
        self.calls = []

    async def ainvoke(self, state, config, context):
        self.calls.append((state, config, context))
        return {
            "messages": [
                ToolMessage(
                    content="result",
                    tool_call_id=f"call-{index}",
                    name=artifact["tool"],
                    artifact=artifact,
                )
                for index, artifact in enumerate(self.artifacts)
            ]
        }


class _FakeLedgerAggregator:
    name = "RAG_Ledger_Aggregator"
    model = SimpleNamespace(model="gpt-oss:120b-cloud", temperature=0.0)

    def __init__(self, artifacts=(), blocked_calls=0):
        self.artifacts = artifacts
        self.blocked_calls = blocked_calls
        self.calls = []

    async def ainvoke(self, state, config, context):
        self.calls.append((state, config, context))
        messages = [
            ToolMessage(
                content="follow-up evidence",
                tool_call_id=f"ledger-{index}",
                name="retrieve_more",
                artifact=artifact,
            )
            for index, artifact in enumerate(self.artifacts)
        ]
        messages.append(
            AIMessage(
                content="Scottish Parliament is supported; Holyrood is its location.",
                additional_kwargs={"blocked_retrieval_calls": self.blocked_calls},
            )
        )
        return {"messages": messages}


class _SeedMilvus:
    def search(self, **_):
        return [{"id": "semantic-1", "uri": "semantic-1", "text": "semantic", "score": 1.0}]


class _SeedEmbedder:
    def get_embedding(self, _):
        return [0.0]


class _SeedGraph:
    def passage_search(self, *_args, **_kwargs):
        return [{"id": "textbook-1", "uri": "textbook-1", "text": "textbook", "score": 1.0}]


def _invoke_agentic(preset, artifacts):
    agent = _FakeRagAgent(artifacts)
    wf = build_retrieve_wf(agents={"RAG": agent}, resources=None)
    state = {
        "messages": [HumanMessage(content="q")],
        "user_query": "q",
        "worker_results": {},
        "language": "eng",
    }
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(preset, harness_id=RetrievalHarnessId.AGENTIC_V1)
    )
    out = asyncio.run(wf.ainvoke(state, context=context))
    return out, agent


def test_agentic_harness_injects_context_and_uses_contract_recursion_limit():
    artifact = {
        "tool": "semantic_search",
        "source": "semantic",
        "args": {"query": "q"},
        "chunks": [],
        "error": "",
    }

    out, agent = _invoke_agentic("RAG", [artifact])

    assert agent.calls[0][1]["recursion_limit"] == 20
    assert agent.calls[0][2].harness.id is RetrievalHarnessId.AGENTIC_V1
    assert out["worker_results"]["RAG"]["validity"] == "valid"
    assert out["worker_results"]["RAG"]["status"] == "SUCCESS"
    assert out["worker_results"]["RAG"]["agent_status"] == "FAIL"


def test_agentic_required_arm_without_attempt_is_policy_invalid():
    out, _ = _invoke_agentic("FULL", [])

    assert out["worker_results"]["RAG"]["validity"] == "policy-invalid"
    assert out["worker_results"]["RAG"]["status"] == "FAIL"
    assert out["status_flag"] == "SUCCESS"


def test_agentic_model_mismatch_is_invalid():
    artifact = {
        "tool": "semantic_search",
        "source": "semantic",
        "args": {"query": "q"},
        "chunks": [],
        "error": "",
    }
    agent = _FakeRagAgent([artifact])
    agent.model = SimpleNamespace(model="different-model", temperature=0.0)
    wf = build_retrieve_wf(agents={"RAG": agent}, resources=None)
    state = {
        "messages": [HumanMessage(content="q")],
        "user_query": "q",
        "worker_results": {},
        "language": "eng",
    }
    context = resolve_retrieval_context(
        Retrieve_param.from_preset("RAG", harness_id=RetrievalHarnessId.AGENTIC_V1)
    )

    out = asyncio.run(wf.ainvoke(state, context=context))

    assert out["worker_results"]["RAG"]["validity"] == "invalid"
    assert "model mismatch" in out["worker_results"]["RAG"]["errors"][0]


def test_harnesses_return_identical_rag_envelopes():
    agentic, _ = _invoke_agentic("RAG", [])
    fanout, _ = _invoke("RAG")

    assert set(agentic["worker_results"]["RAG"]) == set(fanout["worker_results"]["RAG"])


def test_agentic_v2_seeds_full_then_allows_tool_only_expansion():
    extra = {
        "tool": "semantic_search",
        "source": "semantic",
        "args": {"query": "focused"},
        "chunks": [],
        "error": "",
    }
    aggregator = _FakeAggregator([extra])
    resources = {
        "milvus_db": _SeedMilvus(),
        "graph_db": _SeedGraph(),
        "embedder": _SeedEmbedder(),
    }
    wf = build_retrieve_wf(agents={"RAG_AGGREGATOR": aggregator}, resources=resources)
    state = {
        "messages": [HumanMessage(content="q")],
        "user_query": "q",
        "worker_results": {},
        "language": "eng",
    }
    context = resolve_retrieval_context(
        Retrieve_param.from_preset("FULL", harness_id=RetrievalHarnessId.AGENTIC_V2)
    )
    tracer = AgentTracer(session_id="agentic-v2")
    chat_id = tracer.begin_chat("q")

    out = asyncio.run(
        wf.ainvoke(
            state,
            context=context,
            config={"configurable": {"tracer": tracer, "chat_id": chat_id}},
        )
    )

    assert aggregator.calls[0][0]["current_node"] == "Retrieval_Aggregator"
    assert aggregator.calls[0][0]["retrieval_call_count"] == 2
    assert "semantic" in aggregator.calls[0][0]["messages"][0][1]
    assert "textbook" in aggregator.calls[0][0]["messages"][0][1]
    assert aggregator.calls[0][1]["recursion_limit"] == 20
    assert out["worker_results"]["RAG"]["retrieval_attempts"] == 3
    assert out["worker_results"]["RAG"]["validity"] == "valid"
    assert out["worker_results"]["RAG"]["status"] == "SUCCESS"
    aggregation = next(step for step in tracer._active_chats[chat_id].agent if step.node == "Retrieval_Aggregator")
    assert aggregation.tool_result["seed_calls"] == 2
    assert aggregation.tool_result["aggregator_calls"] == 1
    assert aggregation.latency_ms >= 0


def test_agentic_v2_accepts_seed_without_extra_model_tool_calls():
    aggregator = _FakeAggregator()
    resources = {
        "milvus_db": _SeedMilvus(),
        "graph_db": _SeedGraph(),
        "embedder": _SeedEmbedder(),
    }
    wf = build_retrieve_wf(agents={"RAG_AGGREGATOR": aggregator}, resources=resources)
    state = {
        "messages": [HumanMessage(content="q")],
        "user_query": "q",
        "worker_results": {},
        "language": "eng",
    }
    context = resolve_retrieval_context(
        Retrieve_param.from_preset("FULL", harness_id=RetrievalHarnessId.AGENTIC_V2)
    )

    out = asyncio.run(wf.ainvoke(state, context=context))

    assert out["worker_results"]["RAG"]["retrieval_attempts"] == 2
    assert out["worker_results"]["RAG"]["validity"] == "valid"


def test_agentic_v3_keeps_seed_and_followup_gold_in_append_only_ledger():
    first_gold = "Bench_MuSiQue/paragraph/d587"
    second_gold = "Bench_MuSiQue/paragraph/b7fdd"
    followup = {
        "tool": "retrieve_more",
        "source": "parallel",
        "args": {"query": "Scottish Parliament official home since 2004", "sources": ["textbook"]},
        "requested_sources": ["textbook"],
        "executed_sources": ["textbook"],
        "invalid_sources": [],
        "source_artifacts": [{
            "tool": "textbook_search",
            "source": "textbook",
            "args": {"query": "Scottish Parliament official home since 2004"},
            "chunks": [
                {"id": second_gold, "uri": second_gold, "text": "Official home is at Holyrood.", "score": 1.0, "source": "textbook"},
                {"id": first_gold, "uri": first_gold, "text": "Matters devolve to the Scottish Parliament.", "score": 0.9, "source": "textbook"},
            ],
            "latency_ms": 2.0,
            "error": "",
        }],
        "chunks": [
            {"id": second_gold, "uri": second_gold, "text": "Official home is at Holyrood.", "score": 1.0,
             "source": "textbook", "rrf_score": 0.2, "source_ranks": {"textbook": 1}, "source_scores": {"textbook": 1.0}},
            {"id": first_gold, "uri": first_gold, "text": "Matters devolve to the Scottish Parliament.", "score": 0.9,
             "source": "textbook", "rrf_score": 0.1, "source_ranks": {"textbook": 2}, "source_scores": {"textbook": 0.9}},
        ],
        "latency_ms": 2.0,
        "error": "",
    }
    aggregator = _FakeLedgerAggregator([followup])
    resources = {
        "milvus_db": _SeedMilvus(),
        "graph_db": type("_Graph", (), {"passage_search": lambda self, *_a, **_k: [
            {"id": first_gold, "uri": first_gold, "text": "Matters devolve to the Scottish Parliament.", "score": 1.0}
        ]})(),
        "embedder": _SeedEmbedder(),
    }
    wf = build_retrieve_wf(
        agents={"RAG_LEDGER_AGGREGATOR": aggregator},
        resources=resources,
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )
    state = {
        "messages": [HumanMessage(content="question")],
        "user_query": "question",
        "worker_results": {},
        "language": "eng",
    }

    out = asyncio.run(wf.ainvoke(state, context=context))
    rag = out["worker_results"]["RAG"]

    assert first_gold in rag["entity_ids"]
    assert second_gold in rag["entity_ids"]
    assert "Holyrood" in rag["content"]
    assert rag["thought"].startswith("Scottish Parliament")
    assert rag["validity"] == "valid"
    assert aggregator.calls[0][0]["current_node"] == "Retrieval_Ledger_Aggregator"


def test_agentic_v3_traces_blocked_parallel_tool_calls():
    aggregator = _FakeLedgerAggregator(blocked_calls=2)
    wf = build_retrieve_wf(
        agents={"RAG_LEDGER_AGGREGATOR": aggregator},
        resources={"milvus_db": _SeedMilvus(), "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )
    tracer = AgentTracer(session_id="blocked-calls")
    chat_id = tracer.begin_chat("question")
    state = {
        "messages": [HumanMessage(content="question")],
        "user_query": "question",
        "worker_results": {},
        "language": "eng",
    }

    asyncio.run(
        wf.ainvoke(
            state,
            config={"configurable": {"tracer": tracer, "chat_id": chat_id}},
            context=context,
        )
    )

    final = next(
        step
        for step in tracer._active_chats[chat_id].agent
        if step.node == "Agentic_Retrieve"
    )
    assert final.tool_result["blocked_tool_calls"] == 2
