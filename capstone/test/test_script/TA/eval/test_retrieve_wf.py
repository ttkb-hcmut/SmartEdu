"""Fan-out retrieve workflow — routing, fusion, chunk normalization. No DBs."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from core.config import Retrieve_param
from core.schema.retrieval import RetrievalHarnessId, RetrievalPreset
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


def test_plain_skips_components():
    out, nodes = _invoke("PLAIN")
    assert nodes == ["Retrieve_Dispatch", "Fusion"]
    assert out["worker_results"]["RAG"]["status"] == "SUCCESS"
    assert out["worker_results"]["RAG"]["content"] == ""


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
    for name in ["PLAIN", "RAG", "FULL"]:
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


def _invoke_agentic(preset, artifacts):
    agent = _FakeRagAgent(artifacts)
    wf = build_retrieve_wf(agents={"RAG": agent}, resources=None)
    state = {
        "messages": [HumanMessage(content="q")],
        "user_query": "q",
        "worker_results": {},
        "language": "eng",
    }
    context = resolve_retrieval_context(Retrieve_param.from_preset(preset))
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
    context = resolve_retrieval_context(Retrieve_param.from_preset("RAG"))

    out = asyncio.run(wf.ainvoke(state, context=context))

    assert out["worker_results"]["RAG"]["validity"] == "invalid"
    assert "model mismatch" in out["worker_results"]["RAG"]["errors"][0]


def test_harnesses_return_identical_rag_envelopes():
    agentic, _ = _invoke_agentic("PLAIN", [])
    fanout, _ = _invoke("PLAIN")

    assert set(agentic["worker_results"]["RAG"]) == set(fanout["worker_results"]["RAG"])
