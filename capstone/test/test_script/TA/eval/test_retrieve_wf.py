"""Fan-out retrieve workflow — routing, fusion, chunk normalization. No DBs."""

import asyncio

import pytest
from langchain_core.messages import HumanMessage

from core.config import Retrieve_param
from TA.edu.workflow.retrieve import build_retrieve_wf, rrf_merge, _norm_chunk
from TA.tracing.tracer import AgentTracer


# ── rrf_merge ─────────────────────────────────────────────────────────

def _c(uri, source="rag"):
    return {"id": uri, "uri": uri, "text": "t", "score": 1.0, "source": source}


def test_rrf_cross_pool_agreement_wins():
    merged = rrf_merge({"rag": [_c("a"), _c("b")],
                        "graphrag": [_c("b", "graphrag"), _c("c", "graphrag")]},
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
    cfg = {"configurable": {"retrieve_param": Retrieve_param.from_preset(preset),
                            "tracer": tracer, "chat_id": chat_id}}
    out = asyncio.run(wf.ainvoke(state, config=cfg))
    nodes = [s.node for s in tracer._active_chats[chat_id].agent]
    return out, nodes


def test_plain_skips_components():
    out, nodes = _invoke("PLAIN")
    assert nodes == ["Retrieve_Dispatch", "Fusion"]
    assert out["worker_results"]["RAG"]["status"] == "PLAIN"
    assert out["worker_results"]["RAG"]["content"] == ""


def test_rag_runs_single_component():
    _, nodes = _invoke("RAG")
    assert "Comp_RAG" in nodes and "Comp_GraphRAG" not in nodes
    assert nodes.count("Fusion") == 1


def test_full_fans_out_both_components():
    _, nodes = _invoke("FULL")
    assert "Comp_RAG" in nodes and "Comp_GraphRAG" in nodes
    assert nodes.count("Fusion") == 1  # fan-in runs fusion once


# ── preset semantics ──────────────────────────────────────────────────

def test_presets_round_trip():
    for name in ["PLAIN", "RAG", "FULL"]:
        assert Retrieve_param.from_preset(name).preset == name


def test_preset_override_keeps_flags():
    rp = Retrieve_param.from_preset("RAG", benchmark_course="Bench_MuSiQue")
    assert rp.preset == "RAG" and rp.benchmark_course == "Bench_MuSiQue"
