"""Deterministic evaluator metrics — pure functions over trace JSON."""

import pytest

from TA.tracing.schema import ChatTrace, StepTrace, TraceSession
from TA.tracing.evaluator import (
    context_scores,
    trajectory_scores,
    evaluate_chat,
    evaluate_session,
    render_table,
)

GOLD = ["Bench_MuSiQue/q1/2", "Bench_MuSiQue/q1/16"]


def _chunk(uri, source="rag"):
    return {"id": uri, "uri": uri, "text": "t", "score": 1.0, "source": source}


def _chat(preset="FULL", comp_steps=None, fusion_chunks=None):
    steps = []
    for node, uris in (comp_steps or {}).items():
        steps.append(StepTrace(node=node, chunks=[_chunk(u) for u in uris], latency_ms=100.0))
    steps.append(StepTrace(node="Fusion", chunks=[_chunk(u) for u in (fusion_chunks or [])], latency_ms=10.0))
    return ChatTrace(chat_id="c1", query="What is q1?", agent=steps,
                     final_output="ans", preset=preset,
                     retrieve_flags={"rag": True, "graphrag": preset == "FULL"})


# ── context_scores ────────────────────────────────────────────────────

def test_context_perfect():
    s = context_scores(GOLD, GOLD)
    assert s["context_precision"] == 1.0 and s["context_recall"] == 1.0


def test_context_partial():
    s = context_scores([GOLD[0], "Bench_MuSiQue/q1/5"], GOLD)
    assert s["context_precision"] == 0.5
    assert s["context_recall"] == 0.5


def test_context_empty_retrieval():
    s = context_scores([], GOLD)
    assert s["context_precision"] == 0.0 and s["context_recall"] == 0.0


def test_context_no_gold_is_none():
    s = context_scores(GOLD, [])
    assert s["context_precision"] is None and s["context_recall"] is None


# ── trajectory_scores ─────────────────────────────────────────────────

def test_trajectory_all_components_useful():
    chat = _chat(comp_steps={"Comp_RAG": [GOLD[0]], "Comp_GraphRAG": [GOLD[1]]},
                 fusion_chunks=GOLD)
    t = trajectory_scores(chat, GOLD)
    assert t["traj_precision"] == 1.0
    assert t["traj_recall"] == 1.0
    assert t["latency_ms"] == pytest.approx(210.0)


def test_trajectory_wasted_hop_punished():
    chat = _chat(comp_steps={"Comp_RAG": [GOLD[0]], "Comp_GraphRAG": ["Bench_MuSiQue/q1/9"]},
                 fusion_chunks=[GOLD[0]])
    t = trajectory_scores(chat, GOLD)
    assert t["traj_precision"] == 0.5   # graphrag call contributed no gold
    assert t["traj_recall"] == 0.5      # only 1 of 2 gold gathered


def test_trajectory_plain_has_no_calls():
    chat = _chat(preset="PLAIN", comp_steps={}, fusion_chunks=[])
    t = trajectory_scores(chat, GOLD)
    assert t["traj_precision"] is None and t["traj_recall"] == 0.0


# ── evaluate_chat / evaluate_session ──────────────────────────────────

def test_evaluate_chat_row():
    chat = _chat(comp_steps={"Comp_RAG": GOLD}, fusion_chunks=GOLD)
    row = evaluate_chat(chat, {"id": "q1", "question": "What is q1?",
                               "gold_answer": "a", "gold_chunk_ids": GOLD})
    assert row["preset"] == "FULL" and row["fixture_id"] == "q1"
    assert row["context_recall"] == 1.0


def test_evaluate_session_matches_by_query():
    session = TraceSession(session_id="bench_full_musique", chat=[
        _chat(comp_steps={"Comp_RAG": GOLD}, fusion_chunks=GOLD)])
    fixture = [{"id": "q1", "question": "What is q1?", "gold_answer": "a",
                "gold_chunk_ids": GOLD},
               {"id": "q2", "question": "unmatched?", "gold_answer": "b",
                "gold_chunk_ids": []}]
    rows = evaluate_session(session, fixture)
    assert len(rows) == 1 and rows[0]["fixture_id"] == "q1"


def test_evaluate_session_matches_case_identity_and_skips_warmup():
    benchmark = _chat(comp_steps={"Comp_Semantic": GOLD}, fusion_chunks=GOLD)
    benchmark.question_id = "q2"
    benchmark.query = "duplicate question"
    warmup = _chat(comp_steps={"Comp_Semantic": GOLD}, fusion_chunks=GOLD)
    warmup.question_id = "q1"
    warmup.query = "duplicate question"
    warmup.warmup = True
    session = TraceSession(session_id="run", chat=[warmup, benchmark])
    fixture = [
        {"id": "q1", "question": "duplicate question", "gold_chunk_ids": []},
        {"id": "q2", "question": "duplicate question", "gold_chunk_ids": GOLD},
    ]

    rows = evaluate_session(session, fixture)

    assert [row["fixture_id"] for row in rows] == ["q2"]


def test_agentic_result_chunks_are_scored():
    chat = ChatTrace(
        chat_id="c1",
        query="q",
        preset="RAG",
        agent=[StepTrace(node="Agentic_Retrieve", chunks=[_chunk(GOLD[0])])],
    )

    row = evaluate_chat(chat, {"id": "q", "question": "q", "gold_chunk_ids": GOLD})

    assert row["context_precision"] == 1.0
    assert row["context_recall"] == 0.5


def test_render_table_groups_presets():
    rows = [
        {"preset": "PLAIN", "fixture_id": "q1", "context_precision": None,
         "context_recall": 0.0, "traj_precision": None, "traj_recall": 0.0,
         "latency_ms": 10.0},
        {"preset": "FULL", "fixture_id": "q1", "context_precision": 1.0,
         "context_recall": 1.0, "traj_precision": 1.0, "traj_recall": 1.0,
         "latency_ms": 210.0},
    ]
    table = render_table(rows)
    assert "PLAIN" in table and "FULL" in table and "1.00" in table
