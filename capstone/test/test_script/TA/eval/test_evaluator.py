"""Deterministic evaluator metrics — pure functions over trace JSON."""

import pytest

from TA.tracing.schema import ChatTrace, StepTrace, TraceSession
from TA.tracing.evaluator import (
    answer_scores,
    context_scores,
    trajectory_scores,
    evaluate_chat,
    evaluate_session,
    render_table,
    render_report,
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

def test_answer_scores_match_musique_style_normalization():
    scores = answer_scores("The Time Warner Cable.", ["Time Warner Cable"])

    assert scores == {"answer_exact_match": 1.0, "answer_f1": 1.0}


def test_answer_scores_choose_best_gold_alias_and_partial_overlap():
    scores = answer_scores("Warner Cable", ["Time Warner Cable", "TWC"])

    assert scores["answer_exact_match"] == 0.0
    assert scores["answer_f1"] == pytest.approx(0.8)


def test_answer_scores_are_none_without_a_gold_answer():
    assert answer_scores("some answer", []) == {
        "answer_exact_match": None,
        "answer_f1": None,
    }


# ── context_scores ────────────────────────────────────────────────────

def test_context_perfect():
    s = context_scores(GOLD, GOLD)
    assert s["context_precision"] == 1.0 and s["context_recall"] == 1.0


def test_context_partial():
    s = context_scores([GOLD[0], "Bench_MuSiQue/q1/5"], GOLD)
    assert s["context_precision"] == 0.5
    assert s["context_recall"] == 0.5
    assert s["support_f1"] == 0.5
    assert s["support_exact_match"] == 0.0
    assert s["complete_chain"] == 0.0


def test_context_empty_retrieval():
    s = context_scores([], GOLD)
    assert s["context_precision"] == 0.0 and s["context_recall"] == 0.0


def test_v3_round_metrics_find_first_gold_and_complete_chain_round():
    from TA.tracing.evaluator import round_scores

    chat = ChatTrace(
        chat_id="c1",
        query="q",
        preset="FULL",
        harness_id="agentic-v3",
        agent=[
            StepTrace(node="Retrieval_Round", tool_result={"round": 0}, chunks=[_chunk(GOLD[0])]),
            StepTrace(node="Retrieval_Round", tool_result={"round": 1}, chunks=[_chunk("distractor")]),
            StepTrace(node="Retrieval_Round", tool_result={"round": 2}, chunks=[_chunk(GOLD[1])]),
            StepTrace(node="Agentic_Retrieve", chunks=[_chunk(uri) for uri in GOLD]),
        ],
    )

    scores = round_scores(chat, GOLD)

    assert scores["first_gold_round"] == 0
    assert scores["complete_chain_round"] == 2
    assert scores["ledger_growth"] == [1, 1, 1]


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
    assert row["answer_exact_match"] == 0.0
    assert row["answer_f1"] == 0.0


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


def test_agentic_call_and_answer_latencies_are_exposed():
    chat = ChatTrace(
        chat_id="c1",
        query="q",
        preset="RAG",
        agent=[
            StepTrace(
                node="Agentic_Retrieve",
                chunks=[_chunk(GOLD[0])],
                tool_result={
                    "seed_calls": 1,
                    "aggregator_calls": 2,
                    "blocked_tool_calls": 3,
                    "aggregator_latency_ms": 15.0,
                },
            ),
            StepTrace(node="TA_Retrieve_Finish", latency_ms=30.0),
        ],
    )

    row = evaluate_chat(chat, {"id": "q", "question": "q", "gold_chunk_ids": GOLD})

    assert row["seed_calls"] == 1
    assert row["aggregator_calls"] == 2
    assert row["blocked_tool_calls"] == 3
    assert row["aggregator_latency_ms"] == 15.0
    assert row["answer_latency_ms"] == 30.0


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


def test_render_report_explains_latency_and_effective_node_config():
    rows = [{
        "preset": "FULL",
        "fixture_id": "q1",
        "status": "FAIL",
        "context_precision": 0.0,
        "context_recall": 0.0,
        "source_hit_rate": 0.5,
        "candidate_recall": 0.5,
        "selection_recall_loss": 0.5,
        "latency_ms": 120.0,
        "workflow_latency_ms": 120.0,
        "turn_latency_ms": 130.0,
        "time_to_first_token_ms": None,
        "node_configs": {
            "RAG_Core": {
                "kind": "llm", "model": "qwen3:8b", "num_ctx": 8192,
                "temperature": 0.0, "max_retries": 0, "harness_id": "agentic-v1",
                "top_k": 5,
            }
        },
    }]

    report = render_report(rows)

    assert "execution completed | execution errors" in report
    assert "Workflow latency is measured with a monotonic clock" in report
    assert "RAG_Core" in report and "qwen3:8b" in report and "8192" in report


def test_paired_harness_summary_uses_question_level_deltas_deterministically():
    from TA.tracing.evaluator import paired_harness_summary

    rows = [
        {"fixture_id": "q1", "preset": "FULL", "harness_id": "agentic-v2", "support_f1": 0.0, "answer_f1": 0.0, "complete_chain": 0.0, "latency_ms": 10.0},
        {"fixture_id": "q1", "preset": "FULL", "harness_id": "agentic-v3", "support_f1": 1.0, "answer_f1": 1.0, "complete_chain": 1.0, "latency_ms": 20.0},
        {"fixture_id": "q2", "preset": "FULL", "harness_id": "agentic-v2", "support_f1": 0.5, "answer_f1": 0.5, "complete_chain": 0.0, "latency_ms": 30.0},
        {"fixture_id": "q2", "preset": "FULL", "harness_id": "agentic-v3", "support_f1": 1.0, "answer_f1": 0.5, "complete_chain": 1.0, "latency_ms": 40.0},
    ]

    first = paired_harness_summary(rows, resamples=200, seed=42)
    second = paired_harness_summary(rows, resamples=200, seed=42)

    assert first == second
    support = next(item for item in first if item["metric"] == "support_f1")
    assert support["n"] == 2
    assert support["mean_delta"] == pytest.approx(0.75)
