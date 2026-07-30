import pytest
import json
from unittest.mock import Mock

from TA.tracing.schema import ChatTrace, TraceSession
from test.eval.run_ablation import session_name, validate_run_sessions, verify_corpus_ready


def _session(question_id, *, run_id="run-1", status="SUCCESS", warmup=False):
    return TraceSession(
        session_id="s",
        chat=[
            ChatTrace(
                chat_id="c",
                query="q",
                question_id=question_id,
                run_id=run_id,
                status=status,
                warmup=warmup,
            )
        ],
    )


def test_session_name_contains_exact_run_identity():
    assert session_name("FULL", "musique", "run-1") == "bench_full_musique_run-1"


def test_validate_run_sessions_requires_every_case_once():
    chats = validate_run_sessions(
        [_session("q1"), _session("q2")],
        expected_ids={"q1", "q2"},
        run_id="run-1",
    )

    assert {chat.question_id for chat in chats} == {"q1", "q2"}


@pytest.mark.parametrize(
    "sessions",
    [
        [_session("q1")],
        [_session("q1"), _session("q1")],
        [_session("q1"), _session("q2", status="FAIL")],
        [_session("q1"), _session("q2", warmup=True)],
        [_session("q1"), _session("q2", run_id="stale")],
    ],
)
def test_validate_run_sessions_rejects_incomplete_or_invalid_runs(sessions):
    with pytest.raises(ValueError):
        validate_run_sessions(sessions, expected_ids={"q1", "q2"}, run_id="run-1")


def test_corpus_gate_requires_canonical_ids_in_both_indexes(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"paragraphs": {"Bench/p1": {}, "Bench/p2": {}}}),
        encoding="utf-8",
    )
    milvus = Mock()
    graph = Mock()
    milvus.list_ids.return_value = {"Bench/p1", "Bench/p2"}
    graph.list_passage_uris.return_value = {"Bench/p1", "Bench/p2"}

    verify_corpus_ready(milvus, graph, manifest, "Bench")

    graph.list_passage_uris.return_value = {"Bench/p1"}
    with pytest.raises(ValueError, match="Neo4j missing 1"):
        verify_corpus_ready(milvus, graph, manifest, "Bench")
