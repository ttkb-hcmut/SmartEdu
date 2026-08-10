import pytest
import json
from unittest.mock import Mock

from TA.tracing.schema import ChatTrace, TraceSession
from test.eval.run_ablation import (
    harness_run_ids,
    harness_policy_id,
    repeat_fixture,
    session_name,
    validate_run_sessions,
    verify_corpus_ready,
    write_run_manifest,
)


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
        [_session("q1"), _session("q2", warmup=True)],
        [_session("q1"), _session("q2", run_id="stale")],
    ],
)
def test_validate_run_sessions_rejects_incomplete_or_invalid_runs(sessions):
    with pytest.raises(ValueError):
        validate_run_sessions(sessions, expected_ids={"q1", "q2"}, run_id="run-1")


def test_validate_run_sessions_keeps_failed_cases_for_measurement():
    chats = validate_run_sessions(
        [_session("q1"), _session("q2", status="FAIL")],
        expected_ids={"q1", "q2"},
        run_id="run-1",
    )

    assert [chat.status for chat in chats] == ["SUCCESS", "FAIL"]


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


def test_agentic_v3_selects_v4_policy_while_controls_remain_v3():
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId

    assert harness_policy_id(RetrievalHarnessId.AGENTIC_V3) is RetrievalPolicyId.BASELINE_V4
    assert harness_policy_id(RetrievalHarnessId.AGENTIC_V2) is RetrievalPolicyId.BASELINE_V3
    assert harness_policy_id(RetrievalHarnessId.FANOUT_V1) is RetrievalPolicyId.BASELINE_V3


def test_multi_harness_run_ids_are_isolated_but_single_run_stays_compatible():
    assert harness_run_ids("run", ["agentic-v3"]) == {"agentic-v3": "run"}
    assert harness_run_ids("run", ["agentic-v2", "agentic-v3"]) == {
        "agentic-v2": "run__agentic-v2",
        "agentic-v3": "run__agentic-v3",
    }


def test_repeat_fixture_assigns_unique_case_ids_without_changing_gold():
    fixture = [{"id": "q1", "question": "q", "gold_chunk_ids": ["gold"]}]

    repeated = repeat_fixture(fixture, 3)

    assert [item["id"] for item in repeated] == ["q1__repeat_1", "q1__repeat_2", "q1__repeat_3"]
    assert all(item["source_fixture_id"] == "q1" for item in repeated)
    assert all(item["gold_chunk_ids"] == ["gold"] for item in repeated)


def test_run_manifest_links_corpus_digest_and_ingestion_report(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text('{"paragraphs":{"p":{}}}', encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    ingestion = results / "ingest_Bench_20260810.json"
    ingestion.write_text(
        json.dumps({"status": "COMPLETED", "corpus": str(corpus)}),
        encoding="utf-8",
    )
    newer_wrong_corpus = results / "ingest_Bench_20260811.json"
    newer_wrong_corpus.write_text(
        json.dumps({"status": "COMPLETED", "corpus": str(tmp_path / "other")}),
        encoding="utf-8",
    )

    path = write_run_manifest(
        results,
        run_id="run",
        fixture=[{"id": "q1"}],
        corpus_dir=corpus,
        course="Bench",
        harness_runs={"agentic-v2": "run__agentic-v2", "agentic-v3": "run__agentic-v3"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["corpus_manifest_sha256"]
    assert payload["ingestion_report"] == str(ingestion)
    assert payload["harness_runs"]["agentic-v3"] == "run__agentic-v3"
