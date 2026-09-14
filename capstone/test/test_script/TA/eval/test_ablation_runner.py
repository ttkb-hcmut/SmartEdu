import asyncio
import pytest
import json
from unittest.mock import Mock

from TA.tracing.schema import ChatTrace, TraceSession
from test.eval.run_ablation import (
    build_run_state,
    classify_provider_error,
    harness_run_ids,
    harness_policy_id,
    load_run_state,
    paired_schedule,
    pending_schedule,
    record_run_attempt,
    repeat_fixture,
    run_resumable_schedule,
    save_run_state,
    session_name,
    summarize_run_state,
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


def test_validate_run_sessions_allows_missing_cases_only_for_partial_report():
    chats = validate_run_sessions(
        [_session("q1")],
        expected_ids={"q1", "q2"},
        run_id="run-1",
        allow_partial=True,
    )

    assert [chat.question_id for chat in chats] == ["q1"]
    with pytest.raises(ValueError, match="extra"):
        validate_run_sessions(
            [_session("q3")],
            expected_ids={"q1", "q2"},
            run_id="run-1",
            allow_partial=True,
        )


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


def test_agentic_v4_selects_v5_policy():
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId

    assert harness_policy_id(RetrievalHarnessId.AGENTIC_V4) is RetrievalPolicyId.BASELINE_V5


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


def test_paired_schedule_alternates_arm_order_per_fixture():
    fixture = [{"id": "q0"}, {"id": "q1"}, {"id": "q2"}]

    schedule = paired_schedule(fixture, ["RAG", "FULL"])

    assert [(case["question_id"], case["preset"]) for case in schedule] == [
        ("q0", "RAG"),
        ("q0", "FULL"),
        ("q1", "FULL"),
        ("q1", "RAG"),
        ("q2", "RAG"),
        ("q2", "FULL"),
    ]


@pytest.mark.parametrize(
    ("message", "category", "pause"),
    [
        ("Ollama status code: 429 session usage limit reached", "session_usage_limit", True),
        ("status code: 503 upstream unavailable", "transient", False),
        ("TimeoutError: model timed out", "transient", False),
        ("status code: 401 unauthorized", "permanent_4xx", False),
        ("repository unavailable", "execution", False),
    ],
)
def test_provider_error_classification(message, category, pause):
    classified = classify_provider_error(message)

    assert classified["category"] == category
    assert classified["pause_immediately"] is pause


def _run_contract():
    return build_run_state(
        run_id="run-v4",
        fixture=[{"id": "q0"}, {"id": "q1"}],
        presets=["RAG", "FULL"],
        fixture_digest="fixture-a",
        corpus_digest="corpus-a",
        policy_digests={"RAG": "policy-rag", "FULL": "policy-full"},
        harness_id="agentic-v4",
        harness_digest="harness-a",
        code_digest="code-a",
    )


def test_run_state_resume_rejects_any_contract_drift(tmp_path):
    path = tmp_path / "run_state.json"
    state = _run_contract()
    save_run_state(path, state)

    assert load_run_state(path, state, resume=True)["run_id"] == "run-v4"
    drifted = {**state, "code_digest": "code-b"}
    with pytest.raises(ValueError, match="code_digest"):
        load_run_state(path, drifted, resume=True)


def test_resume_skips_only_successful_pairs_and_keeps_failed_attempts():
    state = _run_contract()
    first, second = state["schedule"][:2]
    record_run_attempt(state, first, status="SUCCESS", errors=[])
    record_run_attempt(
        state,
        second,
        status="FAIL",
        errors=["status code: 503"],
        provider_error={"category": "transient", "pause_immediately": False},
    )

    pending = pending_schedule(state)

    assert first not in pending
    assert second in pending
    assert state["successful_cases"] == [first["case_key"]]
    assert state["failed_attempts"][0]["case_key"] == second["case_key"]


def test_run_state_summary_counts_planned_arms_and_provider_attempts():
    state = _run_contract()
    record_run_attempt(
        state,
        state["schedule"][0],
        status="FAIL",
        errors=["status code: 429 session usage limit"],
        provider_error={"category": "session_usage_limit", "pause_immediately": True},
    )
    state["status"] = "PAUSED"

    summary = summarize_run_state(state)

    assert summary["expected_counts"] == {"RAG": 2, "FULL": 2}
    assert summary["provider_errors"] == {"session_usage_limit": 1}
    assert summary["partial"] is True


def test_resumable_scheduler_pauses_on_quota_and_resumes_failed_pair(monkeypatch, tmp_path):
    import test.eval.run_ablation as runner

    state = _run_contract()
    path = tmp_path / "run_state.json"
    calls = []

    async def quota_run_case(*_args, preset, item, **_kwargs):
        calls.append((item["id"], preset))
        if len(calls) == 2:
            return {
                "status": "FAIL",
                "errors": ["Ollama status code: 429 session usage limit reached"],
                "session_id": "failed-session",
            }
        return {"status": "SUCCESS", "errors": [], "session_id": "ok-session"}

    monkeypatch.setattr(runner, "run_case", quota_run_case)
    completed = asyncio.run(run_resumable_schedule(
        ta=object(),
        tracker=object(),
        fixture=[{"id": "q0"}, {"id": "q1"}],
        track="musique",
        course="Bench",
        run_id="run-v4",
        harness_id=object(),
        code_state=object(),
        state=state,
        state_path=path,
    ))

    assert completed is False
    assert calls == [("q0", "RAG"), ("q0", "FULL")]
    paused = json.loads(path.read_text(encoding="utf-8"))
    assert paused["status"] == "PAUSED"
    assert paused["pause_reason"] == "session_usage_limit"

    resumed_calls = []

    async def successful_run_case(*_args, preset, item, **_kwargs):
        resumed_calls.append((item["id"], preset))
        return {"status": "SUCCESS", "errors": [], "session_id": "ok-session"}

    monkeypatch.setattr(runner, "run_case", successful_run_case)
    completed = asyncio.run(run_resumable_schedule(
        ta=object(),
        tracker=object(),
        fixture=[{"id": "q0"}, {"id": "q1"}],
        track="musique",
        course="Bench",
        run_id="run-v4",
        harness_id=object(),
        code_state=object(),
        state=paused,
        state_path=path,
    ))

    assert completed is True
    assert resumed_calls == [("q0", "FULL"), ("q1", "FULL"), ("q1", "RAG")]
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "COMPLETED"


def test_resumable_scheduler_pauses_after_three_consecutive_transient_failures(monkeypatch, tmp_path):
    import test.eval.run_ablation as runner

    state = _run_contract()
    path = tmp_path / "run_state.json"
    calls = []

    async def transient_run_case(*_args, preset, item, **_kwargs):
        calls.append((item["id"], preset))
        return {
            "status": "FAIL",
            "errors": ["status code: 503 upstream unavailable"],
            "session_id": "failed-session",
        }

    monkeypatch.setattr(runner, "run_case", transient_run_case)

    completed = asyncio.run(run_resumable_schedule(
        ta=object(),
        tracker=object(),
        fixture=[{"id": "q0"}, {"id": "q1"}],
        track="musique",
        course="Bench",
        run_id="run-v4",
        harness_id=object(),
        code_state=object(),
        state=state,
        state_path=path,
    ))

    assert completed is False
    assert len(calls) == 3
    assert state["pause_reason"] == "three_consecutive_transient_failures"


def test_resumable_scheduler_marks_partial_when_no_case_succeeds(monkeypatch, tmp_path):
    import test.eval.run_ablation as runner

    state = _run_contract()
    path = tmp_path / "run_state.json"

    async def failing_run_case(*_args, preset, item, **_kwargs):
        return {
            "status": "FAIL",
            "errors": ["execution failed"],
            "session_id": "failed-session",
            "latency_s": 0.5,
        }

    monkeypatch.setattr(runner, "run_case", failing_run_case)

    completed = asyncio.run(run_resumable_schedule(
        ta=object(),
        tracker=object(),
        fixture=[{"id": "q0"}, {"id": "q1"}],
        track="musique",
        course="Bench",
        run_id="run-v4",
        harness_id=object(),
        code_state=object(),
        state=state,
        state_path=path,
    ))

    # the loop finished the whole schedule -- it was never paused --
    # but zero cases succeeded, so completion must not read as COMPLETED
    assert completed is True
    assert state["successful_cases"] == []
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["status"] == "PARTIAL"
    assert persisted["pause_reason"] == "incomplete_cases"


def test_score_emits_availability_report_when_zero_rows_survive(monkeypatch, tmp_path):
    import test.eval.run_ablation as runner

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    results_dir = tmp_path / "results"
    monkeypatch.setattr(runner, "TRACE_DIR", trace_dir)
    monkeypatch.setattr(runner, "RESULTS_DIR", results_dir)

    state = _run_contract()
    for case in state["schedule"]:
        record_run_attempt(
            state, case, status="FAIL", errors=["execution failed"], latency_s=1.5,
        )
    state["status"] = "PARTIAL"

    runner.score(
        fixture=[{"id": "q0"}, {"id": "q1"}],
        presets=["RAG", "FULL"],
        track="musique",
        harness_runs={"agentic-v4": "run-v4"},
        base_run_id="run-v4",
        judge=False,
        run_states={"agentic-v4": state},
    )

    reports = list(results_dir.glob("ablation_musique_run-v4_*.md"))
    assert reports
    assert "PARTIAL EXECUTION" in reports[0].read_text(encoding="utf-8")
