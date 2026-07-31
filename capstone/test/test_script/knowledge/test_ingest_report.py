import asyncio
import json

import pytest

from core.schema.ingest import new_report
from knowledge.ingest.persist import persist_report


class _ObjectClient:
    def __init__(self):
        self.objects = {}

    def put_object(self, bucket, object_name, data, length, content_type):
        payload = data.read()
        assert len(payload) == length
        assert content_type == "application/json"
        self.objects[(bucket, object_name)] = payload


class _Storage:
    bucket_name = "courses"

    def __init__(self):
        self.client = _ObjectClient()

    def object_exists(self, object_name):
        return (self.bucket_name, object_name) in self.client.objects

    def get_object_bytes(self, object_name):
        return self.client.objects[(self.bucket_name, object_name)]


def _stored(storage, object_name):
    return json.loads(storage.get_object_bytes(object_name))


def test_terminal_reports_are_immutable_and_latest_rejects_older_run():
    storage = _Storage()
    old = new_report("Course", "run-old", "2026-07-31T01:00:00+00:00")
    new = new_report("Course", "run-new", "2026-07-31T02:00:00+00:00")

    persist_report(storage, "Course", "run-old", old)
    persist_report(storage, "Course", "run-new", new)

    old["status"] = "FAILED"
    old["finished_at"] = "2026-07-31T03:00:00+00:00"
    persist_report(storage, "Course", "run-old", old)

    assert _stored(storage, "Course/_reports/run-old.json")["status"] == "FAILED"
    assert _stored(storage, "Course/_reports/latest.json")["run_id"] == "run-new"

    new["status"] = "COMPLETED"
    new["finished_at"] = "2026-07-31T03:01:00+00:00"
    persist_report(storage, "Course", "run-new", new)
    assert _stored(storage, "Course/_reports/latest.json")["status"] == "COMPLETED"

    with pytest.raises(FileExistsError):
        persist_report(storage, "Course", "run-new", new)


def test_report_reducer_assembles_source_outcomes_without_mutating_running_seed():
    from knowledge.pipeline.reporting import ReportOutcome, reduce_report

    seed = new_report("Course", "run-1", "2026-07-31T01:00:00+00:00")
    decision = reduce_report(
        seed,
        (
            ReportOutcome("slides", value={"file": "good.pdf", "nodes": 2, "edges": 1}),
            ReportOutcome("videos", file_name="bad.mp4", error=ValueError("asr failed")),
            ReportOutcome("anchors", value=2),
        ),
        duration_s=1.2,
        finished_at="2026-07-31T01:00:02+00:00",
    )

    assert seed["status"] == "RUNNING"
    assert seed["slides"] == []
    assert decision.report["status"] == "PARTIAL"
    assert decision.report["slides"] == [{"file": "good.pdf", "nodes": 2, "edges": 1}]
    assert decision.report["errors"] == [{"file": "bad.mp4", "error": "asr failed"}]
    assert decision.error.args == ("1 ingestion source(s) failed",)


def test_report_reducer_preserves_fatal_error_after_recording_it():
    from knowledge.pipeline.reporting import reduce_report

    fatal = OSError("neo4j unavailable")
    decision = reduce_report(
        new_report("Course", "run-1", "2026-07-31T01:00:00+00:00"),
        (),
        fatal=fatal,
        duration_s=0.1,
        finished_at="2026-07-31T01:00:01+00:00",
    )

    assert decision.report["status"] == "FAILED"
    assert decision.report["errors"] == [{
        "stage": "course-flow", "error": "neo4j unavailable",
    }]
    assert decision.error is fatal


def test_course_flow_delegates_terminal_report_to_reducer(monkeypatch):
    from knowledge.pipeline import flows
    from knowledge.pipeline.reporting import ReportDecision

    calls, reports = [], []

    def reduce(seed, outcomes, **kwargs):
        calls.append((seed, tuple(outcomes), kwargs))
        return ReportDecision({**seed, "status": "COMPLETED"}, None)

    monkeypatch.setattr(flows, "reduce_report", reduce, raising=False)
    monkeypatch.setattr(flows.deps, "minio_repo", lambda: object())
    monkeypatch.setattr(
        flows,
        "persist_report",
        lambda storage, course, run_id, report: reports.append(report),
    )

    out = asyncio.run(flows.course_flow.fn("Course", [], [], video_files=[], reset=False))

    assert calls and calls[0][1] == ()
    assert reports[-1]["status"] == "COMPLETED"
    assert out["status"] == "COMPLETED"


def test_course_flow_persists_failed_report_then_raises(monkeypatch):
    from knowledge.pipeline import flows

    reports = []

    async def fail_slide(course_name, file_name):
        raise ValueError("broken deck")

    monkeypatch.setattr(flows, "slide_flow", fail_slide)
    monkeypatch.setattr(
        flows,
        "persist_report",
        lambda storage, course, run_id, report: reports.append(dict(report)),
    )
    monkeypatch.setattr(flows.deps, "minio_repo", lambda: object())

    with pytest.raises(RuntimeError, match="1 ingestion source"):
        asyncio.run(
            flows.course_flow.fn(
                "Course",
                ["broken.pdf"],
                [],
                video_files=[],
                reset=False,
            )
        )

    assert [report["status"] for report in reports] == ["RUNNING", "FAILED"]
    assert reports[-1]["errors"] == [{"file": "broken.pdf", "error": "broken deck"}]
    assert reports[-1]["finished_at"]


def test_course_flow_marks_mixed_source_results_partial(monkeypatch):
    from knowledge.pipeline import flows

    reports = []

    async def mixed_slide(course_name, file_name):
        if file_name == "broken.pdf":
            raise ValueError("broken deck")
        return ([{"id": "c1", "typeNode": "Concept"}], [], [])

    async def persist_slide(*_args):
        return None

    monkeypatch.setattr(flows, "slide_flow", mixed_slide)
    monkeypatch.setattr(flows, "persist_slide_task", persist_slide)
    monkeypatch.setattr(
        flows,
        "persist_report",
        lambda storage, course, run_id, report: reports.append(dict(report)),
    )
    monkeypatch.setattr(flows.deps, "minio_repo", lambda: object())

    with pytest.raises(RuntimeError, match="1 ingestion source"):
        asyncio.run(
            flows.course_flow.fn(
                "Course",
                ["good.pdf", "broken.pdf"],
                [],
                video_files=[],
                reset=False,
            )
        )

    assert reports[-1]["status"] == "PARTIAL"
    assert reports[-1]["slides"] == [{"file": "good.pdf", "nodes": 1, "edges": 0}]


def test_course_flow_persists_fatal_state_and_keeps_original_error(monkeypatch):
    from knowledge.pipeline import flows

    reports = []

    class _Graph:
        def reset(self, _db_name):
            raise OSError("neo4j unavailable")

    monkeypatch.setattr(flows.deps, "graph_db", lambda: _Graph())
    monkeypatch.setattr(flows.deps, "minio_repo", lambda: object())
    monkeypatch.setattr(
        flows,
        "persist_report",
        lambda storage, course, run_id, report: reports.append(dict(report)),
    )

    with pytest.raises(OSError, match="neo4j unavailable"):
        asyncio.run(
            flows.course_flow.fn(
                "Course",
                [],
                [],
                video_files=[],
                reset=True,
            )
        )

    assert reports[-1]["status"] == "FAILED"
    assert reports[-1]["errors"] == [{
        "stage": "course-flow",
        "error": "neo4j unavailable",
    }]
