import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from httpx import HTTPStatusError, Request, Response
from fastapi import BackgroundTasks, HTTPException
from prefect.exceptions import PrefectHTTPStatusError

from knowledge.api import route


def test_required_capabilities_match_ingestion_shape():
    from knowledge.pipeline.readiness import required_capabilities

    assert required_capabilities(["slides.pdf"], [], [], True) == (
        "ingest-ocr", "ingest-llm"
    )
    assert required_capabilities([], ["textbook.pdf"], [], True) == ("ingest-ocr",)
    assert required_capabilities([], ["textbook.pdf"], [], False) == ()
    assert required_capabilities([], [], ["lecture.mp4"], False) == ("ingest-asr",)


def test_capability_status_ignores_stale_and_offline_workers():
    from knowledge.pipeline.readiness import capability_status

    now = datetime(2026, 7, 31, tzinfo=timezone.utc)

    class _Worker:
        def __init__(self, status, heartbeat):
            self.status = status
            self.last_heartbeat_time = heartbeat

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def read_workers_for_work_pool(self, pool):
            return {
                "ingest-ocr": [_Worker("ONLINE", now - timedelta(seconds=89))],
                "ingest-llm": [_Worker("ONLINE", now - timedelta(seconds=91))],
                "ingest-asr": [_Worker("OFFLINE", now)],
            }[pool]

    status = asyncio.run(capability_status(
        ("ingest-ocr", "ingest-llm", "ingest-asr"),
        client_factory=_Client,
        now=now,
    ))

    assert status.present == ("ingest-ocr",)
    assert status.missing == ("ingest-llm", "ingest-asr")


def test_capability_status_treats_a_missing_pool_as_unavailable():
    from knowledge.pipeline.readiness import capability_status

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def read_workers_for_work_pool(self, _pool):
            response = Response(404, request=Request("POST", "http://prefect/workers"))
            raise PrefectHTTPStatusError.from_httpx_error(
                HTTPStatusError("missing", request=response.request, response=response)
            )

    status = asyncio.run(capability_status(("ingest-asr",), client_factory=_Client))

    assert status.present == ()
    assert status.missing == ("ingest-asr",)


def test_no_required_capability_skips_prefect_entirely():
    from knowledge.pipeline.readiness import capability_status

    def no_prefect_client():
        raise AssertionError("textbook reuse must not preflight Prefect")

    status = asyncio.run(capability_status((), client_factory=no_prefect_client))

    assert status.required == status.present == status.missing == ()


def test_ingest_returns_503_without_scheduling_when_capability_is_missing(monkeypatch):
    class _Storage:
        def raw_object_name(self, course_name, file_name):
            return f"{course_name}/_raw/{file_name}"

        def object_exists(self, _object_name):
            return True

    class _Service:
        minio_repo = _Storage()

        def __init__(self):
            self.validated = []

        def validate_files(self, course_name, pdf_names, video_names):
            self.validated.append((course_name, pdf_names, video_names))

    async def unavailable(_required):
        from knowledge.pipeline.readiness import CapabilityStatus
        return CapabilityStatus(
            required=("ingest-asr",), present=(), missing=("ingest-asr",)
        )

    async def submit(_params):
        raise AssertionError("503 must not schedule a root flow")

    monkeypatch.setenv("INGEST_ORCHESTRATOR", "prefect")
    monkeypatch.setattr(route, "capability_status", unavailable, raising=False)
    monkeypatch.setattr(route, "course_submit", submit)
    req = route.CourseIngestionRequest(
        course_name="Machine Learning",
        slide_files=[], textbook_files=[], video_files=["lecture.mp4"],
    )

    with pytest.raises(HTTPException) as err:
        asyncio.run(route.ingest_course(req, BackgroundTasks(), _Service(), None))

    assert err.value.status_code == 503
    assert err.value.detail == {
        "code": "ingest_capability_unavailable",
        "missing": ["ingest-asr"],
        "present": [],
    }
