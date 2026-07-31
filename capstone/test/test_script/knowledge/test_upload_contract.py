import asyncio
import os

import pytest
from fastapi import BackgroundTasks, HTTPException

from core.repo.storage.minio_repo import MinioDB
from knowledge.api import route
from knowledge.ingest import fetch
from knowledge.service.course_ingest import CourseIngestionService


class _Storage:
    def raw_object_name(self, course_name, file_name):
        return f"{course_name}/_raw/{file_name}"

    def object_exists(self, _object_name):
        return True

    def presigned_put_url(self, course_name, file_name):
        return f"https://storage/{course_name}/{file_name}"


class _Service:
    def __init__(self):
        self.minio_repo = _Storage()
        self.validated = []

    def validate_files(self, course_name, pdf_names, video_names):
        self.validated.append((course_name, pdf_names, video_names))

    async def run(self, _req):
        raise AssertionError("inline task must not run in this test")


def _request(videos=None):
    return route.CourseIngestionRequest(
        course_name="Machine Learning",
        slide_files=["slides.pdf"],
        textbook_files=[],
        video_files=videos or [],
    )


def test_ingest_route_declares_202_and_counts_videos(monkeypatch):
    service = _Service()

    async def _submit(params):
        assert params["video_files"] == ["lecture.mp4"]
        return "flow-123"

    monkeypatch.setenv("INGEST_ORCHESTRATOR", "prefect")
    monkeypatch.setattr(route, "course_submit", _submit)

    async def _available(required):
        from knowledge.pipeline.readiness import CapabilityStatus
        return CapabilityStatus(required=tuple(required), present=tuple(required), missing=())

    monkeypatch.setattr(route, "capability_status", _available)

    out = asyncio.run(
        route.ingest_course(
            _request(["lecture.mp4"]), BackgroundTasks(), service, None
        )
    )
    endpoint = next(item for item in route.router.routes if item.path == "/ingest-course")

    assert endpoint.status_code == 202
    assert out["flow_run_id"] == "flow-123"
    assert out["details"]["videos_count"] == 1


def test_inline_video_is_rejected_before_validation(monkeypatch):
    service = _Service()
    monkeypatch.setenv("INGEST_ORCHESTRATOR", "inline")

    with pytest.raises(HTTPException, match="Prefect") as err:
        asyncio.run(
            route.ingest_course(
                _request(["lecture.mp4"]), BackgroundTasks(), service, None
            )
        )

    assert err.value.status_code == 409
    assert service.validated == []


@pytest.mark.parametrize(
    "names",
    [
        ["../lecture.pdf"],
        ["folder/lecture.pdf"],
        ["lecture\nnotes.pdf"],
        ["slides.pdf", "slides.pdf"],
    ],
)
def test_upload_url_rejects_unsafe_or_duplicate_names(names):
    with pytest.raises(HTTPException, match="file name") as err:
        asyncio.run(
            route.get_upload_urls(
                route.UploadUrlRequest(course_name="Machine Learning", file_names=names),
                _Service(),
                None,
            )
        )

    assert err.value.status_code == 400


def test_upload_urls_return_file_name():
    out = asyncio.run(
        route.get_upload_urls(
            route.UploadUrlRequest(course_name="Machine Learning", file_names=["lecture.mp4"]),
            _Service(),
            None,
        )
    )

    assert out.model_dump() == {
        "targets": [{
            "file_name": "lecture.mp4",
            "url": "https://storage/Machine Learning/lecture.mp4",
        }]
    }


def test_validate_files_rejects_duplicate_names_across_file_groups():
    service = object.__new__(CourseIngestionService)
    service.minio_repo = _Storage()

    with pytest.raises(HTTPException, match="file name") as err:
        service.validate_files("Machine Learning", ["slides.pdf", "slides.pdf"])

    assert err.value.status_code == 400


def test_minio_download_object_uses_fget_object(tmp_path):
    class _Client:
        def __init__(self):
            self.calls = []

        def fget_object(self, bucket_name, object_name, file_path):
            self.calls.append((bucket_name, object_name, file_path))
            with open(file_path, "wb") as file:
                file.write(b"video")

        def get_object(self, *_args):
            raise AssertionError("streamed download must not call get_object")

    db = object.__new__(MinioDB)
    db.bucket_name = "courses"
    db.client = _Client()
    target = tmp_path / "lecture.mp4"

    db.download_object("Machine Learning/_raw/lecture.mp4", str(target))

    assert target.read_bytes() == b"video"
    assert db.client.calls == [
        ("courses", "Machine Learning/_raw/lecture.mp4", str(target))
    ]


def test_temp_raw_cleans_streamed_download(tmp_path, monkeypatch):
    class _Repo:
        def raw_object_name(self, course_name, file_name):
            return f"{course_name}/_raw/{file_name}"

        def download_object(self, object_name, file_path):
            assert object_name == "Machine Learning/_raw/lecture.mp4"
            with open(file_path, "wb") as file:
                file.write(b"video")

    monkeypatch.setattr(fetch.tempfile, "gettempdir", lambda: str(tmp_path))

    with fetch.temp_raw(_Repo(), "Machine Learning", "lecture.mp4") as path:
        assert os.path.exists(path)
        assert open(path, "rb").read() == b"video"

    assert not os.path.exists(path)


def test_temp_raw_removes_partial_file_after_download_error(tmp_path, monkeypatch):
    paths = []

    class _Repo:
        def raw_object_name(self, course_name, file_name):
            return f"{course_name}/_raw/{file_name}"

        def download_object(self, _object_name, file_path):
            paths.append(file_path)
            with open(file_path, "wb") as file:
                file.write(b"partial")
            raise RuntimeError("download failed")

    monkeypatch.setattr(fetch.tempfile, "gettempdir", lambda: str(tmp_path))

    with pytest.raises(RuntimeError, match="download failed"):
        with fetch.temp_raw(_Repo(), "Machine Learning", "lecture.mp4"):
            pass

    assert len(paths) == 1
    assert not os.path.exists(paths[0])
