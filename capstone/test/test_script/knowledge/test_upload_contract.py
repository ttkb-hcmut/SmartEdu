import asyncio
import os

import pytest
from fastapi import BackgroundTasks, HTTPException

from core.repo.storage.minio_repo import MinioDB, resolve_pdf_reference
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


def test_pdf_reference_resolves_book_topic_and_missing_source():
    assert resolve_pdf_reference({"uri": "ML/_raw/book.pdf", "p_lo": 12}) == ("ML/_raw/book.pdf", 12)
    assert resolve_pdf_reference({"uri": "ML/topic/chunks/1.txt", "p_lo": 7}) == ("ML/topic/page.pdf", 1)
    assert resolve_pdf_reference({"hard_ref": {"id": "ML/_raw/hard.pdf", "p_num": [4, 5]}, "uri": "ML/_raw/other.pdf", "p_lo": 9}) == ("ML/_raw/hard.pdf", 4)
    assert resolve_pdf_reference({"uri": "missing", "p_lo": 2}) == (None, None)


@pytest.mark.parametrize("file_name", ["../book.pdf", "book.txt", "folder/book.pdf", ".."])
def test_raw_pdf_route_rejects_invalid_file_names(file_name):
    with pytest.raises(HTTPException) as error:
        asyncio.run(route.get_raw_pdf("ML", file_name, _Service(), None))
    assert error.value.status_code == 400


def test_search_groups_concepts_and_passages_with_pdf_targets():
    from types import SimpleNamespace
    from unittest.mock import Mock
    module = SimpleNamespace(embedder=Mock(), milvus_db=Mock(), graph_db=Mock())
    module.embedder.get_embedding.return_value = [0.1]
    module.milvus_db._course_expr.return_value = 'community == "ML"'
    module.milvus_db.search_vec.return_value = [{"id": "e1", "name": "Vector", "community": "Ml", "text": "concept", "score": 0.9},
                                                   {"id": "e2", "name": "Loose", "community": "Ml", "score": 0.4},
                                                   {"id": "e3", "name": "Anchor", "community": "Ml", "score": 0.3}]
    module.graph_db.resolve_entity_sources.return_value = [{"id": "e1", "hard_ref": {"id": "ML/_raw/book.pdf", "p_num": [3]}},
                                                           {"id": "e3", "uri": "ML/_raw/book.pdf", "p_lo": 5}]
    module.graph_db.passage_search.return_value = [{"id": "p1", "uri": "ML/_raw/book.pdf", "p_lo": 8, "text": "passage", "score": 0.5}]
    result = route._search_payload(module, "vector", "ML")
    assert result["concepts"][0]["document"] == "ML/_raw/book.pdf"
    assert result["concepts"][0]["page"] == 3
    assert (result["concepts"][1]["document"], result["concepts"][1]["page"]) == (None, None)
    assert result["concepts"][2]["page"] == 5
    assert result["passages"][0]["page"] == 8
    module.graph_db.passage_search.assert_called_once_with([0.1], query_text="vector", top_k=5, uri_prefix="Ml/")


@pytest.mark.parametrize("query", ["", "x", "x" * 201])
def test_search_rejects_invalid_query_before_services(query):
    with pytest.raises(HTTPException) as error:
        asyncio.run(route.search_materials(query, None, object(), None))
    assert error.value.status_code == 400


class _Service:
    def __init__(self):
        self.minio_repo = _Storage()
        self.validated = []

    def f_valid(self, course_name, pdfs, vids):
        self.validated.append((course_name, pdfs, vids))

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

    monkeypatch.setenv("INGEST_ORCH", "prefect")
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
    monkeypatch.setenv("INGEST_ORCH", "inline")

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


def test_f_valid_rejects_duplicate_names_across_file_groups():
    service = object.__new__(CourseIngestionService)
    service.minio_repo = _Storage()

    with pytest.raises(HTTPException, match="file name") as err:
        service.f_valid("Machine Learning", ["slides.pdf", "slides.pdf"])

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
