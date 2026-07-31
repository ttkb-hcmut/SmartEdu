from types import SimpleNamespace

from core.repo.storage.minio_repo import MinioDB


class _Storage:
    def __init__(self, revision: str):
        self.revision = revision

    def raw_object_name(self, course_name: str, file_name: str) -> str:
        return f"{course_name}/_raw/{file_name}"

    def object_revision(self, object_name: str) -> str:
        assert object_name == "Course/_raw/deck.pdf"
        return self.revision


def test_object_revision_prefers_version_id_then_etag():
    db = object.__new__(MinioDB)
    db.bucket_name = "courses"
    db.client = SimpleNamespace(
        stat_object=lambda *_: SimpleNamespace(version_id="v42", etag="e42")
    )
    assert db.object_revision("Course/_raw/deck.pdf") == "v42"

    db.client = SimpleNamespace(
        stat_object=lambda *_: SimpleNamespace(version_id=None, etag="e42")
    )
    assert db.object_revision("Course/_raw/deck.pdf") == "e42"


def test_cache_key_hits_for_same_object_and_misses_when_source_changes():
    from knowledge.pipeline.cache import source_cache_key

    first = source_cache_key("transcribe-video", "Course", "deck.pdf", _Storage("v1"))
    again = source_cache_key("transcribe-video", "Course", "deck.pdf", _Storage("v1"))
    changed = source_cache_key("transcribe-video", "Course", "deck.pdf", _Storage("v2"))

    assert first == again
    assert changed != first


def test_cache_key_separates_stages_and_behavior_revisions(monkeypatch):
    from knowledge.pipeline import cache

    storage = _Storage("v1")
    parse = cache.source_cache_key("parse-slide", "Course", "deck.pdf", storage)
    transcribe = cache.source_cache_key("transcribe-video", "Course", "deck.pdf", storage)
    monkeypatch.setattr(cache, "STAGE_VERSION", f"{cache.STAGE_VERSION}-changed")
    revised = cache.source_cache_key("parse-slide", "Course", "deck.pdf", storage)

    assert parse != transcribe
    assert revised != parse


def test_cache_key_changes_with_the_release_revision(monkeypatch):
    from knowledge.pipeline import cache

    storage = _Storage("v1")
    monkeypatch.setenv("INGEST_RELEASE_REVISION", "sha-one")
    first = cache.source_cache_key("parse-slide", "Course", "deck.pdf", storage)
    monkeypatch.setenv("INGEST_RELEASE_REVISION", "sha-two")
    second = cache.source_cache_key("parse-slide", "Course", "deck.pdf", storage)

    assert first != second


def test_compressed_serializer_round_trips_large_task_results():
    from knowledge.pipeline.cache import CACHE_SERIALIZER

    payload = (123.0, [{"t_lo": 0.0, "t_hi": 1.0, "text": "lecture " * 500}])
    assert CACHE_SERIALIZER.loads(CACHE_SERIALIZER.dumps(payload)) == payload


def test_expensive_tasks_use_the_shared_compressed_serializer():
    from knowledge.pipeline.cache import CACHE_SERIALIZER
    from knowledge.pipeline.tasks import (
        extract_slide_task,
        parse_slide_task,
        parse_textbook_task,
        transcribe_task,
    )

    for task in (parse_slide_task, extract_slide_task, parse_textbook_task, transcribe_task):
        assert task.persist_result is True
        assert task.result_storage == "remote-file-system/prefect-sftp-results"
        assert task.result_serializer == CACHE_SERIALIZER


def test_prefect_cache_hits_unchanged_source_and_misses_changed_source(tmp_path, monkeypatch):
    from prefect import flow, task
    from prefect.filesystems import LocalFileSystem
    from prefect.testing.utilities import prefect_test_harness
    from knowledge.pipeline import cache

    storage = _Storage("v1")
    monkeypatch.setattr(cache.deps, "minio_repo", lambda: storage)
    calls = []

    with prefect_test_harness():
        LocalFileSystem(basepath=str(tmp_path)).save("cache-test", overwrite=True)

        @task(
            cache_key_fn=cache.file_cache_key,
            persist_result=True,
            result_storage="local-file-system/cache-test",
            result_serializer=cache.CACHE_SERIALIZER,
        )
        def transcribe_video(course_name: str, file_name: str) -> int:
            calls.append((course_name, file_name))
            return len(calls)

        @flow
        def run():
            return transcribe_video("Course", "deck.pdf")

        assert run() == 1
        assert run() == 1
        storage.revision = "v2"
        assert run() == 2

    assert calls == [("Course", "deck.pdf"), ("Course", "deck.pdf")]
