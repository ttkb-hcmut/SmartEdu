import asyncio

from core.schema.graph.type import Ref


def test_stage_contracts_round_trip_with_the_shared_serializer():
    from core.schema import ingest
    from knowledge.pipeline.cache import CACHE_SERIALIZER

    payload = {
        "slide": {"chunks": [{
            "chunk_id": "slide-1",
            "heading": "Gradient descent",
            "content": "Move opposite the gradient.",
            "page_num": (1, 2),
        }]},
        "textbook": {"sections": [], "items": [{
            "id": "passage-1",
            "text": "A textbook passage.",
            "order": 0,
            "section_id": "section-1",
            "p_num": (1, 1),
        }]},
        "kg": {"nodes": [{"id": "concept-1"}], "edges": [], "clusters": []},
        "transcript": {"duration": 10.0, "segments": [{
            "t_lo": 0.0,
            "t_hi": 10.0,
            "text": "A spoken explanation.",
        }]},
    }

    assert ingest.ParsedSlideResult.__annotations__["chunks"]
    assert ingest.ParsedTextbookResult.__annotations__["items"]
    assert ingest.KGExtractionResult.__annotations__["nodes"]
    assert ingest.TranscriptResult.__annotations__["segments"]
    assert CACHE_SERIALIZER.loads(CACHE_SERIALIZER.dumps(payload)) == payload


def test_stage_flows_persist_result_contracts_in_shared_storage():
    from knowledge.pipeline import flows
    from knowledge.pipeline.cache import CACHE_SERIALIZER, RESULT_STORAGE

    for stage in (
        flows.ocr_slide_stage,
        flows.ocr_textbook_stage,
        flows.llm_slide_stage,
        flows.asr_video_stage,
    ):
        assert stage.persist_result is True
        assert stage.result_storage == RESULT_STORAGE
        assert stage.result_serializer == CACHE_SERIALIZER


def test_stage_flows_return_only_stable_contracts(monkeypatch):
    from knowledge.pipeline import flows

    async def parse_slide(course_name, file_name):
        return [{"chunk_id": "slide-1", "page_num": (1, 1)}]

    async def parse_textbook(course_name, file_name):
        return {"sections": [], "items": []}

    async def extract_slide(course_name, file_name, items, num_workers):
        return ([{"id": "concept-1"}], [{"src": "concept-1"}], [])

    async def transcribe(course_name, file_name):
        return 12.5, [{"t_lo": 0.0, "t_hi": 12.5, "text": "lecture"}]

    monkeypatch.setattr(flows, "parse_slide_task", parse_slide)
    monkeypatch.setattr(flows, "parse_textbook_task", parse_textbook)
    monkeypatch.setattr(flows, "extract_slide_task", extract_slide)
    monkeypatch.setattr(flows, "transcribe_task", transcribe)

    items = [{
        "index": 0,
        "heading": "Topic",
        "content": "Slide text",
        "hard_ref": {"id": "Course/chunk.md", "name": "Topic"},
    }]

    assert asyncio.run(flows.ocr_slide_stage.fn("Course", "slide.pdf")) == {
        "chunks": [{"chunk_id": "slide-1", "page_num": (1, 1)}]
    }
    assert asyncio.run(flows.ocr_textbook_stage.fn("Course", "book.pdf")) == {
        "sections": [], "items": []
    }
    assert asyncio.run(flows.llm_slide_stage.fn("Course", "slide.pdf", items)) == {
        "nodes": [{"id": "concept-1"}],
        "edges": [{"src": "concept-1"}],
        "clusters": [],
    }
    assert asyncio.run(flows.asr_video_stage.fn("Course", "lecture.mp4")) == {
        "duration": 12.5,
        "segments": [{"t_lo": 0.0, "t_hi": 12.5, "text": "lecture"}],
    }


def test_dispatch_waits_for_terminal_child_result_and_uses_idempotency(monkeypatch):
    from knowledge.pipeline import flows

    calls = []

    class _State:
        def result(self, raise_on_failure):
            assert raise_on_failure is True
            return {"duration": 3.0, "segments": []}

    class _Run:
        state = _State()

    class _Storage:
        def raw_object_name(self, course_name, file_name):
            return f"{course_name}/_raw/{file_name}"

        def object_revision(self, object_name):
            assert object_name == "Course/_raw/lecture.mp4"
            return "etag-1"

    def fake_run_deployment(**kwargs):
        calls.append(kwargs)
        return _Run()

    monkeypatch.setattr(flows, "run_deployment", fake_run_deployment, raising=False)
    monkeypatch.setattr(flows, "_parent_run_id", lambda: "parent-run", raising=False)
    monkeypatch.setattr(flows.deps, "minio_repo", lambda: _Storage())
    monkeypatch.setenv("INGEST_RELEASE_REVISION", "sha-1")

    result = asyncio.run(
        flows.dispatch_stage("asr-video", "Course", "lecture.mp4")
    )

    assert result == {"duration": 3.0, "segments": []}
    assert calls == [{
        "name": "asr-video-stage/asr-video",
        "parameters": {"course_name": "Course", "file_name": "lecture.mp4"},
        "timeout": None,
        "as_subflow": True,
        "idempotency_key": flows.stage_idempotency_key(
            "parent-run", "asr-video", "Course", "lecture.mp4", _Storage()
        ),
    }]


def test_dispatch_propagates_a_failed_child_result(monkeypatch):
    from knowledge.pipeline import flows

    class _State:
        def result(self, raise_on_failure):
            raise RuntimeError("child failed")

    class _Run:
        state = _State()

    class _Storage:
        def raw_object_name(self, course_name, file_name):
            return f"{course_name}/_raw/{file_name}"

        def object_revision(self, object_name):
            return "etag-1"

    monkeypatch.setattr(flows, "run_deployment", lambda **_: _Run(), raising=False)
    monkeypatch.setattr(flows.deps, "minio_repo", lambda: _Storage())

    try:
        asyncio.run(flows.dispatch_stage("ocr-slide", "Course", "slide.pdf"))
    except RuntimeError as exc:
        assert str(exc) == "child failed"
    else:
        raise AssertionError("failed child result must propagate")


def test_publish_task_returns_json_safe_items(monkeypatch):
    from knowledge.pipeline import tasks

    monkeypatch.setattr(tasks, "fetch_raw_pdf", lambda *_: b"pdf")
    monkeypatch.setattr(tasks.deps, "minio_repo", lambda: object())
    monkeypatch.setattr(
        tasks,
        "publish_slide_chunks",
        lambda *_: ([], [], [(0, "Topic", "Slide text", Ref(
            id="Course/chunk.md", name="Topic"
        ))]),
    )

    assert asyncio.run(tasks.publish_slide_task.fn("Course", "slide.pdf", [])) == [{
        "index": 0,
        "heading": "Topic",
        "content": "Slide text",
        "hard_ref": {
            "db": "",
            "id": "Course/chunk.md",
            "name": "Topic",
            "summary": "",
            "p_num": (1, 1),
        },
    }]
