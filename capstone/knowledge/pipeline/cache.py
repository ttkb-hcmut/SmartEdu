from dataclasses import asdict
from hashlib import sha256
import json
import os

from prefect.serializers import CompressedPickleSerializer

from core.config import ASR_conf, Ingest_param
from core.llm.config import LLMConfig
from knowledge.pipeline import deps


STAGE_VERSION = "v2"
RESULT_STORAGE = "remote-file-system/prefect-sftp-results"
CACHE_SERIALIZER = CompressedPickleSerializer(compressionlib="zlib")


def release_revision(required: bool = False) -> str:
    revision = os.getenv("INGEST_RELEASE_REVISION") or "dev"
    if required and revision == "dev":
        raise RuntimeError("INGEST_RELEASE_REVISION must identify the deployed Git revision")
    return revision


def _digest(value) -> str:
    raw = json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode()).hexdigest()


def _prompt_revision() -> str:
    from core.llm.prompt.graph import prompt_p0, prompt_p1, prompt_p2

    return _digest((str(prompt_p0), str(prompt_p1), str(prompt_p2)))


def _stage_revision(task_name: str) -> str:
    revision = {
        "stage": STAGE_VERSION,
        "task": task_name,
        "release": release_revision(),
    }
    if task_name in {"parse-slide", "parse-textbook"}:
        revision["ingest"] = asdict(Ingest_param())
    elif task_name == "extract-slide-kg":
        revision["llm"] = asdict(LLMConfig())
        revision["prompt"] = _prompt_revision()
    elif task_name == "transcribe-video":
        revision["asr"] = asdict(ASR_conf())
    return _digest(revision)


def source_cache_key(task_name: str, course_name: str, file_name: str, storage) -> str:
    source = storage.raw_object_name(course_name, file_name)
    return _digest({
        "source": source,
        "object_revision": storage.object_revision(source),
        "stage_revision": _stage_revision(task_name),
    })


def stage_idempotency_key(parent_run_id: str, stage_name: str, course_name: str,
                          file_name: str, storage) -> str:
    source = storage.raw_object_name(course_name, file_name)
    return _digest({
        "parent_run_id": parent_run_id,
        "stage": stage_name,
        "course": course_name,
        "file": file_name,
        "object_revision": storage.object_revision(source),
        "release": release_revision(),
    })


def file_cache_key(ctx, params) -> str:
    return source_cache_key(
        ctx.task.name,
        params["course_name"],
        params["file_name"],
        deps.minio_repo(),
    )
