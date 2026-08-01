"""Prefect worker capability admission for course ingestion."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

from prefect.client.orchestration import get_client
from prefect.exceptions import ObjectNotFound, PrefectHTTPStatusError


STALE_HEARTBEAT = timedelta(seconds=90)


@dataclass(frozen=True)
class CapabilityStatus:
    required: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]


def required_capabilities(slide_files: Sequence[str], textbook_files: Sequence[str],
                          video_files: Sequence[str], reset: bool) -> tuple[str, ...]:
    required = []
    if slide_files:
        required.extend(("ingest-ocr", "ingest-llm"))
    if textbook_files:
        required.append("ingest-ocr")
    if video_files:
        required.append("ingest-asr")
    return tuple(dict.fromkeys(required))


def _worker_is_live(worker, now: datetime) -> bool:
    heartbeat = worker.last_heartbeat_time
    status = getattr(worker.status, "value", worker.status)
    return status == "ONLINE" and heartbeat is not None and now - heartbeat <= STALE_HEARTBEAT


async def capability_status(required: Sequence[str], *, client_factory: Callable = get_client,
                            now: datetime = None) -> CapabilityStatus:
    required = tuple(required)
    if not required:
        return CapabilityStatus(required=(), present=(), missing=())
    now = now or datetime.now(timezone.utc)
    present = []
    async with client_factory() as client:
        for pool in required:
            try:
                workers = await client.read_workers_for_work_pool(pool)
            except (ObjectNotFound, PrefectHTTPStatusError):
                workers = []
            if any(_worker_is_live(worker, now) for worker in workers):
                present.append(pool)
    present = tuple(present)
    return CapabilityStatus(
        required=required,
        present=present,
        missing=tuple(pool for pool in required if pool not in present),
    )
