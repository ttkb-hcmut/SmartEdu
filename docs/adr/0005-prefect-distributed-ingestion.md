# ADR-0005: Ingestion as a distributed modular monolith orchestrated by Prefect

**Status:** Superseded by [ADR-0007](0007-separate-prefect-control-and-result-planes.md)
(2026-07-19)

## Context

`CourseIngestionService.run()` grew into a god-method: reset, textbook/slide gathers,
Milvus insert, anchoring, legacy fallback, and report assembly inline; stages pass
in-memory lists; any failure loses all prior work; observability is one `last_report`
dict. Video ingestion adds a minutes-long GPU transcription step that must never be
redone because a downstream write failed. Available hardware: 2 local machines plus
optional 1–2 VPS. True microservices were considered and rejected: per-stage API
contracts, artifact serialization, distributed debugging, and ops burden for one
maintainer outweigh the isolation benefit.

## Decision

- **Distributed modular monolith.** One codebase; execution distributed via Prefect
  work pools + workers; the work pool is the service boundary — no per-stage HTTP
  services, no versioned inter-service contracts.
- **Placement doctrine (extends "Cypher to core"):** stage LOGIC lives in
  `core/ingest/` — pure functions, typed contracts, zero Prefect imports — so a logic
  update takes immediate effect in every module (TA, student) without dragging an
  orchestration dependency. Prefect ORCHESTRATION lives in `knowledge/pipeline/`:
  thin `@task` wrappers (retries, cache keys, pool tags) and one `@flow` per source
  type composed under a parent `course_flow`.
- **Execution mode: worker + deployments.** FastAPI only triggers deployments and
  returns the flow-run id; it never executes ingestion in-process. The ingest report
  is persisted to MinIO (`{course}/_reports/`) because in-memory `last_report` cannot
  survive the worker split.
- **Pools:** `cpu-pool` (fetch, parse, persist) and `gpu-pool` (embed, ASR). The main
  machine runs a worker in every pool — remote workers add capacity when alive; the
  local worker picks up queued tasks when they die (fallback is native, not custom).
- **Artifact handoff:** anything large moves as a MinIO URI through task boundaries,
  never through Prefect state. Heavy models (Whisper, SciBERT) load once per worker
  process and stay warm.
- **Task caching** keyed on `(course, file, stage_version)` — a failed downstream
  stage re-runs from cache, not from Whisper.

## Consequences

- One new infra service (Prefect server) in docker-compose; Neo4j/Milvus/MinIO/Prefect
  endpoints must be env-configurable and reachable from every worker machine.
- PDF paths migrate first, behavior-preserving, gated by a parity check (identical
  node/edge/anchor counts on the benchmark course) before video lands.
- `core` stays framework-free: if Prefect is ever replaced, only
  `knowledge/pipeline/` is rewritten.
