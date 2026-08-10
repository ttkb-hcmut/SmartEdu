# ADR-0007: Separate Prefect control and result planes

**Status:** Accepted (2026-07-19) — supersedes ADR-0005

## Context

Ingestion must survive failures after expensive OCR, LLM extraction, or Whisper transcription and
may execute on two local machines plus optional remote workers. The project intentionally keeps
`reset=true` during its flexible-testing phase, so overlapping course runs could erase Neo4j or
Milvus state while another run is writing. The original ADR-0005 colocated a SQLite Prefect server
with the application repository stack and assumed that connecting workers to the same Prefect API
made cached task results portable. Prefect centralizes run metadata, but cached result payloads
still require storage reachable by every worker.

The application remains a modular monolith. This decision separates orchestration deployment from
application ownership; it does not create per-stage microservices.

## Decision drivers

- Never overlap global-reset course ingestion.
- Reuse completed OCR/LLM/Whisper results after downstream failure, including on another machine.
- Keep orchestration failure independent from the FastAPI process and application repository stack.
- Keep reusable domain mechanisms independent of Prefect.
- Preserve local/private operation without using MinIO as Prefect's cache.

## Considered options

### Colocated Prefect with SQLite and worker-local results

Smallest deployment, but a host loss removes orchestration state and another worker cannot reuse
the cached result. Rejected for distributed execution.

### Independent Prefect with PostgreSQL only

Centralizes scheduling, history, and concurrency, but PostgreSQL stores orchestration metadata—not
the serialized task-result payloads used by cache hits. Rejected as incomplete.

### Independent Prefect with PostgreSQL and MinIO result storage

Provides portable object storage, but couples orchestration caching to the application's domain
object store and was explicitly rejected for this role.

### Independent Prefect with PostgreSQL and private SFTP result storage

Separates control and result planes, keeps both owned by the Prefect host, and is reachable from
Windows/Linux workers through Prefect's `RemoteFileSystem`. Chosen.

## Decision

- Deploy a pinned Prefect API/UI independently from the application repository compose stack.
- Back the Prefect server with PostgreSQL 14.9+; do not use SQLite for distributed operation.
- Connect FastAPI and every worker over WireGuard/Tailscale or an equivalent private network.
- Configure the course-ingestion deployment with server-side concurrency `1` and collision strategy
  `ENQUEUE`. A runner-local limit is not sufficient across multiple machines.
- Store task results in one saved Prefect `RemoteFileSystem` block backed by private SFTP storage
  on the Prefect host. Every worker resolves the same block.
- Use a compressed serializer and pin matching Prefect/project dependencies on every worker.
- Build cache identity from source object ETag/version, task and stage version, and every relevant
  config/model/prompt/embedder revision. Replacing an object must invalidate its prior result.
- Keep MinIO as the domain object store for uploaded files, generated artifacts, and ingestion
  reports only. Replacing the domain store is outside this decision.
- Keep stable contracts in `core/schema`, reusable deterministic mechanisms in `core/ingest`,
  source-specific composition in `knowledge/ingest`, and all Prefect imports in
  `knowledge/pipeline`.

## Consequences

### Positive

- Global reset cannot race another ingestion run.
- A retry on another worker can reuse completed expensive tasks.
- Prefect upgrades and outages are isolated from the FastAPI/domain-storage deployment.
- Prefect remains replaceable without moving reusable domain logic.

### Negative

- PostgreSQL, SFTP storage, private networking, credentials, and backups become operational duties.
- Cache deserialization requires compatible code and dependency versions on every worker.
- One active course flow limits throughput while global reset remains the default.

### Mitigations

- Pin server/client/dependency versions and use one tracked lock file across workers.
- Keep the result endpoint private and grant workers read/write access only to its result path.
- Verify two submitted runs queue rather than overlap.
- Verify same-object cross-worker cache hits and changed-object cache misses.
- Revisit serialization when course-scoped replacement removes the global-reset invariant.

## Revisit triggers

- Course-scoped replace/reconcile makes concurrent course ingestion safe.
- Result volume or SFTP throughput becomes a measured bottleneck.
- Multiple Prefect server replicas or higher availability become necessary.
- The project replaces MinIO as its domain object store.

## Related decisions

- [ADR-0005](0005-prefect-distributed-ingestion.md) — superseded orchestration deployment.
- [ADR-0006](0006-video-anchor-only-substrate.md) — video anchor and novelty doctrine.
