# ADR-0009: Route heavy ingestion stages through capability-specific Process pools

**Status:** Accepted (2026-07-31) — complements ADR-0007

## Context

OCR, slide KG extraction, and Whisper transcription have different hardware and model-loading
requirements. A static `flow.serve()` runner starts each flow in its own local subprocess, but
cannot route a child flow to a machine selected for a capability. Keeping those stages as task
decorators alone therefore does not operationally separate Docling, the LLM, and Whisper.

ADR-0007 already makes Prefect an independent control plane backed by PostgreSQL and gives it a
private SFTP result plane. This decision adds execution routing without turning the modular
monolith into HTTP microservices.

## Decision

- Keep `course-flow/course-ingest` on its static runner with global concurrency `1` and `ENQUEUE`.
- Create three Prefect **Process** pools: `ingest-ocr`, `ingest-llm`, and `ingest-asr`.
- Register exactly four child deployments:
  - `ocr-slide-stage/ocr-slide` and `ocr-textbook-stage/ocr-textbook` use `ingest-ocr`.
  - `llm-slide-stage/llm-slide` uses `ingest-llm`.
  - `asr-video-stage/asr-video` uses `ingest-asr`.
- Run each worker with `--limit 1`. More than one machine may poll the same pool for capacity or
  failover. Pool membership expresses capability, not source-to-host affinity: either OCR worker
  may receive slide or textbook work.
- The deployment registrar uses the pre-provisioned `INGEST_CHECKOUT` as Process `working_dir`,
  uses `build=False`/`push=False`, and records `INGEST_RELEASE_REVISION` as the deployment
  version. Registration and the process tracer reject an unset or `dev` revision. Every worker
  must use that checkout at the same Git SHA and run `uv sync --locked`.
- The parent dispatches child deployments and receives their compressed-SFTP result contracts;
  raw sources, temporary paths, and embedding arrays do not cross this boundary.
- `serve.py` starts only the root static runner. It never starts or registers stage workers.

## Considered options

### Static runners for every capability

Simple, but a static runner has no Prefect work-pool routing. A request cannot be assigned to a
machine advertising OCR, LLM, or ASR capacity. Rejected.

### HTTP stage services or a message broker

Adds service discovery, network APIs, queue durability, and another operational data plane for
one codebase. Rejected.

### Consul service discovery

No Consul. Prefect work pools and workers already provide the capability-routing control surface.

### Docker or Kubernetes pools

Would improve per-run isolation, but requires image/distribution infrastructure not needed for
the current local-worker phase. Deferred.

## Consequences

### Positive

- Heavy models load only on workers whose pool receives their stage deployment.
- A machine advertises capability by polling one or more named pools.
- Parent/child retries and results remain observable in the existing Prefect control/result planes.

### Costs and limits

- Workers need a matching checkout, environment, and dependency lock before they can accept work.
- The v1 Process deployment records one absolute `INGEST_CHECKOUT`; all capability workers must
  make that path available. A future heterogeneous-host deployment should use shared source
  storage or a job-template mechanism rather than silently relying on a different path.
- Readiness is not guaranteed by pool creation; HTTP admission checks belong to T7.5c.

## Related decisions

- [ADR-0007](0007-separate-prefect-control-and-result-planes.md) — control/result planes and
  root-flow serialization remain authoritative.
- [ADR-0006](0006-video-anchor-only-substrate.md) — ASR produces anchor-only video substrate.
