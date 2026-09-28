# Independent Prefect service

This stack is intentionally not included by `capstone/core/repo/docker-compose.yaml`.
It owns Prefect scheduling metadata in PostgreSQL and cached task results in the private SFTP
volume. MinIO remains the SmartEdu domain object store.

## Bootstrap

1. Copy the following values into a local, untracked `.env` beside `compose.yml`:

   ```dotenv
   PREFECT_DB_PASSWORD=replace-me
   PREFECT_SFTP_USER=prefect-result
   PREFECT_SFTP_PASSWORD=replace-me
   # 127.0.0.1 for one host; a VPN address for remote workers.
   PREFECT_BIND_ADDR=127.0.0.1
   PREFECT_PUBLIC_HOST=127.0.0.1
   PREFECT_RESULT_HOST=127.0.0.1
   PREFECT_RESULT_PORT=2222
   ```

2. Start the independent service:

   ```powershell
   docker compose --env-file .env up -d
   ```

3. On every worker, sync the application exact dependency lock and point it at the private
   Prefect API. The task decorators resolve the saved SFTP block by this exact slug. Set the
   release value to the checkout SHA; every capability worker must use the same SHA and the same
   absolute `INGEST_CHECKOUT` path because Process deployments use it as their working directory:

   ```powershell
   $env:INGEST_CHECKOUT = (Resolve-Path .).Path
   $env:INGEST_RELEASE_REVISION = (git rev-parse HEAD).Trim()
   uv sync --locked
   $env:PREFECT_API_URL = "http://<private-host>:4200/api"
   $env:PREFECT_RESULTS_DEFAULT_STORAGE_BLOCK = "remote-file-system/prefect-sftp-results"
   ```

4. Save the shared result block once, against that same API. Use `result-store.yml` values:

   ```powershell
   uv run python -c "from os import environ as e; from prefect.filesystems import RemoteFileSystem as R; R(basepath=f'sftp://{e[\"PREFECT_SFTP_USER\"]}@{e[\"PREFECT_RESULT_HOST\"]}:{e.get(\"PREFECT_RESULT_PORT\", \"2222\")}/results', settings={'username': e['PREFECT_SFTP_USER'], 'password': e['PREFECT_SFTP_PASSWORD'], 'port': int(e.get('PREFECT_RESULT_PORT', '2222'))}).save('prefect-sftp-results', overwrite=True)"
   ```

5. Register the three Process pools and four child-stage deployments once for that release, from
   `capstone/`. Registration rejects an unset or `dev` release revision:

   ```powershell
   uv run python -m knowledge.pipeline.deploy
   ```

6. Start the static root runner on one machine only. It owns `course-ingest` and its global
   `ENQUEUE` slot; it does not start stage workers:

   ```powershell
   uv run python -m knowledge.pipeline.serve
   ```

7. Start one Process worker for each capability present on a machine. Every worker has a local
   flow-run limit of one; start multiple commands on the same machine only when it intentionally
   provides multiple capabilities:

   ```powershell
   uv run prefect worker start --pool ingest-ocr --type process --limit 1
   uv run prefect worker start --pool ingest-llm --type process --limit 1
   uv run prefect worker start --pool ingest-asr --type process --limit 1
   ```

## Process-worker smoke test

With an `ingest-asr` worker running against the same API and checkout, verify the execution
boundary without loading Whisper or touching course data:

```powershell
uv run python -m knowledge.pipeline.tracer
```

It succeeds only when the root flow receives a persisted token from a different process PID and
the ASR worker reports the same non-`dev` release revision as the root runner. Record the configured
Git SHA for the root runner and every capability worker before a T9 or T10 gate run.

## API capability admission

`POST /ingest-course` validates object names, extensions, and MinIO presence before it asks
Prefect whether a suitable worker is alive. A worker heartbeat older than 90 seconds is stale.
Slides require `ingest-ocr` and `ingest-llm`; videos require `ingest-asr`; textbooks require
`ingest-ocr` regardless of `reset`. Missing capabilities receive HTTP 503 and schedule no root
flow.

`ingest-ocr` is a shared capability pool. Any machine polling it may receive slide or textbook
OCR; the pool provides failover and capacity, not source-to-host affinity.

`course-ingest` is registered with one global slot and `ENQUEUE`; no local worker setting is
allowed to weaken that reset-safety invariant. Do not expose Process workers, SFTP, or Prefect
outside the private network.

Keep `PREFECT_BIND_ADDR` on loopback or a VPN address. fsspec's built-in SFTP implementation
accepts host keys automatically, so this result plane is not suitable for a public network.

## Sources

- [Prefect self-hosted Docker Compose](https://docs.prefect.io/v3/how-to-guides/self-hosted/docker-compose)
- [Prefect deployment concurrency](https://docs.prefect.io/v3/concepts/deployments)
- [Prefect persisted results and remote storage blocks](https://docs.prefect.io/v3/advanced/results)
- [fsspec SFTP implementation dependency](https://filesystem-spec.readthedocs.io/en/stable/_modules/fsspec/registry.html)
