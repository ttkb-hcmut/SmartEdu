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
   Prefect API:

   ```powershell
   uv sync --locked
   $env:PREFECT_API_URL = "http://<private-host>:4200/api"
   ```

4. Save the shared result block once, against that same API. Use `result-store.yml` values:

   ```powershell
   uv run python -c "from os import environ as e; from prefect.filesystems import RemoteFileSystem as R; R(basepath=f'sftp://{e[\"PREFECT_SFTP_USER\"]}@{e[\"PREFECT_RESULT_HOST\"]}:{e.get(\"PREFECT_RESULT_PORT\", \"2222\")}/results', settings={'username': e['PREFECT_SFTP_USER'], 'password': e['PREFECT_SFTP_PASSWORD'], 'port': int(e.get('PREFECT_RESULT_PORT', '2222'))}).save('prefect-sftp-results', overwrite=True)"
   ```

5. Start a worker from `capstone/` on each eligible machine:

   ```powershell
   uv run python -m knowledge.pipeline.serve
   ```

`course-ingest` is registered with one global slot and `ENQUEUE`; no local worker setting is
allowed to weaken that reset-safety invariant.

Keep `PREFECT_BIND_ADDR` on loopback or a VPN address. fsspec's built-in SFTP implementation
accepts host keys automatically, so this result plane is not suitable for a public network.

## Sources

- [Prefect self-hosted Docker Compose](https://docs.prefect.io/v3/how-to-guides/self-hosted/docker-compose)
- [Prefect deployment concurrency](https://docs.prefect.io/v3/concepts/deployments)
- [Prefect persisted results and remote storage blocks](https://docs.prefect.io/v3/advanced/results)
- [fsspec SFTP implementation dependency](https://filesystem-spec.readthedocs.io/en/stable/_modules/fsspec/registry.html)
