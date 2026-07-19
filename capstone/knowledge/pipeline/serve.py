"""Ingestion worker entry.

Main machine (after `docker compose up` brings prefect-server on :4200):
    set PREFECT_API_URL=http://127.0.0.1:4200/api
    uv run python -m knowledge.pipeline.serve

Remote machine: same command with PREFECT_API_URL pointing at the main machine.
Every machine running this serves the same deployment — Prefect distributes runs;
the main-machine instance is the fallback worker of ADR-0005.
"""
from knowledge.pipeline.flows import course_flow

if __name__ == "__main__":
    ## serve() = runner deployment, no work-pool infra on one box;
    ## multi-pool split (gpu/cpu) waits for flow.deploy per ADR-0005
    course_flow.serve(name="course-ingest")
