"""Worker entry for the independent Prefect service.

Set PREFECT_API_URL to the private control-plane address before starting this module.
"""
from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyLimitStrategy

from knowledge.pipeline.flows import course_flow

if __name__ == "__main__":
    course_flow.serve(
        name="course-ingest",
        global_limit=ConcurrencyLimitConfig(
            limit=1,
            collision_strategy=ConcurrencyLimitStrategy.ENQUEUE,
        ),
    )
