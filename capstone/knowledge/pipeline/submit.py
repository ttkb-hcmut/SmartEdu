import inspect
from typing import Dict

from prefect.deployments import run_deployment


async def course_submit(param: Dict) -> str:
    run = run_deployment(
        name="course-flow/course-ingest",
        parameters=param,
        timeout=0,
    )
    if inspect.isawaitable(run):
        run = await run
    return str(run.id)
