"""Run a synthetic child through a real Process worker."""

import asyncio
import inspect
import os
from pathlib import Path
from uuid import uuid4

from prefect import flow
from prefect.deployments import run_deployment

from knowledge.pipeline.cache import release_revision
from knowledge.pipeline.deploy import checkout_path


@flow(name="ingest-process-tracer-child", persist_result=True)
def tracer_child(token: str) -> dict:
    return {"token": token, "pid": os.getpid()}


@flow(name="ingest-process-tracer-root")
async def tracer_root(token: str) -> dict:
    run = run_deployment(
        name="ingest-process-tracer-child/local",
        parameters={"token": token},
        timeout=None,
        as_subflow=True,
    )
    if inspect.isawaitable(run):
        run = await run
    result = run.state.result(raise_on_failure=True)
    if inspect.isawaitable(result):
        result = await result
    return result


async def deploy_tracer(checkout: Path = None):
    checkout = checkout or checkout_path()
    source = tracer_child.from_source(
        source=checkout,
        entrypoint="knowledge/pipeline/tracer.py:tracer_child",
    )
    if inspect.isawaitable(source):
        source = await source
    deployment = source.deploy(
        name="local",
        work_pool_name="ingest-asr",
        build=False,
        push=False,
        job_variables={"working_dir": str(checkout)},
        version=release_revision(),
        ignore_warnings=True,
        print_next_steps=False,
    )
    if inspect.isawaitable(deployment):
        deployment = await deployment
    return deployment


async def run() -> dict:
    await deploy_tracer()
    token = uuid4().hex
    result = await tracer_root(token)
    if result["token"] != token or result["pid"] == os.getpid():
        raise RuntimeError("Process worker tracer did not cross a process boundary")
    return result


if __name__ == "__main__":
    print(asyncio.run(run()))
