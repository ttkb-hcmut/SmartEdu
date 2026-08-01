"""Register capability-routed Process-pool deployments."""

import asyncio
import inspect
import os
from pathlib import Path

from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import WorkPoolCreate

from knowledge.pipeline.flows import (
    asr_video_stage,
    llm_slide_stage,
    ocr_slide_stage,
    ocr_textbook_stage,
)
from knowledge.pipeline.cache import release_revision


POOL_NAMES = ("ingest-ocr", "ingest-llm", "ingest-asr")

STAGES = {
    "ocr-slide": (
        ocr_slide_stage,
        "ingest-ocr",
        "knowledge/pipeline/flows.py:ocr_slide_stage",
    ),
    "ocr-textbook": (
        ocr_textbook_stage,
        "ingest-ocr",
        "knowledge/pipeline/flows.py:ocr_textbook_stage",
    ),
    "llm-slide": (
        llm_slide_stage,
        "ingest-llm",
        "knowledge/pipeline/flows.py:llm_slide_stage",
    ),
    "asr-video": (
        asr_video_stage,
        "ingest-asr",
        "knowledge/pipeline/flows.py:asr_video_stage",
    ),
}


def checkout_path() -> Path:
    return Path(os.getenv("INGEST_CHECKOUT", Path(__file__).parents[2])).resolve()


async def deploy_stages(checkout: Path = None) -> list:
    checkout = checkout or checkout_path()
    revision = release_revision(required=True)
    deployment_ids = []
    for name, (stage, pool, entrypoint) in STAGES.items():
        deployed = stage.from_source(source=checkout, entrypoint=entrypoint)
        if inspect.isawaitable(deployed):
            deployed = await deployed
        deployment = deployed.deploy(
            name=name,
            work_pool_name=pool,
            build=False,
            push=False,
            job_variables={"working_dir": str(checkout)},
            version=revision,
            ignore_warnings=True,
            print_next_steps=False,
        )
        if inspect.isawaitable(deployment):
            deployment = await deployment
        deployment_ids.append(deployment)
    return deployment_ids


async def create_process_pools() -> list:
    async with get_client() as client:
        return [await client.create_work_pool(
            WorkPoolCreate(name=name, type="process"), overwrite=True
        ) for name in POOL_NAMES]


async def register(checkout: Path = None) -> dict:
    pools = await create_process_pools()
    deployments = await deploy_stages(checkout or checkout_path())
    return {"pools": pools, "deployments": deployments}


if __name__ == "__main__":
    print(asyncio.run(register()))
