import asyncio
import inspect
import logging
from datetime import datetime, timezone
from time import time
from typing import Dict, List, Optional, Tuple

from prefect import flow
from prefect.deployments import run_deployment
from prefect.runtime import flow_run

from core.config import Ingest_param, DB_NAME
from core.schema.ingest import (
    KGExtractionResult,
    ParsedSlideResult,
    ParsedTextbookResult,
    PublishedSlideItem,
    TranscriptResult,
    new_report,
)
from knowledge.ingest.persist import persist_report
from knowledge.pipeline import deps
from knowledge.pipeline.legacy import process_textbook_legacy
from knowledge.pipeline.reporting import ReportOutcome, reduce_report, source_outcomes
from knowledge.pipeline.tasks import (
    anchor_task,
    extract_slide_task,
    parse_slide_task,
    parse_textbook_task,
    persist_slide_task,
    publish_slide_task,
    segment_persist_textbook_task,
    transcribe_task,
    vid_persist_task,
)
from knowledge.pipeline.cache import (
    CACHE_SERIALIZER,
    RESULT_STORAGE,
    stage_idempotency_key,
)


STAGE_DEPLOYMENTS = {
    "ocr-slide": "ocr-slide-stage/ocr-slide",
    "ocr-textbook": "ocr-textbook-stage/ocr-textbook",
    "llm-slide": "llm-slide-stage/llm-slide",
    "asr-video": "asr-video-stage/asr-video",
}


def _parent_run_id() -> str:
    return str(flow_run.root_flow_run_id or flow_run.id or "local")


async def dispatch_stage(stage_name: str, course_name: str, file_name: str,
                         **extra) -> Dict:
    storage = deps.minio_repo()
    params = {"course_name": course_name, "file_name": file_name, **extra}
    run = run_deployment(
        name=STAGE_DEPLOYMENTS[stage_name],
        parameters=params,
        timeout=None,
        as_subflow=True,
        idempotency_key=stage_idempotency_key(
            _parent_run_id(), stage_name, course_name, file_name, storage
        ),
    )
    if inspect.isawaitable(run):
        run = await run
    result = run.state.result(raise_on_failure=True)
    if inspect.isawaitable(result):
        result = await result
    return result


@flow(name="ocr-slide-stage", persist_result=True, result_storage=RESULT_STORAGE,
      result_serializer=CACHE_SERIALIZER)
async def ocr_slide_stage(course_name: str, file_name: str) -> ParsedSlideResult:
    return {"chunks": await parse_slide_task(course_name, file_name)}


@flow(name="ocr-textbook-stage", persist_result=True, result_storage=RESULT_STORAGE,
      result_serializer=CACHE_SERIALIZER)
async def ocr_textbook_stage(course_name: str, file_name: str) -> ParsedTextbookResult:
    tree = await parse_textbook_task(course_name, file_name)
    return {"sections": tree["sections"], "items": tree["items"]}


@flow(name="llm-slide-stage", persist_result=True, result_storage=RESULT_STORAGE,
      result_serializer=CACHE_SERIALIZER)
async def llm_slide_stage(course_name: str, file_name: str,
                          items: List[PublishedSlideItem],
                          num_workers: int = 3) -> KGExtractionResult:
    nodes, edges, clusters = await extract_slide_task(
        course_name, file_name, items, num_workers
    )
    return {"nodes": nodes, "edges": edges, "clusters": clusters}


@flow(name="asr-video-stage", persist_result=True, result_storage=RESULT_STORAGE,
      result_serializer=CACHE_SERIALIZER)
async def asr_video_stage(course_name: str, file_name: str) -> TranscriptResult:
    duration, segments = await transcribe_task(course_name, file_name)
    return {"duration": duration, "segments": segments}


@flow(name="textbook-flow")
async def textbook_flow(course_name: str, file_name: str) -> Dict:
    tree = await dispatch_stage("ocr-textbook", course_name, file_name)
    return await segment_persist_textbook_task(course_name, file_name, tree)


@flow(name="slide-flow")
async def slide_flow(course_name: str, file_name: str,
                     num_workers: int = 3) -> Optional[Tuple[List, List, List]]:
    parsed = await dispatch_stage("ocr-slide", course_name, file_name)
    chunks = parsed["chunks"]
    if not chunks:
        return None
    items = await publish_slide_task(course_name, file_name, chunks)
    if not items:
        return None
    result = await dispatch_stage(
        "llm-slide", course_name, file_name, items=items, num_workers=num_workers
    )
    return result["nodes"], result["edges"], result["clusters"]


@flow(name="video-flow")
async def vid_flow(course_name: str, file_name: str) -> Dict:
    transcript = await dispatch_stage("asr-video", course_name, file_name)
    return await vid_persist_task(
        course_name, file_name, transcript["duration"], transcript["segments"]
    )


@flow(name="course-flow")
async def course_flow(course_name: str, slide_files: List[str],
                      textbook_files: List[str], video_files: List[str] = None,
                      reset: bool = True) -> Dict:
    ## mirrors old CourseIngestionService.run stage-for-stage — parity gate depends on it
    start = time()
    cfg = Ingest_param()
    video_files = video_files or []
    run_id = str(flow_run.id) if flow_run.id else "local"
    report = new_report(course_name, run_id)
    storage = deps.minio_repo()
    persist_report(storage, course_name, run_id, report)
    fatal = None
    outcomes = []

    try:
        if reset:
            deps.graph_db().reset(DB_NAME)
            deps.milvus_db().reset()

        if cfg.textbook_first and textbook_files:
            tb_results = await asyncio.gather(*[
                textbook_flow(course_name, f) for f in textbook_files
            ], return_exceptions=True)
            outcomes.extend(source_outcomes("textbooks", textbook_files, tb_results))

        results = await asyncio.gather(*[
            slide_flow(course_name, f) for f in slide_files
        ], return_exceptions=True)

        concept_nodes = []
        for outcome in source_outcomes("slides", slide_files, results):
            if outcome.error is not None:
                outcomes.append(outcome)
                continue
            if outcome.value is None:
                continue
            nodes, edges, clusters = outcome.value
            await persist_slide_task(course_name, nodes, edges, clusters)
            concept_nodes += [n for n in nodes if n.get("typeNode") == "Concept"]
            outcomes.append(ReportOutcome(
                "slides",
                value={"file": outcome.file_name, "nodes": len(nodes), "edges": len(edges)},
            ))

        if cfg.textbook_first:
            if textbook_files:
                outcomes.append(ReportOutcome(
                    "anchors", value=await anchor_task(course_name, concept_nodes)
                ))
        else:
            await asyncio.gather(*[
                process_textbook_legacy(course_name, f) for f in textbook_files
            ])

        ## vid after slides; concept ANN needs Milvus
        if video_files:
            vid_res = await asyncio.gather(*[
                vid_flow(course_name, f) for f in video_files
            ], return_exceptions=True)
            outcomes.extend(source_outcomes("videos", video_files, vid_res))
    except Exception as exc:
        fatal = exc
    finally:
        decision = reduce_report(
            report,
            outcomes,
            fatal=fatal,
            duration_s=round(time() - start, 1),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        logging.info(f"[ingest] {decision.report}")
        persist_report(storage, course_name, run_id, decision.report)

    if decision.error is not None:
        raise decision.error
    return decision.report
