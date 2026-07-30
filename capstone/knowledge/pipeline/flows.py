import asyncio
import logging
from time import time
from typing import Dict, List, Optional, Tuple

from prefect import flow
from prefect.runtime import flow_run

from core.config import Ingest_param, DB_NAME
from core.schema.ingest import new_report
from knowledge.ingest.persist import persist_report
from knowledge.pipeline import deps
from knowledge.pipeline.legacy import process_textbook_legacy
from knowledge.pipeline.tasks import (
    anchor_task,
    extract_slide_task,
    parse_slide_task,
    parse_textbook_task,
    persist_slide_task,
    publish_slide_task,
    segment_persist_textbook_task,
    transcribe_task,
    video_persist_task,
)


@flow(name="textbook-flow")
async def textbook_flow(course_name: str, file_name: str) -> Dict:
    tree = await parse_textbook_task(course_name, file_name)
    return await segment_persist_textbook_task(course_name, file_name, tree)


@flow(name="slide-flow")
async def slide_flow(course_name: str, file_name: str,
                     num_workers: int = 3) -> Optional[Tuple[List, List, List]]:
    chunks = await parse_slide_task(course_name, file_name)
    if not chunks:
        return None
    items = await publish_slide_task(course_name, file_name, chunks)
    if not items:
        return None
    return await extract_slide_task(course_name, file_name, items, num_workers)


@flow(name="video-flow")
async def video_flow(course_name: str, file_name: str) -> Dict:
    duration, whisper_segments = await transcribe_task(course_name, file_name)
    return await video_persist_task(course_name, file_name, duration, whisper_segments)


@flow(name="course-flow")
async def course_flow(course_name: str, slide_files: List[str],
                      textbook_files: List[str], video_files: List[str] = None,
                      reset: bool = True) -> Dict:
    ## mirrors old CourseIngestionService.run stage-for-stage — parity gate depends on it
    start = time()
    cfg = Ingest_param()
    video_files = video_files or []
    report = new_report(course_name)
    report["videos"] = []

    if reset:
        deps.graph_db().reset(DB_NAME)
        deps.milvus_db().reset()

    # textbook first: build the anchor substrate before slides
    if cfg.textbook_first and textbook_files:
        tb_results = await asyncio.gather(*[
            textbook_flow(course_name, f) for f in textbook_files
        ], return_exceptions=True)
        for f, res in zip(textbook_files, tb_results):
            if isinstance(res, BaseException):
                report["errors"].append({"file": f, "error": str(res)})
            elif res:
                report["textbooks"].append(res)

    # slides -> taught concepts
    results = await asyncio.gather(*[
        slide_flow(course_name, f) for f in slide_files
    ], return_exceptions=True)

    concept_nodes = []
    for f, res in zip(slide_files, results):
        if isinstance(res, BaseException):
            report["errors"].append({"file": f, "error": str(res)})
            continue
        if res is None:
            continue
        nodes, edges, clusters = res
        await persist_slide_task(course_name, nodes, edges, clusters)
        concept_nodes += [n for n in nodes if n.get("typeNode") == "Concept"]
        report["slides"].append({"file": f, "nodes": len(nodes), "edges": len(edges)})

    # anchor concepts into passages, or fall back to legacy link-after
    if cfg.textbook_first:
        if textbook_files:
            report["anchors"] = await anchor_task(course_name, concept_nodes)
    else:
        await asyncio.gather(*[
            process_textbook_legacy(course_name, f) for f in textbook_files
        ])

    # videos: anchor-only, AFTER concepts exist in milvus (ADR-0006)
    if video_files:
        vid_results = await asyncio.gather(*[
            video_flow(course_name, f) for f in video_files
        ], return_exceptions=True)
        for f, res in zip(video_files, vid_results):
            if isinstance(res, BaseException):
                report["errors"].append({"file": f, "error": str(res)})
            elif res:
                report["videos"].append(res)

    report["duration_s"] = round(time() - start, 1)
    logging.info(f"[ingest] {report}")

    run_id = str(flow_run.id) if flow_run.id else "local"
    try:
        persist_report(deps.minio_repo(), course_name, run_id, report)
    except Exception as e:
        logging.warning(f"[ingest] report persist failed: {e}")
    return report
