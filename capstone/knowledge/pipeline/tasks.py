import asyncio
from typing import Dict, List, Tuple

from prefect import task

from core.config import Ingest_param, Emb_conf, DB_NAME
from core.schema.ingest import PublishedSlideItem
from core.schema.graph.type import Ref
from core.schema.graph.graph import KG_Instance
from core.repo.graph.insert import serialize_kg_to_dict
from core.ingest.segment import group_passages
from knowledge.ingest.anchor import anchor_concepts
from knowledge.ingest.fetch import fetch_raw_pdf, temp_raw
from knowledge.ingest.parse import parse_slide_pdf, parse_textbook_tree
from knowledge.ingest.persist import persist_slide_kg
from knowledge.ingest.publish import publish_slide_chunks
from knowledge.ingest.vid import build_video_segments
from knowledge.engine.extract import GraphExtractionService
from knowledge.engine.graph.graph_constructor import KG_Handler
from knowledge.pipeline import deps
from knowledge.pipeline.cache import CACHE_SERIALIZER, RESULT_STORAGE, file_cache_key

## same-process serialization as old ocr_sem, no server-side tag config needed
_ocr_sem = asyncio.Semaphore(1)


@task(name="parse-slide", retries=1, cache_key_fn=file_cache_key, persist_result=True,
      result_storage=RESULT_STORAGE, result_serializer=CACHE_SERIALIZER,
      tags=["ocr"])
async def parse_slide_task(course_name: str, file_name: str) -> List[Dict]:
    ## cache = never re-OCR a deck because a later stage failed
    file_bytes = await asyncio.to_thread(fetch_raw_pdf, deps.minio_repo(), course_name, file_name)
    async with _ocr_sem:
        return await asyncio.to_thread(parse_slide_pdf, file_bytes, Ingest_param())


@task(name="publish-slide-chunks", retries=2)
async def publish_slide_task(course_name: str, file_name: str,
                             chunks: List[Dict]) -> List[PublishedSlideItem]:
    ## uploads idempotent (same object keys) — safe to retry, no cache needed
    file_bytes = await asyncio.to_thread(fetch_raw_pdf, deps.minio_repo(), course_name, file_name)
    _, _, items = await asyncio.to_thread(
        publish_slide_chunks, deps.minio_repo(), file_bytes, chunks, course_name, file_name
    )
    return [{
        "index": index,
        "heading": heading,
        "content": content,
        "hard_ref": ref.model_dump(),
    } for index, heading, content, ref in items]


@task(name="extract-slide-kg", cache_key_fn=file_cache_key, persist_result=True,
      result_storage=RESULT_STORAGE, result_serializer=CACHE_SERIALIZER,
      tags=["llm"])
async def extract_slide_task(course_name: str, file_name: str, items: List[PublishedSlideItem],
                             num_workers: int = 3) -> Tuple[List, List, List]:
    ## cache = never re-run llm extraction; knowledge-side stage, core can't import engine
    q = asyncio.Queue()
    for item in items:
        await q.put((
            item["index"], item["heading"], item["content"],
            Ref.model_validate(item["hard_ref"]),
        ))
    for _ in range(num_workers):
        await q.put(None)

    extractor = GraphExtractionService(llm_engine=deps.llm())
    extractor.kg_handler = KG_Handler()
    kg: KG_Instance = await extractor.extract_pipeline(
        extract_queue=q, course_name=course_name, num_workers=num_workers
    )
    return serialize_kg_to_dict(kg)


@task(name="persist-slide-kg", retries=2)
async def persist_slide_task(course_name: str, nodes: List[Dict],
                             edges: List[Dict], clusters: List[Dict]):
    await asyncio.to_thread(
        persist_slide_kg, deps.graph_db(), deps.milvus_db(), deps.embedder(),
        DB_NAME, course_name, nodes, edges, clusters
    )


@task(name="parse-textbook", retries=1, cache_key_fn=file_cache_key, persist_result=True,
      result_storage=RESULT_STORAGE, result_serializer=CACHE_SERIALIZER,
      tags=["ocr"])
async def parse_textbook_task(course_name: str, file_name: str) -> Dict:
    file_bytes = await asyncio.to_thread(fetch_raw_pdf, deps.minio_repo(), course_name, file_name)
    return await asyncio.to_thread(parse_textbook_tree, file_bytes)


@task(name="segment-persist-textbook", retries=1, tags=["gpu"])
async def segment_persist_textbook_task(course_name: str, file_name: str, tree: Dict) -> Dict:
    ## fused: passage embs (768 floats each) must never cross a task boundary
    if not tree.get("items"):
        return {"file": file_name, "sections": 0, "passages": 0}

    def _run():
        embedder = deps.embedder()
        passages = group_passages(tree["items"], embedder.get_embedding, Ingest_param())
        raw_obj = deps.minio_repo().raw_object_name(course_name, file_name)
        deps.graph_db().write_textbook_tree(
            tree["sections"], passages, raw_obj, DB_NAME, Emb_conf().dim
        )
        return {"file": file_name, "sections": len(tree["sections"]), "passages": len(passages)}

    return await asyncio.to_thread(_run)


@task(name="anchor-concepts", retries=1, tags=["gpu"])
async def anchor_task(course_name: str, concept_nodes: List[Dict]) -> int:
    return await asyncio.to_thread(
        anchor_concepts, deps.embedder(), deps.graph_db(), concept_nodes,
        course_name, DB_NAME, Ingest_param()
    )


@task(name="transcribe-video", retries=1, cache_key_fn=file_cache_key, persist_result=True,
      result_storage=RESULT_STORAGE, result_serializer=CACHE_SERIALIZER,
      tags=["gpu"])
async def transcribe_task(course_name: str, file_name: str) -> Tuple[float, List[Dict]]:
    ## THE cache that justifies the refactor — never re-run whisper on downstream failure
    def _run():
        with temp_raw(deps.minio_repo(), course_name, file_name) as path:
            return deps.transcriber().transcribe(path)

    return await asyncio.to_thread(_run)


@task(name="segment-anchor-persist-video", retries=1, tags=["gpu"])
async def video_persist_task(course_name: str, file_name: str, duration: float,
                             whisper_segments: List[Dict]) -> Dict:
    ## fused: segment embs never cross a task boundary
    def _run():
        cfg = Ingest_param()
        video_id = f"{course_name}/{file_name}"
        seg_nodes, anchor_links, novel_entries = build_video_segments(
            deps.embedder(), deps.milvus_db(), video_id, course_name,
            whisper_segments, cfg
        )
        if not seg_nodes:
            return {"file": file_name, "duration": duration, "segments": 0,
                    "anchored_segments": 0, "novel_candidates": []}

        video = {"id": video_id, "title": file_name,
                 "uri": deps.minio_repo().raw_object_name(course_name, file_name),
                 "duration": duration}
        deps.graph_db().write_video(video, seg_nodes, DB_NAME, Emb_conf().dim)
        if anchor_links:
            deps.graph_db().write_segment_anchors(anchor_links, DB_NAME)

        anchored = len({l["segment_id"] for l in anchor_links})
        return {"file": file_name, "duration": duration, "segments": len(seg_nodes),
                "anchored_segments": anchored, "novel_candidates": novel_entries}

    return await asyncio.to_thread(_run)
