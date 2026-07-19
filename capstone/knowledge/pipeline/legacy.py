import asyncio
import uuid

from core.config import Ingest_param, DB_NAME
from core.ingest.stages.fetch import fetch_raw_pdf
from core.ingest.stages.parse import parse_textbook_chunks
from core.repo.storage.minio_repo import make_topic_name
from knowledge.engine.extract import GraphExtractionService
from knowledge.pipeline import deps


async def process_textbook_legacy(course_name: str, file_name: str, num_workers: int = 3):
    ## dormant path (textbook_first=False): llm link-after, no caching ceremony
    file_bytes = await asyncio.to_thread(fetch_raw_pdf, deps.minio_repo(), course_name, file_name)
    chunks = await asyncio.to_thread(parse_textbook_chunks, file_bytes, Ingest_param())

    extractor = GraphExtractionService(llm_engine=deps.llm())
    sem = asyncio.Semaphore(num_workers)

    async def _handle_chunk(chunk):
        async with sem:
            text_content = chunk.get("content")
            chunk_id = chunk.get("chunk_id")
            heading = chunk.get("heading", "")

            if not text_content:
                return

            topic = make_topic_name(file_name=file_name, heading=heading, chunk_id=chunk_id)
            storage_uri = deps.minio_repo().upload_chunk(
                chunk_id=chunk_id, content=text_content, topic=topic, course_name=course_name
            )
            search_query = f"{heading}: {text_content}"
            candidates = deps.milvus_db().search(query=search_query, embedder=deps.embedder(), top_k=10)

            if not candidates:
                return

            links = await extractor.link_textbook_chunk(text=text_content, candidates=candidates)
            if not links:
                virtual_id = f"ref_{uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id).hex[:8]}"
                links = [{
                    "anchor_id": virtual_id,
                    "justification": f"Text book related directly to course {course_name} "
                }]
            if links:
                deps.graph_db().update_links(chunk_id, heading, storage_uri, links, DB_NAME)

    tasks = [asyncio.create_task(_handle_chunk(chunk)) for chunk in chunks]
    await asyncio.gather(*tasks)
