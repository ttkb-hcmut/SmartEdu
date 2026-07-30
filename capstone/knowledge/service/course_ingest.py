# Utis
from typing import List, Dict, Optional
import uuid
import logging
from time import time
import asyncio

# Logic
from core.schema.graph.graph import KG_Instance
from core.repo.graph.insert import serialize_kg_to_dict
from core.ingest.segment import group_passages
from core.repo.storage.minio_repo import make_topic_name
from knowledge.ingest.anchor import anchor_concepts
from knowledge.ingest.fetch import fetch_raw_pdf
from knowledge.ingest.parse import parse_slide_pdf, parse_textbook_chunks, parse_textbook_tree
from knowledge.ingest.persist import persist_slide_kg
from knowledge.ingest.publish import publish_slide_chunks
from knowledge.engine.extract import GraphExtractionService
from knowledge.engine.graph.graph_constructor import KG_Handler

#shared
from fastapi import HTTPException
from core.repo.graph.graphdb import GraphDB
from core.repo.milvus_db.mil import MilvusDB
from core.repo.storage.minio_repo import MinioDB
from core.model.embedding import Embedder
from core.config import *


class CourseIngestionService:
    def __init__(self, llm, graph_db, milvus_db, minio_repo, embedder):
        self.llm = llm
        self.graph_db : GraphDB = graph_db
        self.milvus_db : MilvusDB = milvus_db
        self.minio_repo : MinioDB= minio_repo
        self.extractor = GraphExtractionService(llm_engine=self.llm)
        self.embedder : Embedder = embedder
        self.db_name = DB_NAME
        self.ocr_sem = asyncio.Semaphore(1)
        self.last_report: Optional[Dict] = None

        self.config = Ingest_param()

    async def _process_slide(self, name: str, course_name, num_workers: int = 3):
        file_bytes = await asyncio.to_thread(fetch_raw_pdf, self.minio_repo, course_name, name)

        async with self.ocr_sem:
            chunks: List[Dict[str, str]] = await asyncio.to_thread(
                parse_slide_pdf, file_bytes, self.config
            )

        if not chunks:
            return None

        texts, _, items = publish_slide_chunks(
            self.minio_repo, file_bytes, chunks, course_name, name
        )

        if not texts:
            return None

        ## queue built after publish: unbounded, removes latent >10-chunk deadlock
        q = asyncio.Queue()
        for item in items:
            await q.put(item)
        for _ in range(num_workers):
            await q.put(None)

        extractor = GraphExtractionService(llm_engine=self.llm)
        extractor.kg_handler = KG_Handler()

        kg: KG_Instance = await extractor.extract_pipeline(
            extract_queue=q,
            course_name=course_name,
            num_workers=num_workers
        )
        nodes, edges, clusters = serialize_kg_to_dict(kg)
        return nodes, edges, clusters

    async def _process_textbook(self, name: str, course_name):
        ## textbook = primitive anchor: build :Section tree + :Passage units, no LLM
        file_bytes = await asyncio.to_thread(fetch_raw_pdf, self.minio_repo, course_name, name)
        tree = await asyncio.to_thread(parse_textbook_tree, file_bytes)

        if not tree.get("items"):
            return {"file": name, "sections": 0, "passages": 0}

        passages = await asyncio.to_thread(
            group_passages, tree["items"], self.embedder.get_embedding, self.config
        )
        raw_obj = self.minio_repo.raw_object_name(course_name, name)
        await asyncio.to_thread(
            self.graph_db.write_textbook_tree,
            tree["sections"], passages, raw_obj, self.db_name, Emb_conf().dim
        )
        return {"file": name, "sections": len(tree["sections"]), "passages": len(passages)}

    async def _anchor_concepts(self, concept_nodes: List[Dict], course_name: str) -> int:
        return await asyncio.to_thread(
            anchor_concepts, self.embedder, self.graph_db, concept_nodes,
            course_name, self.db_name, self.config
        )

    async def _process_textbook_legacy(self, name: str, course_name, num_workers: int = 3):
        file_bytes = await asyncio.to_thread(fetch_raw_pdf, self.minio_repo, course_name, name)
        chunks = await asyncio.to_thread(parse_textbook_chunks, file_bytes, self.config)

        sem = asyncio.Semaphore(num_workers)

        async def _handle_chunk(chunk):
            async with sem:
                text_content = chunk.get("content")
                chunk_id = chunk.get("chunk_id")
                heading = chunk.get("heading", "")

                if not text_content:
                    return

                # textbook chunk gets its own topic folder (text only, no page.pdf)
                topic = make_topic_name(file_name=name, heading=heading, chunk_id=chunk_id)
                storage_uri = self.minio_repo.upload_chunk(
                    chunk_id=chunk_id,
                    content=text_content,
                    topic=topic,
                    course_name=course_name
                )
                search_query = f"{heading}: {text_content}"
                candidates = self.milvus_db.search(query=search_query, embedder=self.embedder, top_k=10)

                if not candidates:
                    return

                links = await self.extractor.link_textbook_chunk(text=text_content, candidates=candidates)
                if not links:
                    virtual_id = f"ref_{uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id).hex[:8]}"
                    links = [{
                        "anchor_id": virtual_id,
                        "justification": f"Text book related directly to course {course_name} "
                    }]
                if links:
                    self.graph_db.update_links(chunk_id, heading, storage_uri, links, self.db_name)

        tasks = [asyncio.create_task(_handle_chunk(chunk)) for chunk in chunks]
        await asyncio.gather(*tasks)

    def reset_db(self):
        self.graph_db.reset(self.db_name)
        self.milvus_db.reset()
    async def run(self, req):
        start = time()
        report = {"course": req.course_name, "textbooks": [], "slides": [],
                  "anchors": 0, "errors": []}
        if req.reset:
            self.reset_db()

        # textbook first: build the anchor substrate before slides
        if self.config.textbook_first and req.textbook_files:
            tb_results = await asyncio.gather(*[
                self._process_textbook(f, req.course_name) for f in req.textbook_files
            ], return_exceptions=True)
            for f, res in zip(req.textbook_files, tb_results):
                if isinstance(res, BaseException):
                    report["errors"].append({"file": f, "error": str(res)})
                elif res:
                    report["textbooks"].append(res)

        # slides -> taught concepts
        results = await asyncio.gather(*[
            self._process_slide(f, req.course_name) for f in req.slide_files
        ], return_exceptions=True)

        concept_nodes = []
        for f, res in zip(req.slide_files, results):
            if isinstance(res, BaseException):
                report["errors"].append({"file": f, "error": str(res)})
                continue
            if res is None:
                continue
            nodes, edges, clusters = res
            persist_slide_kg(self.graph_db, self.milvus_db, self.embedder,
                             self.db_name, req.course_name, nodes, edges, clusters)
            concept_nodes += [n for n in nodes if n.get("typeNode") == "Concept"]
            report["slides"].append({"file": f, "nodes": len(nodes), "edges": len(edges)})

        # anchor concepts into passages, or fall back to legacy link-after
        if self.config.textbook_first:
            if req.textbook_files:
                report["anchors"] = await self._anchor_concepts(concept_nodes, req.course_name)
        else:
            await asyncio.gather(*[
                self._process_textbook_legacy(f, req.course_name) for f in req.textbook_files
            ])

        report["duration_s"] = round(time() - start, 1)
        logging.info(f"[ingest] {report}")
        self.last_report = report
        return report

    def validate_files(self, course_name: str, pdf_names: List[str],
                       video_names: List[str] = None):
        video_names = video_names or []
        if not pdf_names and not video_names:
            raise HTTPException(status_code=400, detail="File list is empty.")

        video_exts = (".mp4", ".mkv", ".webm", ".mp3", ".m4a", ".wav")
        checks = [(n, (".pdf",)) for n in pdf_names] + [(n, video_exts) for n in video_names]

        for name, allowed in checks:
            if not name.lower().endswith(allowed):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {name}."
                )

            # presence check (object must already exist via presigned PUT)
            raw_obj = self.minio_repo.raw_object_name(course_name, name)
            if not self.minio_repo.object_exists(raw_obj):
                raise HTTPException(
                    status_code=404,
                    detail=f"File not uploaded to storage: {name}. PUT it via the presigned URL first."
                )

        return True
