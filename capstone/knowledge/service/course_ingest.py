# Utis
from typing import List, Dict, Optional
import os
import uuid
import logging
import tempfile
from time import time
import re
import asyncio

try:
    import fitz  # PyMuPDF — slice pdfs into small page pieces
except ImportError:
    fitz = None

# Logic
from core.schema.graph.graph import KG_Instance
from core.util.file_extractor import extract_pdf, extract_tree
from core.repo.graph.insert import serialize_kg_to_dict
from core.schema.graph.type import Ref
from core.repo.storage.minio_repo import make_topic_name
from knowledge.engine.extract import GraphExtractionService
from knowledge.engine.graph.graph_constructor import KG_Handler
from knowledge.engine.graph.helper.semantic_merge import group_passages

#shared
from core.dependencies import * 
from core.config import *


def clean_content(text: str) -> str:
    pattern = r'==== PAGE \d+ ====\s*\n?'
    return re.sub(pattern, '', text).strip()


def clean_slide_name(name: str) -> str:
    if not name: return ""
        
    name = re.sub(r'\.[a-z0-9]+$', '', name, flags=re.IGNORECASE)
    

    name = re.sub(r'^(chapter|chap|ch|slide|lecture|bài|chương)\s*\d*\s*[:\-\.]?\s*', '', name, flags=re.IGNORECASE)
    
    name = re.sub(r'[^\w\s]', ' ', name).lower().strip()
    name = re.sub(r'\s+', ' ', name) 
    
    return name

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

    @staticmethod
    def _slice_pages_pdf(doc, start_page: int, end_page: int) -> bytes:
        # cut pages [start..end] (1-indexed, inclusive) into a new small pdf
        sub = fitz.open()
        sub.insert_pdf(doc, from_page=start_page - 1, to_page=end_page - 1)
        data = sub.tobytes()
        sub.close()
        return data

    async def _process_slide(self, name: str, course_name, num_workers: int = 3):
        # pull the raw pdf from minio staging (browser dropped it via presigned url)
        raw_obj = self.minio_repo.raw_object_name(course_name, name)
        file_bytes = await asyncio.to_thread(self.minio_repo.get_object_bytes, raw_obj)

        # docling wants a path: write bytes to temp, extract, then remove
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(file_bytes)
        tmp.close()
        try:
            async with self.ocr_sem:
                chunks: List[Dict[str, str]] = await asyncio.to_thread(
                    extract_pdf, tmp.name, pages_per_batch=self.config.PAGE_PER_SLIDE,
                    step=self.config.PAGE_PER_SLIDE - self.config.slide_overlap
                )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        if not chunks:
            return None

        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) not installed — cannot split page pdfs.")
        # open the big pdf once so each chunk's page range can be sliced out
        big_doc = fitz.open(stream=file_bytes, filetype="pdf")

        q = asyncio.Queue(maxsize=10)
        hard_refs = []
        texts = []

        for i, c in enumerate(chunks):
            content = c.get('content', None)
            if not content: continue
            content: str = clean_content(text=content)

            heading = c.get('heading') or name
            texts.append((heading, content))

            # one unique topic folder per chunk
            chunk_id = c.get("chunk_id")
            topic = make_topic_name(file_name=name, heading=c.get("heading"), chunk_id=chunk_id)

            # page_num is a (start, end) tuple from the extractor; slice that range
            start_p, end_p = c["page_num"]
            page_pdf = self._slice_pages_pdf(big_doc, start_p, end_p)
            self.minio_repo.upload_topic_pdf(topic=topic, course_name=course_name, file_data=page_pdf)

            uri = self.minio_repo.upload_chunk(
                chunk_id=chunk_id, content=c.get("content"), topic=topic, course_name=course_name
            )

            ref = Ref(
                db="minio",
                id=uri,
                name=clean_slide_name(heading),
                summary=f"{content[:100]} ......".replace("\n", " "),
                p_num=c["page_num"]
            )
            hard_refs.append(ref)
            await q.put((i, heading, content, ref))

        big_doc.close()

        if not texts:
            return None

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
        raw_obj = self.minio_repo.raw_object_name(course_name, name)
        file_bytes = await asyncio.to_thread(self.minio_repo.get_object_bytes, raw_obj)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(file_bytes)
        tmp.close()
        try:
            tree = await asyncio.to_thread(extract_tree, tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        if not tree.get("items"):
            return {"file": name, "sections": 0, "passages": 0}

        passages = await asyncio.to_thread(
            group_passages, tree["items"], self.embedder.get_embedding, self.config
        )
        await asyncio.to_thread(
            self.graph_db.write_textbook_tree,
            tree["sections"], passages, raw_obj, self.db_name, Emb_conf().dim
        )
        return {"file": name, "sections": len(tree["sections"]), "passages": len(passages)}

    async def _anchor_concepts(self, concept_nodes: List[Dict], course_name: str) -> int:
        ## linked-after: concept -> passage by vector ANN, scoped to own course
        seen, names, texts = set(), [], []
        for c in concept_nodes:
            name = c.get("name")
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            names.append(name)
            texts.append(f"{name}. {c.get('content', '')}".strip())
        if not names:
            return 0

        embs = await asyncio.to_thread(self.embedder.get_embeddings, texts)
        links = []
        for name, emb in zip(names, embs):
            hits = await asyncio.to_thread(
                self.graph_db.anchor_search, emb, self.config.anchor_top_k,
                self.db_name, f"{course_name}/"
            )
            for h in hits:
                if h.get("score", 0) >= self.config.anchor_score_min:
                    links.append({"entity_name": name, "passage_id": h["passage_id"],
                                  "score": h["score"], "justification": ""})
        if links:
            await asyncio.to_thread(self.graph_db.write_anchors, links, self.db_name)
        return len(links)

    async def _process_textbook_legacy(self, name: str, course_name, num_workers: int = 3):
        # pull raw textbook from minio staging, temp-file for docling, then remove
        raw_obj = self.minio_repo.raw_object_name(course_name, name)
        file_bytes = await asyncio.to_thread(self.minio_repo.get_object_bytes, raw_obj)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(file_bytes)
        tmp.close()
        try:
            chunks = await asyncio.to_thread(
                extract_pdf, tmp.name, pages_per_batch=self.config.PAGE_PER_TB
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

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
            self.graph_db.import_data(db_name=self.db_name, nodes=nodes, edges=edges, clusters=clusters)
            self.milvus_db.insert_data(nodes=nodes, embedder=self.embedder)
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

    def validate_files(self, course_name: str, names: List[str]):
 
        if not names:
            raise HTTPException(status_code=400, detail="File list is empty.")

        for name in names:
            # extension check
            if not name.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {name}. Only PDF is allowed."
                )

            # presence check (object must already exist via presigned PUT)
            raw_obj = self.minio_repo.raw_object_name(course_name, name)
            if not self.minio_repo.object_exists(raw_obj):
                raise HTTPException(
                    status_code=404,
                    detail=f"File not uploaded to storage: {name}. PUT it via the presigned URL first."
                )
        
        return True