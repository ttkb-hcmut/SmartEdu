
import asyncio
import logging
import logging.config
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.llm.llm_engine import CoreLLMEngine

from core.repo.graph.graphdb import GraphDB
from core.repo.milvus_db.mil import MilvusDB
from core.model.embedding import Embedder
from core.repo.storage.minio_repo import MinioDB
from core.repo.nosql.mongo_db import Mongo_DB
from core.repo.sql.sql_db import SQL_DB

from knowledge.knowledge_construction_service import KnowledgeModule
from TA.ta_module import TAModule
from student.Student_Tracker import Student_Tracker
from core.config import *

TA_TASK_TTL_SEC = 30 * 60
TA_TASK_SWEEP_SEC = 5 * 60


async def sweep_ta_tasks(ta_tasks: dict, ttl: float = TA_TASK_TTL_SEC, interval: float = TA_TASK_SWEEP_SEC):
    ## TTL not delete-on-stream-end, /chat/status reads entries after they finish
    while True:
        await asyncio.sleep(interval)
        cutoff = time.monotonic() - ttl
        stale = [tid for tid, e in ta_tasks.items() if e.get("done_at") is not None and e["done_at"] < cutoff]
        for tid in stale:
            ta_tasks.pop(tid, None)
        if stale:
            logging.getLogger(__name__).info("Swept %d finished TA task(s).", len(stale))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Logging setup ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    # Silence overly chatty third-party libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langgraph").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

    llm = CoreLLMEngine()
    graph_db = GraphDB(config=Neo())
    milvus_db = MilvusDB(config=Mil_conf())
    embedder = Embedder(config=Emb_conf())
    minio_repo = MinioDB(config=Minio_conf())
    mongo = Mongo_DB()
    sql = SQL_DB(config=MySQL_conf())

    graph_db_student = GraphDB(config=NeoStudent)

    student_tracker = Student_Tracker(graphdb=graph_db_student, sqldb=sql, mongodb=mongo)

    knowledge_mod = KnowledgeModule(
        llm=llm,
        graph_db=graph_db,
        milvus_db=milvus_db,
        embedder=embedder,
        minio_repo=minio_repo
    )
    app.state.knowledge = knowledge_mod

    ta = TAModule(
        llm=llm, 
        graph_db=graph_db, 
        milvus_db=milvus_db, 
        embedder=embedder, 
        minio=minio_repo,
        student_tracker=student_tracker,
        config=TA_conf()
    )
    app.state.TA = ta
    app.state.student_tracker = student_tracker
    app.state.ta_tasks = {}   # task_id -> {status, result|error, queue, done_at}
    ta_task_sweeper = asyncio.create_task(sweep_ta_tasks(app.state.ta_tasks))
    yield

    ta_task_sweeper.cancel()
    try:
        await ta_task_sweeper
    except asyncio.CancelledError:
        pass

    knowledge_mod.close()
    graph_db.close()