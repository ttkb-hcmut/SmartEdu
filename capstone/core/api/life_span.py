
import logging
import logging.config
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
        minio_repo=minio_repo,
        config=K_conf()
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
    app.state.ta_tasks: dict = {}   # task_id -> {status, result|error}
    yield
    
    knowledge_mod.close()
    graph_db.close()