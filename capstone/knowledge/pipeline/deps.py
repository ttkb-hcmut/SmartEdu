from functools import lru_cache

from core.llm.llm_engine import CoreLLMEngine
from core.repo.graph.graphdb import GraphDB
from core.repo.milvus_db.mil import MilvusDB
from core.model.embedding import Embedder
from core.model.asr import Transcriber
from core.repo.storage.minio_repo import MinioDB
from core.config import Neo, Mil_conf, Emb_conf, Minio_conf, ASR_conf


## heavy singletons per worker process, lazy so a cpu worker never loads cuda
@lru_cache(maxsize=1)
def llm() -> CoreLLMEngine:
    return CoreLLMEngine()


@lru_cache(maxsize=1)
def graph_db() -> GraphDB:
    return GraphDB(config=Neo())


@lru_cache(maxsize=1)
def milvus_db() -> MilvusDB:
    return MilvusDB(config=Mil_conf())


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    return Embedder(config=Emb_conf())


@lru_cache(maxsize=1)
def minio_repo() -> MinioDB:
    return MinioDB(config=Minio_conf())


@lru_cache(maxsize=1)
def transcriber() -> Transcriber:
    ## whisper model itself lazy-loads inside — only a gpu worker running video pays
    return Transcriber(config=ASR_conf())
