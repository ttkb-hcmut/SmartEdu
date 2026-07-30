from dataclasses import dataclass, field
from enum import Enum
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Prep layer
PAGE_PER_PDF = 10

# App Layer
DB_NAME = os.getenv("DB_NEO4J_DB_NAME", "test")

@dataclass
class App_settings:
    name = "Capstone Gateway" # Doesnt matter
    kg_end = os.getenv("KG_MODULE_ENDPOINT")
    ta_end =os.getenv("TA_MODULE_ENDPOINT")
    stu_end = os.getenv("ST_MODULE_ENDPOINT")
    port = 5000


# Knowledge Module
## Logic Layer

class MergeMet(str, Enum):
    COSINE = "cosine"
    HYBRID = "hybrid"          # a*cos + b*bert; keep b=0

class MergeStrat(str, Enum):
    THRESHOLD = "threshold"
    VALLEY = "valley"          # TextTiling local-minima on smoothed cosine curve

class AnchorIdx(str, Enum):
    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"

@dataclass
class Ingest_param:
    path: str = "data/"
    PAGE_PER_TB: int = 10
    PAGE_PER_SLIDE: int = 15
    slide_overlap: int = 2

    # textbook primitive, slides/papers updatable
    textbook_first: bool = True
    use_section_tree: bool = True
    semantic_merge: bool = True

    merge_metric: MergeMet = MergeMet.COSINE
    merge_strategy: MergeStrat = MergeStrat.VALLEY
    respect_section_boundary: bool = True  
    merge_threshold: float = 0.6
    valley_window: int = 2
    valley_depth: float = 0.1
    w_cosine: float = 1.0
    w_bertscore: float = 0.0

    anchor_index: AnchorIdx = AnchorIdx.HYBRID
    anchor_top_k: int = 5
    anchor_score_min: float = 0.55
    anchor_llm_rerank: bool = False         # off: anchoring pure retrieval
    extract_textbook_entities: bool = False  # off: concepts born from teaching, book = pure anchor

    # video: anchor-only substrate (ADR-0006), novelty stored not decided
    segment_top_k: int = 5
    anchor_gradient_g: float = 1.5          ## cliff ratio, keep successor while s_i > s_prev/g

### Infratructure Layer
@dataclass
class Mil_conf:
    collection_name = os.getenv("MILVUS_COLLECTION", DB_NAME)
    uri: str = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    dim: int = 768
    retries = 5
    delay = 15

@dataclass
class Emb_conf:
    model_name: str = os.getenv("EMBEDDING_MODEL", 'allenai/scibert_scivocab_uncased')
    dim: int = int(os.getenv("DIM", 768))
    retries = 5
    max_token = 512

@dataclass
class ASR_conf:
    model_name: str = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    compute_type: str = os.getenv("WHISPER_COMPUTE", "int8")   ## ~1.5GB VRAM on cuda
    device: str = os.getenv("WHISPER_DEVICE", "auto")
    
@dataclass
class Neo:
    uri: str = os.getenv("DB_NEO4J_URI", "bolt://localhost:7687")
    auth: tuple = (os.getenv("DB_NEO4J_USER", "neo4j"), os.getenv("DB_NEO4J_PASS", "graph123"))
    db_name: str = os.getenv("DB_NEO4J_DB_NAME", DB_NAME)

@dataclass
class Minio_conf:
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

@dataclass
class Mongo_conf:
    user: str = os.getenv("MONGO_USER", "admin")
    passw: str = os.getenv("MONGO_PASS", "password123")
    host: str = os.getenv("MONGO_HOST", "localhost:27017")
    uri: str = f"mongodb://{user}:{passw}@{host}/?authSource=admin"
    db_name: str = os.getenv("MONGO_DB_NAME", DB_NAME)

@dataclass
class MySQL_conf:
    host: str = os.getenv("MYSQL_HOST", "localhost")
    port: int = int(os.getenv("MYSQL_PORT", 3307))
    user: str = os.getenv("MYSQL_USER", "root")
    password: str = os.getenv("MYSQL_ROOT_PASSWORD", "")
    db_name: str = os.getenv("MYSQL_DATABASE", "capstone_db")

from core.schema.retrieval import (
    RetrievalHarnessId,
    RetrievalPolicyId,
    RetrievalPreset,
)


## frozen -> module singleton shared across sessions, mutate via from_preset only
@dataclass(frozen=True)
class Retrieve_param:
    preset: RetrievalPreset = RetrievalPreset.FULL
    policy_id: RetrievalPolicyId = RetrievalPolicyId.BASELINE_V1
    harness_id: RetrievalHarnessId = RetrievalHarnessId.AGENTIC_V1
    course_scope: str = ""

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "Retrieve_param":
        return cls(preset=RetrievalPreset(name.lower()), **overrides)

retrieve_param = Retrieve_param()

# TA module
## Logic Layer

from core.llm.prompt.agents import RAG_PROMPT, TA_PROMPT, GEN_PROMPT,EVAL_PROMPT,WORKER_PROMPT
# from core.schema.wf_state import RAGOutput
class TA_conf:
    AGENTS = [("TA",TA_PROMPT,None ) , 
             ("Generator", GEN_PROMPT, None), 
             ("RAG", RAG_PROMPT, None)
            ]
# ("Evaluator", EVAL_PROMPT, None)
@dataclass
class TA_serv:
    model_type: str = "ta"


@dataclass
class Bloom(Enum):
    """ Bloom's Taxonomy levels """
    REMEMBER = 1
    UNDERSTAND = 2
    APPLY = 3
    ANALYZE = 4
    EVALUATE = 5
    CREATE = 6

NeoStudent = Neo(db_name = "students")

# Student logic

# Tracing
class Config_Tracer:
    enable = False
    key :str = os.getenv("LANGFUSE_SECRET_KEY", "")
    public_key :str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    host :str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

langfuse_config = Config_Tracer()
TEST_LOG = True
