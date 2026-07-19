from dataclasses import dataclass, field
from enum import Enum


class RetrievalPreset(str, Enum):
    PLAIN = "plain"
    RAG = "rag"
    FULL = "full"


class RetrievalPolicyId(str, Enum):
    BASELINE_V1 = "baseline-v1"


class RetrievalHarnessId(str, Enum):
    AGENTIC_V1 = "agentic-v1"
    FANOUT_V1 = "fanout-v1"


class RetrievalPromptId(str, Enum):
    MULTIHOP_V1 = "multihop-v1"


class RetrievalToolId(str, Enum):
    SEMANTIC = "semantic"
    TEXTBOOK = "textbook"


class RetrievalCaseKind(str, Enum):
    BENCHMARK = "benchmark"
    WARMUP = "warmup"


class RetrievalRoute(str, Enum):
    RETRIEVE = "retrieve"
    ROADMAP = "roadmap"
    TEACHING = "teaching"
    CONFIRM = "confirm"
    UNKNOWN = "unknown"


class RetrievalValidity(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    POLICY_INVALID = "policy-invalid"


@dataclass(frozen=True)
class RetrievalPolicyContract:
    id: RetrievalPolicyId
    digest: str
    prompt_id: RetrievalPromptId
    prompt: str
    model_profile: str
    model_name: str
    temperature: float
    allowed_tools: tuple[RetrievalToolId, ...]
    min_tool_calls: int
    empty_hits_valid: bool


@dataclass(frozen=True)
class RetrievalHarnessContract:
    id: RetrievalHarnessId
    digest: str
    max_tool_calls: int
    recursion_limit: int
    top_k: int
    per_source_k: int
    rrf_k: int


@dataclass(frozen=True)
class RetrievalScope:
    course: str = ""


@dataclass(frozen=True)
class RetrievalCase:
    run_id: str = ""
    question_id: str = ""
    kind: RetrievalCaseKind = RetrievalCaseKind.BENCHMARK
    forced_route: RetrievalRoute | None = None


@dataclass(frozen=True)
class RetrievalCodeState:
    revision: str = ""
    dirty: bool = False


@dataclass(frozen=True)
class RetrievalRunContext:
    preset: RetrievalPreset
    policy: RetrievalPolicyContract
    harness: RetrievalHarnessContract
    scope: RetrievalScope = field(default_factory=RetrievalScope)
    case: RetrievalCase = field(default_factory=RetrievalCase)
    code: RetrievalCodeState = field(default_factory=RetrievalCodeState)


@dataclass(frozen=True)
class RetrievalValidation:
    validity: RetrievalValidity
    attempted_calls: int
    errors: tuple[str, ...] = ()
