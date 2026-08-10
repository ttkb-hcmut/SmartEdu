import asyncio
from types import SimpleNamespace

from core.config import Retrieve_param
from core.schema.retrieval import (
    RetrievalCase,
    RetrievalCaseKind,
    RetrievalCodeState,
)
from TA.retrieval.policy import resolve_retrieval_context
from TA.ta_module import TAModule
from TA.tracing.tracer import AgentTracer
from TA.tracing.writer import TraceWriter


def test_trace_records_typed_retrieval_provenance(tmp_path):
    context = resolve_retrieval_context(
        Retrieve_param.from_preset("FULL"),
        case=RetrievalCase(
            run_id="run-1",
            question_id="warmup-1",
            kind=RetrievalCaseKind.WARMUP,
        ),
        code=RetrievalCodeState(revision="abc123", dirty=True),
    )
    tracer = AgentTracer(session_id="trace", writer=TraceWriter(tmp_path))
    chat_id = tracer.begin_chat("q")

    path = asyncio.run(
        tracer.end_chat(
            chat_id,
            status="FAIL",
            errors=["boom"],
            retrieval_context=context,
        )
    )

    payload = path.read_text(encoding="utf-8")
    assert '"policy_id": "baseline-v3"' in payload
    assert '"harness_id": "agentic-v2"' in payload
    assert '"question_id": "warmup-1"' in payload
    assert '"warmup": true' in payload
    assert '"code_revision": "abc123"' in payload
    assert '"dirty": true' in payload
    assert '"errors": [\n        "boom"' in payload


def test_trace_stamps_effective_config_and_monotonic_timings(tmp_path):
    tracer = AgentTracer(session_id="trace", writer=TraceWriter(tmp_path))
    chat_id = tracer.begin_chat(
        "q",
        node_configs={"RAG_Core": {"model": "qwen3:8b", "num_ctx": 8192, "max_retries": 0}},
    )
    tracer.log_step(chat_id, node="RAG_Core")
    tracer.mark_first_token(chat_id)

    path = asyncio.run(tracer.end_chat(chat_id, workflow_latency_ms=12.5))
    payload = path.read_text(encoding="utf-8")

    assert '"schema_version": "1.3"' in payload
    assert '"num_ctx": 8192' in payload
    assert '"workflow_latency_ms": 12.5' in payload
    assert '"time_to_first_token_ms":' in payload


class _Memo:
    def __init__(self):
        self.session = SimpleNamespace(chats=[])

    def get_formatted_history(self, recent_turns):
        return ""


class _Tracker:
    def __init__(self):
        self.session = SimpleNamespace(context=object(), memo=_Memo(), student_state={})

    def get_session(self, session_id):
        return self.session

    def _resolve(self, session_id):
        return "student"


class _FailingEngine:
    async def execute(self, **kwargs):
        return {
            **kwargs["initial_state"],
            "status_flag": "FAIL",
            "_error": "workflow failed",
        }


class _RecordingTracer:
    langfuse_handler = None

    def __init__(self):
        self.ended = []

    def begin_chat(self, query, chat_id=None, node_configs=None):
        return chat_id or "chat"

    async def end_chat(self, **kwargs):
        self.ended.append(kwargs)


def test_ta_failure_returns_errors_and_finalizes_trace():
    ta = TAModule.__new__(TAModule)
    ta.student_tracker = _Tracker()
    ta.engine = _FailingEngine()
    model = SimpleNamespace(
        model="qwen3:8b", temperature=0.0, num_ctx=8192,
        num_predict=512, keep_alive="30m", reasoning=False,
    )
    ta.agents = {"TA": SimpleNamespace(model=model), "RAG": SimpleNamespace(model=model)}
    tracer = _RecordingTracer()
    ta._get_tracer = lambda session_id: tracer

    result = asyncio.run(
        ta.run(
            user_input="q",
            session_id="session",
            retrieve_param=Retrieve_param.from_preset("RAG"),
        )
    )

    assert result["status"] == "FAIL"
    assert result["errors"] == ["workflow failed"]
    assert tracer.ended[0]["status"] == "FAIL"
    assert tracer.ended[0]["errors"] == ["workflow failed"]
