import asyncio
from types import SimpleNamespace

from core.config import Retrieve_param
from core.schema.retrieval import RetrievalCase, RetrievalHarnessId, RetrievalPolicyId, RetrievalRoute
from TA.workflow.smart_edu import SmartEdu
from TA.retrieval.policy import resolve_retrieval_context
from TA.helper.schema import BenchmarkAnswer


class _App:
    def __init__(self):
        self.context = None
        self.config = None

    async def astream(self, state, *, config, context, stream_mode):
        self.context = context
        self.config = config
        if False:
            yield None


def test_execute_injects_resolved_context_not_config_dto():
    app = _App()
    smart = object.__new__(SmartEdu)
    smart.app = app
    smart.teach_tools = {}
    context = resolve_retrieval_context(Retrieve_param.from_preset("RAG"))
    state = {"messages": [], "user_query": "q", "language": "eng"}

    result = asyncio.run(
        smart.execute(
            initial_state=state,
            session_id="s",
            retrieval_context=context,
        )
    )

    assert result == state
    assert app.context is context
    assert "retrieve_param" not in app.config["configurable"]


def test_typed_forced_route_bypasses_router_model():
    smart = object.__new__(SmartEdu)
    smart.agents = {}
    context = resolve_retrieval_context(
        Retrieve_param.from_preset("RAG"),
        case=RetrievalCase(forced_route=RetrievalRoute.RETRIEVE),
    )

    result = asyncio.run(
        smart.ta_router_node(
            {"user_query": "q", "messages": [], "language": "eng"},
            {"configurable": {}},
            SimpleNamespace(context=context),
        )
    )

    assert result == {"intent": "retrieve"}


def test_benchmark_answer_keeps_raw_ollama_fallback_scored():
    raw = SimpleNamespace(content="unknown")

    assert SmartEdu._benchmark_answer_text({"parsed": None, "raw": raw}) == "unknown"


def test_benchmark_answer_prefers_parsed_contract():
    raw = SimpleNamespace(content='{"answer": "wrong"}')

    assert SmartEdu._benchmark_answer_text(
        {"parsed": BenchmarkAnswer(answer="right"), "raw": raw}
    ) == "right"


def test_benchmark_answer_prompt_contains_ledger_and_aggregator_synthesis():
    prompt = SmartEdu._benchmark_answer_prompt(
        "Where is it?",
        {
            "RAG": {
                "thought": "The supported chain identifies the Scottish Parliament.",
                "content": "[gold-a] devolved body\n[gold-b] Holyrood",
            }
        },
    )

    assert "supported chain identifies" in prompt
    assert "[gold-a]" in prompt and "[gold-b]" in prompt


def test_v4_benchmark_uses_dedicated_answerer_without_ta_fallback():
    smart = object.__new__(SmartEdu)
    dedicated = object()
    smart.agents = {"RETRIEVAL_ANSWERER": dedicated}
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )

    selected = smart._benchmark_answer_model(SimpleNamespace(model=object()), context)

    assert selected is dedicated


def test_v3_observability_labels_composite_retrieve_as_deterministic():
    from TA.observability import build_node_config_manifest

    agents = {
        "TA": SimpleNamespace(model=SimpleNamespace(model="ta")),
        "RAG": SimpleNamespace(model=SimpleNamespace(model="qwen3:8b")),
        "RAG_LEDGER_AGGREGATOR": SimpleNamespace(
            model=SimpleNamespace(model="gpt-oss:120b-cloud", temperature=0.0)
        ),
        "RETRIEVAL_ANSWERER": SimpleNamespace(
            model="gpt-oss:120b-cloud", temperature=0.0, num_ctx=65536
        ),
    }
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )

    manifest = build_node_config_manifest(agents, context)

    assert manifest["Agentic_Retrieve"]["kind"] == "deterministic"
    assert manifest["Agentic_Retrieve"]["model"] is None
    assert manifest["Retrieval_Ledger_Aggregator"]["model"] == "gpt-oss:120b-cloud"
    assert manifest["TA_Retrieve_Finish"]["num_ctx"] == 65536
