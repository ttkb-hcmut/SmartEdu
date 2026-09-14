import asyncio
from types import SimpleNamespace

import pytest

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


def test_v5_benchmark_resolves_dedicated_answerer_by_profile():
    smart = object.__new__(SmartEdu)
    dedicated = object()
    smart.agents = {"RETRIEVAL_ANSWERER": dedicated}
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )

    assert smart._benchmark_answer_model(SimpleNamespace(model=object()), context) is dedicated


def test_v3_observability_labels_composite_retrieve_as_deterministic():
    from TA.observability import build_node_config_manifest

    agents = {
        "TA": SimpleNamespace(model=SimpleNamespace(model="ta")),
        "RAG": SimpleNamespace(model=SimpleNamespace(model="qwen3:8b")),
        "RAG_LEDGER_AGGREGATOR": SimpleNamespace(
            model=SimpleNamespace(model="gemini-3.7-flash", temperature=0.0)
        ),
        "RETRIEVAL_ANSWERER": SimpleNamespace(
            model="gemini-3.7-flash", temperature=0.0, num_ctx=65536
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
    assert manifest["Retrieval_Ledger_Aggregator"]["model"] == "gemini-3.7-flash"
    assert manifest["TA_Retrieve_Finish"]["num_ctx"] == 65536


def test_v4_observability_reports_typed_node_contracts():
    from TA.observability import build_node_config_manifest

    model = SimpleNamespace(model="gemini-3.7-flash", temperature=0.0, num_ctx=65536)
    agents = {
        "TA": SimpleNamespace(model=SimpleNamespace(model="ta")),
        "RAG": SimpleNamespace(model=SimpleNamespace(model="qwen3:8b")),
        "RETRIEVAL_PLANNER": model,
        "RETRIEVAL_ANSWERER": model,
    }
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )

    manifest = build_node_config_manifest(agents, context)

    assert manifest["Seed_Retrieve"]["kind"] == "deterministic"
    assert manifest["Retrieve_Hop"]["kind"] == "deterministic"
    assert manifest["agentic-v4"]["kind"] == "deterministic"
    assert manifest["Plan_Hop"]["model"] == "gemini-3.7-flash"
    assert manifest["Plan_Hop"]["timeout_s"] == 120
    assert manifest["Plan_Hop"]["max_retries"] == 1
    assert manifest["Finalize_Chain"]["model"] == "gemini-3.7-flash"


def test_injector_keeps_v3_aggregator_and_v4_planner_as_separate_profiles(monkeypatch):
    import TA.agent.injector as injector
    from TA.helper.schema import RAGCore

    calls = []
    models = {}

    class _Engine:
        def _get_llm(self, profile):
            calls.append(profile)
            return models.setdefault(profile, SimpleNamespace(model=profile))

    def fake_agent(**kwargs):
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(injector, "create_agent", fake_agent)
    monkeypatch.setattr(injector, "AGENT_SPECS", {
        "RAG": {
            "debug": False,
            "tools": lambda _factory: [],
            "schema": RAGCore,
        }
    })
    tools = SimpleNamespace(get_retrieve_more_tool=lambda: object())

    agents = injector.AgentInjector.initialize_all_agents(
        _Engine(), tools, [("RAG", "prompt", None)]
    )

    assert agents["RETRIEVAL_PLANNER"] is models["retrieval_planner"]
    assert agents["RAG_LEDGER_AGGREGATOR"].model is models["retrieval_aggregator"]
    assert "retrieval_planner" in calls and "retrieval_aggregator" in calls


class _ProviderError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"status code: {status_code}")
        self.status_code = status_code


class _AnswerInvocation:
    model = "gemini-3.7-flash"
    temperature = 0.0

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def bind(self, **_kwargs):
        return self

    def with_structured_output(self, *_args, **_kwargs):
        return self

    async def ainvoke(self, *_args, **_kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _v5_runtime():
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )
    return SimpleNamespace(context=context)


def test_benchmark_answer_retries_one_transient_5xx():
    answerer = _AnswerInvocation([
        _ProviderError(503),
        {"parsed": BenchmarkAnswer(answer="right"), "raw": None},
    ])
    smart = object.__new__(SmartEdu)
    smart.agents = {"RETRIEVAL_ANSWERER": answerer}

    answer = asyncio.run(smart._benchmark_answer(
        SimpleNamespace(model=object()),
        {"user_query": "q"},
        {"RAG": {"thought": "chain", "content": "evidence"}},
        {},
        _v5_runtime(),
    ))

    assert answer == "right"
    assert answerer.calls == 2


def test_benchmark_answer_never_retries_429():
    answerer = _AnswerInvocation([_ProviderError(429), {"parsed": BenchmarkAnswer(answer="wrong")}])
    smart = object.__new__(SmartEdu)
    smart.agents = {"RETRIEVAL_ANSWERER": answerer}

    with pytest.raises(_ProviderError):
        asyncio.run(smart._benchmark_answer(
            SimpleNamespace(model=object()),
            {"user_query": "q"},
            {"RAG": {"thought": "chain", "content": "evidence"}},
            {},
            _v5_runtime(),
        ))

    assert answerer.calls == 1
