import asyncio
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.config import Retrieve_param
from TA.retrieval.policy import resolve_retrieval_context


@dataclass(frozen=True)
class _Request:
    state: dict
    tools: list
    runtime: object
    system_message: SystemMessage = field(default_factory=lambda: SystemMessage("base"))

    def override(self, **changes):
        return replace(self, **changes)


@dataclass(frozen=True)
class _ToolRequest:
    tool_call: dict
    tool: object
    state: dict
    runtime: object


class _Milvus:
    def search(self, **kwargs):
        return [{"id": "p_1", "text": "evidence", "score": 0.9}]


class _Embedder:
    def get_embedding(self, text):
        return [0.1]


class _Adapter:
    def __init__(self, source):
        self.source = source
        self.calls = []

    async def _arun(self, query, runtime):
        self.calls.append(query)
        return "result", {
            "tool": f"{self.source}_search",
            "source": self.source,
            "args": {"query": query},
            "chunks": [],
            "latency_ms": 1.0,
            "error": "",
        }


def _run_model_middleware(node, preset):
    from TA.retrieval.middleware import RetrievalPolicyMiddleware

    tools = [
        SimpleNamespace(name="semantic_search", description="semantic"),
        SimpleNamespace(name="textbook_search", description="textbook"),
        SimpleNamespace(name="course_tree", description="roadmap"),
    ]
    request = _Request(
        state={"current_node": node},
        tools=tools,
        runtime=SimpleNamespace(
            context=resolve_retrieval_context(Retrieve_param.from_preset(preset))
        ),
    )

    async def handler(updated):
        return updated

    return asyncio.run(RetrievalPolicyMiddleware().awrap_model_call(request, handler))


def test_middleware_filters_roster_from_policy_contract():
    request = _run_model_middleware("RAG_Core", "RAG")
    context = resolve_retrieval_context(Retrieve_param.from_preset("RAG"))

    assert [tool.name for tool in request.tools] == ["semantic_search"]
    assert context.policy.prompt in str(request.system_message.content)
    assert "semantic_search: semantic" in str(request.system_message.content)
    assert "textbook_search" not in str(request.system_message.content)


def test_middleware_is_inert_outside_retrieval_node():
    request = _run_model_middleware("Roadmap_Explore", "RAG")

    assert [tool.name for tool in request.tools] == [
        "semantic_search",
        "textbook_search",
        "course_tree",
    ]
    assert request.system_message.content == "base"


def test_middleware_blocks_seventh_retrieval_call():
    from TA.retrieval.middleware import RetrievalPolicyMiddleware

    context = resolve_retrieval_context(Retrieve_param.from_preset("RAG"))
    calls = [
        {
            "name": "semantic_search",
            "args": {"query": str(index)},
            "id": str(index),
            "type": "tool_call",
        }
        for index in range(7)
    ]
    request = _Request(
        state={"current_node": "RAG_Core", "retrieval_call_count": 0},
        tools=[SimpleNamespace(name="semantic_search", description="semantic")],
        runtime=SimpleNamespace(context=context),
    )

    async def handler(updated):
        return ModelResponse(result=[AIMessage(content="", tool_calls=calls)])

    response = asyncio.run(RetrievalPolicyMiddleware().awrap_model_call(request, handler))
    accepted = response.result[0].tool_calls
    update = asyncio.run(
        RetrievalPolicyMiddleware().aafter_model(
            {"current_node": "RAG_Core", "messages": response.result},
            SimpleNamespace(context=context),
        )
    )

    assert [call["id"] for call in accepted] == [str(index) for index in range(6)]
    assert update["retrieval_call_count"] == 6


def test_middleware_counts_v2_seed_calls_against_the_same_six_call_budget():
    from TA.retrieval.middleware import RetrievalPolicyMiddleware
    from core.schema.retrieval import RetrievalHarnessId

    context = resolve_retrieval_context(
        Retrieve_param.from_preset("FULL", harness_id=RetrievalHarnessId.AGENTIC_V2)
    )
    calls = [
        {
            "name": "semantic_search",
            "args": {"query": str(index)},
            "id": str(index),
            "type": "tool_call",
        }
        for index in range(5)
    ]
    request = _Request(
        state={"current_node": "Retrieval_Aggregator", "retrieval_call_count": 2},
        tools=[SimpleNamespace(name="semantic_search", description="semantic")],
        runtime=SimpleNamespace(context=context),
    )

    async def handler(updated):
        return ModelResponse(result=[AIMessage(content="", tool_calls=calls)])

    response = asyncio.run(RetrievalPolicyMiddleware().awrap_model_call(request, handler))

    assert [call["id"] for call in response.result[0].tool_calls] == ["0", "1", "2", "3"]


def test_v3_middleware_exposes_only_deep_tool_and_accepts_one_call_per_turn():
    from TA.retrieval.middleware import RetrievalPolicyMiddleware
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId

    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )
    calls = [
        {
            "name": "retrieve_more",
            "args": {"query": f"hop-{index}", "sources": ["semantic"]},
            "id": str(index),
            "type": "tool_call",
        }
        for index in range(3)
    ]
    request = _Request(
        state={"current_node": "Retrieval_Ledger_Aggregator", "retrieval_call_count": 0},
        tools=[
            SimpleNamespace(name="retrieve_more", description="deep"),
            SimpleNamespace(name="semantic_search", description="raw"),
        ],
        runtime=SimpleNamespace(context=context),
    )

    async def handler(updated):
        assert [tool.name for tool in updated.tools] == ["retrieve_more"]
        return ModelResponse(result=[AIMessage(content="", tool_calls=calls)])

    response = asyncio.run(RetrievalPolicyMiddleware().awrap_model_call(request, handler))

    assert [call["id"] for call in response.result[0].tool_calls] == ["0"]
    assert response.result[0].additional_kwargs["blocked_retrieval_calls"] == 2

    update = asyncio.run(
        RetrievalPolicyMiddleware().aafter_model(
            {
                "current_node": "Retrieval_Ledger_Aggregator",
                "messages": response.result,
            },
            SimpleNamespace(context=context),
        )
    )
    assert update["blocked_retrieval_calls"] == 2


def test_middleware_injects_runtime_into_base_retrieval_tool():
    from TA.retrieval.middleware import RetrievalPolicyMiddleware
    from TA.tools.retrieval import SemanticSearch

    context = resolve_retrieval_context(Retrieve_param.from_preset("RAG"))
    request = _ToolRequest(
        tool_call={
            "name": "semantic_search",
            "args": {"query": "question"},
            "id": "tool-1",
            "type": "tool_call",
        },
        tool=SemanticSearch(milvus_db=_Milvus(), embedder=_Embedder()),
        state={"current_node": "RAG_Core"},
        runtime=SimpleNamespace(context=context),
    )

    async def handler(_):
        raise AssertionError("BaseTool handler must not lose ToolRuntime")

    result = asyncio.run(
        RetrievalPolicyMiddleware().awrap_tool_call(request, handler)
    )

    assert result.name == "semantic_search"
    assert result.tool_call_id == "tool-1"
    assert result.artifact["chunks"][0]["id"] == "p_1"


def test_v3_duplicate_query_is_noop_and_consumes_no_repository_call():
    from TA.retrieval.middleware import RetrievalPolicyMiddleware
    from TA.tools.retrieval import RetrieveMore
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId

    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )
    semantic, textbook = _Adapter("semantic"), _Adapter("textbook")
    tool = RetrieveMore(semantic=semantic, textbook=textbook)
    previous = ToolMessage(
        content="old",
        tool_call_id="old",
        name="retrieve_more",
        artifact={"args": {"query": "Bridge Query", "sources": ["semantic"]}},
    )
    request = _ToolRequest(
        tool_call={
            "name": "retrieve_more",
            "args": {"query": "  bridge   query ", "sources": ["semantic"]},
            "id": "new",
            "type": "tool_call",
        },
        tool=tool,
        state={"current_node": "Retrieval_Ledger_Aggregator", "messages": [previous]},
        runtime=SimpleNamespace(context=context),
    )

    async def handler(_):
        raise AssertionError("deep tool must be handled by retrieval middleware")

    result = asyncio.run(RetrievalPolicyMiddleware().awrap_tool_call(request, handler))

    assert result.artifact["repeated_query"] is True
    assert semantic.calls == [] and textbook.calls == []


def test_v3_empty_source_selection_defaults_to_arm_sources_and_is_traced():
    from TA.retrieval.middleware import RetrievalPolicyMiddleware
    from TA.tools.retrieval import RetrieveMore
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId

    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )
    semantic, textbook = _Adapter("semantic"), _Adapter("textbook")
    request = _ToolRequest(
        tool_call={
            "name": "retrieve_more",
            "args": {"query": "enough evidence", "sources": []},
            "id": "stop",
            "type": "tool_call",
        },
        tool=RetrieveMore(semantic=semantic, textbook=textbook),
        state={"current_node": "Retrieval_Ledger_Aggregator", "messages": []},
        runtime=SimpleNamespace(context=context),
    )

    async def handler(_):
        raise AssertionError("deep tool must be handled by retrieval middleware")

    result = asyncio.run(RetrievalPolicyMiddleware().awrap_tool_call(request, handler))

    assert result.artifact["requested_sources"] == []
    assert result.artifact["executed_sources"] == ["semantic", "textbook"]
    assert result.artifact["defaulted_sources"] is True
    assert result.artifact["error"] == ""
    assert semantic.calls == ["enough evidence"]
    assert textbook.calls == ["enough evidence"]


def test_v3_unknown_source_is_mechanically_rejected_without_crashing():
    from TA.retrieval.middleware import RetrievalPolicyMiddleware
    from TA.tools.retrieval import RetrieveMore
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId

    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )
    semantic, textbook = _Adapter("semantic"), _Adapter("textbook")
    request = _ToolRequest(
        tool_call={
            "name": "retrieve_more",
            "args": {"query": "external search", "sources": ["web"]},
            "id": "invalid",
            "type": "tool_call",
        },
        tool=RetrieveMore(semantic=semantic, textbook=textbook),
        state={"current_node": "Retrieval_Ledger_Aggregator", "messages": []},
        runtime=SimpleNamespace(context=context),
    )

    async def handler(_):
        raise AssertionError("deep tool must be handled by retrieval middleware")

    result = asyncio.run(RetrievalPolicyMiddleware().awrap_tool_call(request, handler))

    assert result.artifact["invalid_sources"] == ["web"]
    assert result.artifact["executed_sources"] == []
    assert result.artifact["error"] == ""
    assert semantic.calls == [] and textbook.calls == []


def test_v3_context_guard_stops_before_worst_case_round():
    from TA.retrieval.middleware import RetrievalPolicyMiddleware
    from TA.tools.retrieval import RetrieveMore
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId

    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )
    semantic, textbook = _Adapter("semantic"), _Adapter("textbook")
    request = _ToolRequest(
        tool_call={
            "name": "retrieve_more",
            "args": {"query": "next", "sources": ["semantic", "textbook"]},
            "id": "next",
            "type": "tool_call",
        },
        tool=RetrieveMore(semantic=semantic, textbook=textbook),
        state={
            "current_node": "Retrieval_Ledger_Aggregator",
            "messages": [HumanMessage(content="x" * 150_000)],
        },
        runtime=SimpleNamespace(context=context),
    )

    async def handler(_):
        raise AssertionError("deep tool must be handled by retrieval middleware")

    result = asyncio.run(RetrievalPolicyMiddleware().awrap_tool_call(request, handler))

    assert result.artifact["context_limit"] is True
    assert semantic.calls == [] and textbook.calls == []
