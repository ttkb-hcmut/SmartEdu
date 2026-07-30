import asyncio
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, SystemMessage

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
    request = _run_model_middleware("Roadmap_Explore", "PLAIN")

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
