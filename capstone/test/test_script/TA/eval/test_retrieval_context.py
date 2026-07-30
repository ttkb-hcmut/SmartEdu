import asyncio
from types import SimpleNamespace

from core.config import Retrieve_param
from core.schema.retrieval import RetrievalCase, RetrievalRoute
from TA.workflow.smart_edu import SmartEdu
from TA.retrieval.policy import resolve_retrieval_context


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
