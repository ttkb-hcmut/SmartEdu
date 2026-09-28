import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from core.config import Retrieve_param
from core.schema.retrieval import (
    RetrievalCase, RetrievalCaseKind, RetrievalHarnessId, RetrievalPolicyId, RetrievalRoute,
    RetrievalValidation, RetrievalValidity,
)
from TA.workflow.smart_edu import SmartEdu
from TA.retrieval.policy import resolve_retrieval_context
from TA.helper.schema import BenchmarkAnswer, TeachLectureOutput
from TA.helper.context import EvidenceContextPolicy, build_ui_citations, compile_evidence_context
from TA.retrieval.controller import _envelope
from TA.retrieval.schema import ClaimSupport, FinalChain
from TA.tools.retrieval import _chunk, append_ledger, rank_round
from TA.workflow import teach as teach_module
from TA.workflow.teach import build_teach_wf, teach_lecture, teach_rag, teach_understand
from knowledge.api.route import get_raw_pdf
from student.memo import Chat, ChatMessage, Memo


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


def test_evidence_context_prioritizes_cited_uris_then_ledger_order():
    compiled = compile_evidence_context(
        ledger=[
            {"uri": "u1", "text": "first"},
            {"uri": "u2", "text": "second"},
            {"uri": "u3", "text": "third"},
        ],
        chain=FinalChain(
            answerable=True,
            claims=[ClaimSupport(claim="claim", evidence_uris=["u2", "u1"])],
        ),
        policy=EvidenceContextPolicy(excerpt_chars=20, max_chars=200),
    )

    assert compiled.uris == ("u2", "u1", "u3")
    assert compiled.text.index("[u2]") < compiled.text.index("[u1]") < compiled.text.index("[u3]")
    assert compiled.truncated is False


def test_evidence_context_deduplicates_and_reports_hard_budget():
    compiled = compile_evidence_context(
        ledger=[
            {"uri": "u1", "text": "a" * 50},
            {"uri": "u1", "text": "duplicate"},
            {"uri": "u2", "text": "b" * 50},
        ],
        chain=FinalChain(answerable=False),
        policy=EvidenceContextPolicy(excerpt_chars=50, max_chars=25),
    )

    assert compiled.uris == ("u1",)
    assert len(compiled.text) <= 25
    assert compiled.truncated is True


def _chat(chat_id: str, student: str, ta: str, heading: str) -> Chat:
    return Chat(
        id=chat_id,
        messages=[
            ChatMessage(role="student", heading="Student Query", message=student, timestamp="now"),
            ChatMessage(role="TA", heading=heading, message=ta, timestamp="now"),
        ],
    )


def test_history_defaults_stay_compatible_and_recent_turns_skim_old_ta():
    memo = Memo("session")
    memo.session.chats = [
        _chat("old", "old question", "old full answer", "old heading"),
        _chat("new", "new question", "new full answer", "new heading"),
    ]

    history = memo.get_formatted_history(recent_turns=1)

    assert "Student: old question" in history
    assert "TA: old heading" in history
    assert "old full answer" not in history
    assert "TA: new full answer" in history


def test_history_can_exclude_current_chat_and_hard_cap_recent_context():
    memo = Memo("session")
    memo.session.chats = [
        _chat(f"chat-{index}", f"question-{index}", f"answer-{index}", f"heading-{index}")
        for index in range(10)
    ]

    history = memo.get_formatted_history(
        mode="skim",
        exclude_chat_id="chat-9",
        max_chars=80,
    )

    assert len(history) <= 80
    assert "question-9" not in history
    assert "question-8" in history
    assert "question-0" not in history


def test_teach_history_helper_uses_same_bounded_recent_context_contract():
    from TA.workflow.teach import _bounded_history

    class _Tracker:
        def get_chat_history(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return "bounded"

    tracker = _Tracker()

    assert _bounded_history(tracker, "session", "chat") == "bounded"
    assert tracker.args == ("session",)
    assert tracker.kwargs == {
        "mode": "skim",
        "recent_turns": 4,
        "exclude_chat_id": "chat",
        "max_chars": 4_000,
    }


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
            model=SimpleNamespace(model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0.0)
        ),
        "RETRIEVAL_ANSWERER": SimpleNamespace(
            model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0.0, num_ctx=65536
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
    assert manifest["Retrieval_Ledger_Aggregator"]["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert manifest["TA_Retrieve_Finish"]["num_ctx"] == 65536


def test_v4_observability_reports_typed_node_contracts():
    from TA.observability import build_node_config_manifest

    model = SimpleNamespace(model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0.0, num_ctx=65536)
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
    assert manifest["Plan_Hop"]["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert manifest["Plan_Hop"]["timeout_s"] == 120
    assert manifest["Plan_Hop"]["max_retries"] == 1
    assert manifest["Finalize_Chain"]["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert manifest["Finalize_Chain"]["answer_context_chars"] == 48_000


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
    model = "nvidia/nemotron-3-super-120b-a12b:free"
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


def _v4_context():
    return resolve_retrieval_context(Retrieve_param.from_preset(
        "FULL", policy_id=RetrievalPolicyId.BASELINE_V5,
        harness_id=RetrievalHarnessId.AGENTIC_V4, course_scope="Course",
    ))


def test_two_passages_same_pdf_keep_distinct_ids_and_cited_page():
    rows = [
        {"id": "passage-1", "uri": "Course/_raw/book.pdf", "text": "one", "p_lo": 3},
        {"id": "passage-2", "uri": "Course/_raw/book.pdf", "text": "two", "p_lo": 9},
    ]
    chunks = [_chunk(row, "textbook") for row in rows]
    ranked = rank_round({"textbook": chunks}, 60)
    ledger, _, _ = append_ledger([], ranked, round_index=0, query="q")
    chain = FinalChain(claims=[ClaimSupport(claim="two", evidence_uris=["passage-2"])], answerable=True)
    rag = _envelope(ledger, chain, RetrievalValidation(RetrievalValidity.VALID, 1), _v4_context(), {})

    assert [item["uri"] for item in ledger] == ["passage-1", "passage-2"]
    assert [item["uri"] for item in rag["cited_evidence"]] == ["passage-2"]
    assert "text" not in rag["cited_evidence"][0]
    assert asyncio.run(build_ui_citations(rag)) == [
        {"uri": "passage-2", "document": "Course/_raw/book.pdf", "page": 9}
    ]


def test_semantic_citations_resolve_hard_ref_then_anchor_and_keep_unlinked():
    class Graph:
        def resolve_entity_sources(self, ids):
            assert ids == ["slide", "book", "missing"]
            return [
                {"id": "slide", "hard_ref": json.dumps({
                    "id": "minio://bucket/Course/topic/chunks/c.txt", "p_num": [11, 12]
                }), "uri": "Course/_raw/fallback.pdf", "p_lo": 5},
                {"id": "book", "hard_ref": None, "uri": "Course/_raw/book.pdf", "p_lo": 7},
            ]

    rag = {"cited_evidence": [
        {"uri": uri, "source": "semantic"} for uri in ("slide", "book", "missing")
    ]}
    assert asyncio.run(build_ui_citations(rag, Graph())) == [
        {"uri": "slide", "document": "Course/topic/page.pdf", "page": 1},
        {"uri": "book", "document": "Course/_raw/book.pdf", "page": 7},
        {"uri": "missing", "document": None, "page": None},
    ]


def test_teach_uses_v4_context_and_falls_back_to_general_knowledge():
    class Tracker:
        def get_student_state(self, _sid):
            return {"current_pos": SimpleNamespace(name="Matrices")}

    class Tracer:
        def __init__(self):
            self.steps = []

        def log_step(self, **kwargs):
            self.steps.append(kwargs)

    class Retrieve:
        def __init__(self, fail=False, answerable=True):
            self.fail = fail
            self.answerable = answerable
            self.context = None

        async def ainvoke(self, state, *, config, context):
            self.context = context
            assert state["user_query"] == "Explain the concept of Matrices in detail."
            assert len(state["messages"]) == 1
            if self.fail:
                raise RuntimeError("offline")
            return {"worker_results": {"RAG": {
                "status": "SUCCESS", "answerable": self.answerable, "content": "matrix evidence",
                "cited_evidence": [{"uri": "p1", "source": "textbook",
                                    "document_uri": "Course/_raw/book.pdf", "p_lo": 4}],
            }}}

    state = {"messages": [{"role": "user", "content": "old"}], "worker_results": {}, "_teach_mode": "continue"}
    tracer = Tracer()
    config = {"configurable": {"session_id": "s", "student_tracker": Tracker(),
                               "tracer": tracer, "chat_id": "c"}}
    runtime = SimpleNamespace(context=resolve_retrieval_context(Retrieve_param.from_preset(
        "FULL", course_scope="Course"
    )))
    retrieve = Retrieve()
    result = asyncio.run(teach_rag(state, retrieve, None, config, runtime))
    assert retrieve.context.harness.id is RetrievalHarnessId.AGENTIC_V4
    assert retrieve.context.policy.id is RetrievalPolicyId.BASELINE_V5
    assert retrieve.context.scope.course == "Course"
    assert runtime.context.harness.id is RetrievalHarnessId.AGENTIC_V2
    assert result["ui_action"]["citations"][0]["page"] == 4
    assert result["_teach_context"]["content"] == "matrix evidence"
    assert tracer.steps[-1]["execution_config"]["harness_id"] == "agentic-v4"
    assert tracer.steps[-1]["execution_config"]["policy_id"] == "baseline-v5"

    failed = asyncio.run(teach_rag(state, Retrieve(fail=True), None, config, runtime))
    assert failed["_teach_context"]["source"] == "GENERAL"
    assert failed["ui_action"] is None
    insufficient = asyncio.run(teach_rag(state, Retrieve(answerable=False), None, config, runtime))
    assert insufficient["_teach_context"]["content"] == ""
    assert insufficient["ui_action"] is None
    assert "Teach_Lookup" not in build_teach_wf({"TA": object()}, retrieve).get_graph().nodes


def test_teach_without_current_concept_retrieves_student_question():
    class Retrieve:
        async def ainvoke(self, state, *, config, context):
            assert state["user_query"] == "Compare classification and regression."
            assert state["messages"] == [{"role": "user", "content": state["user_query"]}]
            return {"worker_results": {"RAG": {"status": "FAIL"}}}

    tracker = SimpleNamespace(get_student_state=lambda _sid: {"current_pos": None})
    config = {"configurable": {"session_id": "s", "student_tracker": tracker}}
    runtime = SimpleNamespace(context=resolve_retrieval_context(Retrieve_param.from_preset("FULL")))
    result = asyncio.run(teach_rag(
        {"user_query": "Compare classification and regression.", "worker_results": {}},
        Retrieve(), None, config, runtime,
    ))
    assert result["_teach_context"]["source"] == "GENERAL"


def test_teach_keeps_specific_question_and_passes_it_into_full_lesson_prompt(monkeypatch):
    query = "Compare classification and regression with examples."
    monkeypatch.setattr(
        teach_module.prompt_lib,
        "TEACH_CONTINUE_PROMPT",
        "CURRENT TOPIC: {current_node}\nSOURCE: {source}\nCONTENT: {content}\nHISTORY: {history}",
    )

    class Tracker:
        def get_student_state(self, _sid):
            return {"current_pos": SimpleNamespace(name="Matrices")}

        def get_chat_history(self, *_args, **_kwargs):
            return "No prior context"

        def get_session(self, _sid):
            return SimpleNamespace(student_state={"current_pos": SimpleNamespace(name="Matrices")})

    class Retrieve:
        async def ainvoke(self, state, *, config, context):
            assert state["user_query"] == query
            return {"worker_results": {"RAG": {"status": "FAIL"}}}

    class Agent:
        name = "TA"
        system_prompt_text = "TA teaching system prompt"

        class Model:
            def with_structured_output(self, schema):
                assert schema is TeachLectureOutput
                return self

            async def ainvoke(self, messages, *, config):
                agent.prompt = messages[-1][1]
                assert messages[0] == ("system", "TA teaching system prompt")
                return TeachLectureOutput(
                    thought="", lecture="Lesson", challenge_question="What differs?"
                )

        model = Model()

        async def ainvoke(self, *_args, **_kwargs):
            raise AssertionError("teaching must not call PDF lookup tools")

    tracker = Tracker()
    config = {"configurable": {"session_id": "s", "student_tracker": tracker}}
    runtime = SimpleNamespace(context=resolve_retrieval_context(
        Retrieve_param.from_preset("FULL")
    ))
    state = {"user_query": query, "worker_results": {}, "language": "eng"}
    asyncio.run(teach_rag(state, Retrieve(), None, config, runtime))

    agent = Agent()
    asyncio.run(teach_lecture({
        **state,
        "_teach_mode": "continue",
        "_teach_context": {"source": "GENERAL", "content": "", "mode": "continue"},
    }, agent, config))

    assert query in agent.prompt, agent.prompt
    assert "full lesson" in agent.prompt.lower()
    assert "classification" in agent.prompt.lower()
    assert "regression" in agent.prompt.lower()


def test_teach_finalizer_preserves_the_complete_lesson():
    query = "Teach classification and regression with examples."
    lecture = "## Classification\nA detailed example.\n\n## Regression\nAnother detailed example."
    tracker = SimpleNamespace(
        get_session=lambda _sid: SimpleNamespace(student_state={}),
        mongodb=SimpleNamespace(),
    )
    engine = object.__new__(SmartEdu)
    engine.agents = {"TA": SimpleNamespace(name="TA")}
    engine._stream_answer = AsyncMock()
    engine._bg_save = AsyncMock()
    config = {"configurable": {"session_id": "s", "student_tracker": tracker, "chat_id": "c"}}

    result = asyncio.run(engine.ta_teach_finish({
        "user_query": query,
        "worker_results": {"Teach_Lecture": lecture},
    }, config))

    assert result["messages"][0].content == lecture
    engine._stream_answer.assert_not_awaited()
    engine._bg_save.assert_awaited_once()


def test_teach_finalizer_raises_on_empty_lecture_instead_of_stringifying_fallback_dict():
    ## regression: str({}) is truthy "{}" -- must check the value, not its str(), or "{}" leaks to the student
    tracker = SimpleNamespace(
        get_session=lambda _sid: SimpleNamespace(student_state={}),
        mongodb=SimpleNamespace(),
    )
    engine = object.__new__(SmartEdu)
    engine.agents = {"TA": SimpleNamespace(name="TA")}
    engine._stream_answer = AsyncMock()
    engine._bg_save = AsyncMock()
    config = {"configurable": {"session_id": "s", "student_tracker": tracker, "chat_id": "c"}}

    with pytest.raises(RuntimeError, match="without a lecture"):
        asyncio.run(engine.ta_teach_finish({
            "user_query": "Teach me something.",
            "worker_results": {"Teach_Lecture": ""},  ## e.g. structured-output parse failed and auto-defaulted to ""
        }, config))


def test_router_recovers_explicit_lesson_intent_if_model_output_is_truncated(monkeypatch):
    class Model:
        async def ainvoke(self, *_args, **_kwargs):
            return SimpleNamespace(content="I should classify this as a guided class")

    tracker = SimpleNamespace(
        get_student_state=lambda _sid: {},
        get_chat_history=lambda *_args, **_kwargs: "",
    )
    engine = object.__new__(SmartEdu)
    engine.agents = {"TA": SimpleNamespace(model=object())}
    generation = {}

    def bind_router(_model, _temperature, max_tokens):
        generation["max_tokens"] = max_tokens
        return Model()

    monkeypatch.setattr(SmartEdu, "_bind_generation", staticmethod(bind_router))
    runtime = SimpleNamespace(context=resolve_retrieval_context(
        Retrieve_param.from_preset("FULL")
    ))
    result = asyncio.run(engine.ta_router_node({
        "user_query": "Hãy tạo một bài giảng chi tiết và câu hỏi ôn tập.",
        "language": "vn",
    }, {"configurable": {"session_id": "s", "student_tracker": tracker}}, runtime))

    assert result["intent"] == "teaching"
    assert generation["max_tokens"] == 1024


def test_unknown_finish_marks_its_reply_successful():
    tracker = SimpleNamespace(
        get_student_state=lambda _sid: {},
        get_chat_history=lambda *_args, **_kwargs: "",
    )
    engine = object.__new__(SmartEdu)
    engine.agents = {"TA": SimpleNamespace(name="TA")}
    engine._stream_answer = AsyncMock(return_value="Clarification")
    engine._bg_save = AsyncMock()
    result = asyncio.run(engine.ta_unknown_finish(
        {"user_query": "off-topic", "language": "eng"},
        {"configurable": {"session_id": "s", "student_tracker": tracker, "chat_id": "c"}},
    ))

    assert result["messages"][0].content == "Clarification"
    assert result["status_flag"] == "SUCCESS"


def test_teach_understand_invokes_raw_model():
    class Model:
        async def ainvoke(self, messages, config):
            assert messages[0][0] == "user"
            assert "supervised learning" in messages[0][1]
            return SimpleNamespace(content="continue")

    class Agent:
        name = "TA"
        model = Model()

        async def ainvoke(self, *_args, **_kwargs):
            raise AssertionError("compiled agent requires a state dictionary")

    tracker = SimpleNamespace(get_chat_history=lambda *_args, **_kwargs: "No prior context")
    config = {"configurable": {"session_id": "test-session", "student_tracker": tracker}}
    result = asyncio.run(teach_understand(
        {"user_query": "Teach supervised learning", "language": "eng"}, Agent(), config,
    ))
    assert result == {"_teach_mode": "continue"}


def test_retrieve_finish_persists_only_final_ta_message():
    def unexpected_summary(*_args, **_kwargs):
        raise AssertionError("summary must not appear as a separate TA message")

    tracker = SimpleNamespace(
        get_session=lambda _sid: SimpleNamespace(student_state={}),
        mongodb=SimpleNamespace(push_chat_message=unexpected_summary),
    )
    engine = object.__new__(SmartEdu)
    engine.agents = {"TA": SimpleNamespace(name="TA")}
    engine.retrieve_res = {}
    engine._benchmark_answer = AsyncMock(return_value="Final answer")
    engine._bg_save = AsyncMock()
    config = {"configurable": {
        "session_id": "s", "student_id": "u", "student_tracker": tracker, "chat_id": "c",
    }}
    runtime = SimpleNamespace(context=resolve_retrieval_context(
        Retrieve_param.from_preset("FULL"), RetrievalCase(kind=RetrievalCaseKind.BENCHMARK),
    ))
    result = asyncio.run(engine.ta_retrieve_finish({"worker_results": {}}, config, runtime))
    assert result["messages"][0].content == "Final answer"
    engine._bg_save.assert_awaited_once()


def test_default_case_is_live_and_streams_instead_of_benchmark_answer():
    ## regression: default kind was BENCHMARK, so real students got the short MuSiQue "unknown" answer
    assert RetrievalCase().kind is RetrievalCaseKind.LIVE
    tracker = SimpleNamespace(
        get_session=lambda _sid: SimpleNamespace(student_state={}),
        mongodb=SimpleNamespace(),
    )
    engine = object.__new__(SmartEdu)
    engine.agents = {"TA": SimpleNamespace(name="TA")}
    engine.retrieve_res = {}
    engine._benchmark_answer = AsyncMock(side_effect=AssertionError("live must not use benchmark answerer"))
    engine._stream_answer = AsyncMock(return_value="Streamed lesson")
    engine._bg_save = AsyncMock()
    config = {"configurable": {
        "session_id": "s", "student_id": "u", "student_tracker": tracker, "chat_id": "c",
    }}
    runtime = SimpleNamespace(context=resolve_retrieval_context(Retrieve_param.from_preset("FULL")))
    result = asyncio.run(engine.ta_retrieve_finish({"worker_results": {}}, config, runtime))
    assert result["messages"][0].content == "Streamed lesson"


def test_teach_understand_retries_provider_overloaded(monkeypatch):
    async def no_sleep(_s):
        pass
    monkeypatch.setattr("TA.helper.model_call.asyncio.sleep", no_sleep)

    class Model:
        calls = 0

        async def ainvoke(self, messages, config):
            Model.calls += 1
            if Model.calls == 1:
                raise RuntimeError("Response validation failed ... 'provider_overloaded'")
            return SimpleNamespace(content="review")

    agent = SimpleNamespace(name="TA", model=Model())
    tracker = SimpleNamespace(get_chat_history=lambda *_a, **_k: "No prior context")
    config = {"configurable": {"session_id": "s", "student_tracker": tracker}}
    result = asyncio.run(teach_understand({"user_query": "quiz me", "language": "eng"}, agent, config))
    assert result == {"_teach_mode": "review"}
    assert Model.calls == 2


def test_teach_evaluate_raises_on_overload_instead_of_defaulting_to_failed(monkeypatch):
    ## regression: outage was caught, auto-defaulted to passed=False, and written to mastery
    async def no_sleep(_s):
        pass
    monkeypatch.setattr("TA.helper.model_call.asyncio.sleep", no_sleep)

    class Structured:
        async def ainvoke(self, messages, config):
            raise RuntimeError("'provider_overloaded'")

    class Model:
        def with_structured_output(self, _schema):
            return Structured()

    agent = SimpleNamespace(name="TA", model=Model())
    tracker = SimpleNamespace(
        get_chat_history=lambda *_a, **_k: "No prior context",
        get_session=lambda _sid: SimpleNamespace(student_state={}),
    )
    config = {"configurable": {"session_id": "s", "student_tracker": tracker, "chat_id": "c"}}
    with pytest.raises(RuntimeError, match="provider_overloaded"):
        asyncio.run(teach_module.teach_evaluate({"user_query": "my answer", "language": "eng"}, agent, config))


def test_raw_pdf_route_validates_file_and_serves_existing_object():
    class Minio:
        def raw_object_name(self, course, file):
            return f"{course}/_raw/{file}"

        def object_exists(self, key):
            return key == "Course/_raw/book.pdf"

        def get_object_bytes(self, key):
            assert key == "Course/_raw/book.pdf"
            return b"%PDF"

    service = SimpleNamespace(minio_repo=Minio())
    response = asyncio.run(get_raw_pdf("Course", "book.pdf", service=service, _=object()))
    assert response.body == b"%PDF"
    with pytest.raises(HTTPException) as invalid:
        asyncio.run(get_raw_pdf("Course", "../book.pdf", service=service, _=object()))
    assert invalid.value.status_code == 400
    with pytest.raises(HTTPException) as missing:
        asyncio.run(get_raw_pdf("Course", "missing.pdf", service=service, _=object()))
    assert missing.value.status_code == 404
