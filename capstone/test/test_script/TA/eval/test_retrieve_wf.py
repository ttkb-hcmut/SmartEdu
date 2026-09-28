"""Fan-out retrieve workflow — routing, fusion, chunk normalization. No DBs."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.config import Retrieve_param
from core.schema.retrieval import (
    RetrievalHarnessId,
    RetrievalPolicyId,
    RetrievalPreset,
    RetrievalToolId,
)
from TA.helper.schema import RAGCore
from TA.retrieval.policy import resolve_retrieval_context
from TA.workflow.retrieve import build_retrieve_wf, rrf_merge, _norm_chunk
from TA.tracing.tracer import AgentTracer


# ── rrf_merge ─────────────────────────────────────────────────────────

def _c(uri, source="rag"):
    return {"id": uri, "uri": uri, "text": "t", "score": 1.0, "source": source}


def test_rrf_cross_pool_agreement_wins():
    merged = rrf_merge({"semantic": [_c("a"), _c("b")],
                        "textbook": [_c("b", "textbook"), _c("c", "textbook")]},
                       rrf_k=60, top_k=5)
    assert [c["uri"] for c in merged][0] == "b"
    assert len(merged) == 3  # deduped by uri


def test_rrf_empty_pools():
    assert rrf_merge({}, rrf_k=60, top_k=5) == []
    assert rrf_merge({"rag": []}, rrf_k=60, top_k=5) == []


def test_rrf_caps_at_top_k():
    pool = {"rag": [_c(f"u{i}") for i in range(10)]}
    assert len(rrf_merge(pool, rrf_k=60, top_k=3)) == 3


def test_norm_chunk_uri_falls_back_to_id():
    n = _norm_chunk({"id": "x1", "text": "t", "score": "0.5"}, "rag")
    assert n["uri"] == "x1" and n["score"] == 0.5 and n["source"] == "rag"


# ── graph routing per preset (components no-op without resources) ─────

def _invoke(preset):
    wf = build_retrieve_wf(agents={}, resources=None)
    tracer = AgentTracer(session_id=f"test_wf_{preset}")
    chat_id = tracer.begin_chat("q")
    state = {"messages": [HumanMessage(content="q")], "user_query": "q",
             "worker_results": {}, "language": "eng"}
    cfg = {"configurable": {"tracer": tracer, "chat_id": chat_id}}
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            preset,
            harness_id=RetrievalHarnessId.FANOUT_V1,
        )
    )
    out = asyncio.run(wf.ainvoke(state, config=cfg, context=context))
    nodes = [s.node for s in tracer._active_chats[chat_id].agent]
    return out, nodes


def test_rag_runs_single_component():
    _, nodes = _invoke("RAG")
    assert "Comp_Semantic" in nodes and "Comp_Textbook" not in nodes
    assert nodes.count("Fusion") == 1


def test_full_fans_out_both_components():
    _, nodes = _invoke("FULL")
    assert "Comp_Semantic" in nodes and "Comp_Textbook" in nodes
    assert nodes.count("Fusion") == 1  # fan-in runs fusion once


# ── preset semantics ──────────────────────────────────────────────────

def test_presets_round_trip():
    for name in ["RAG", "FULL"]:
        assert Retrieve_param.from_preset(name).preset is RetrievalPreset[name]


def test_preset_keeps_scope_selection_only():
    rp = Retrieve_param.from_preset("RAG", course_scope="Bench_MuSiQue")
    assert rp.preset is RetrievalPreset.RAG
    assert rp.course_scope == "Bench_MuSiQue"


class _FakeRagAgent:
    name = "RAG"
    model = SimpleNamespace(model="qwen3:8b", temperature=0.0)

    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    async def ainvoke(self, state, config, context):
        self.calls.append((state, config, context))
        messages = [
            ToolMessage(
                content="result",
                tool_call_id=f"call-{index}",
                name=artifact["tool"],
                artifact=artifact,
            )
            for index, artifact in enumerate(self.artifacts)
        ]
        return {
            "messages": messages,
            "structured_response": RAGCore(
                thought="agent thought",
                entity_ids=["entity"],
                content="agent content",
                status="FAIL",
            ),
        }


class _FakeAggregator:
    name = "RAG_Aggregator"
    model = SimpleNamespace(model="qwen3:8b", temperature=0.0)

    def __init__(self, artifacts=()):
        self.artifacts = artifacts
        self.calls = []

    async def ainvoke(self, state, config, context):
        self.calls.append((state, config, context))
        return {
            "messages": [
                ToolMessage(
                    content="result",
                    tool_call_id=f"call-{index}",
                    name=artifact["tool"],
                    artifact=artifact,
                )
                for index, artifact in enumerate(self.artifacts)
            ]
        }


class _FakeLedgerAggregator:
    name = "RAG_Ledger_Aggregator"
    model = SimpleNamespace(model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0.0)

    def __init__(self, artifacts=(), blocked_calls=0):
        self.artifacts = artifacts
        self.blocked_calls = blocked_calls
        self.calls = []

    async def ainvoke(self, state, config, context):
        self.calls.append((state, config, context))
        messages = [
            ToolMessage(
                content="follow-up evidence",
                tool_call_id=f"ledger-{index}",
                name="retrieve_more",
                artifact=artifact,
            )
            for index, artifact in enumerate(self.artifacts)
        ]
        messages.append(
            AIMessage(
                content="Scottish Parliament is supported; Holyrood is its location.",
                additional_kwargs={"blocked_retrieval_calls": self.blocked_calls},
            )
        )
        return {"messages": messages}


class _SeedMilvus:
    def search(self, **_):
        return [{"id": "semantic-1", "uri": "semantic-1", "text": "semantic", "score": 1.0}]


class _SeedEmbedder:
    def get_embedding(self, _):
        return [0.0]


class _SeedGraph:
    def passage_search(self, *_args, **_kwargs):
        return [{"id": "textbook-1", "uri": "textbook-1", "text": "textbook", "score": 1.0}]


class _StructuredPlanner:
    """baseline-v5's model_output_mode is STRUCTURED: typed_call requests
    include_raw=True and normalizes+validates the raw completion itself,
    since with_structured_output's own field names drift from our schema
    (observed live: "type" not "action", singular "source" not "sources")."""

    model = "nvidia/nemotron-3-super-120b-a12b:free"
    temperature = 0.0

    def __init__(self, decisions, final_chain):
        self.decisions = iter(decisions)
        self.final_chain = final_chain
        self.calls = []

    def with_structured_output(self, schema, **_kwargs):
        planner = self

        class _Invocation:
            async def ainvoke(self, messages, config=None):
                planner.calls.append((schema, messages, config))
                value = next(planner.decisions) if schema.__name__ == "HopDecision" else planner.final_chain
                payload = value if isinstance(value, dict) else value.model_dump(mode="json")
                return {
                    "raw": SimpleNamespace(content=json.dumps(payload)),
                    "parsed": None,
                    "parsing_error": None,
                }

        return _Invocation()


class _V4Milvus:
    def __init__(self):
        self.queries = []

    def search(self, query, **_kwargs):
        self.queries.append(query)
        suffix = "seed" if query == "question" else "bridge"
        return [{
            "id": f"semantic-{suffix}",
            "uri": f"semantic-{suffix}",
            "text": f"{suffix} evidence",
            "score": 1.0,
        }]


def _invoke_agentic(preset, artifacts):
    agent = _FakeRagAgent(artifacts)
    wf = build_retrieve_wf(agents={"RAG": agent}, resources=None)
    state = {
        "messages": [HumanMessage(content="q")],
        "user_query": "q",
        "worker_results": {},
        "language": "eng",
    }
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(preset, harness_id=RetrievalHarnessId.AGENTIC_V1)
    )
    out = asyncio.run(wf.ainvoke(state, context=context))
    return out, agent


def test_agentic_harness_injects_context_and_uses_contract_recursion_limit():
    artifact = {
        "tool": "semantic_search",
        "source": "semantic",
        "args": {"query": "q"},
        "chunks": [],
        "error": "",
    }

    out, agent = _invoke_agentic("RAG", [artifact])

    assert agent.calls[0][1]["recursion_limit"] == 20
    assert agent.calls[0][2].harness.id is RetrievalHarnessId.AGENTIC_V1
    assert out["worker_results"]["RAG"]["validity"] == "valid"
    assert out["worker_results"]["RAG"]["status"] == "SUCCESS"
    assert out["worker_results"]["RAG"]["agent_status"] == "FAIL"


def test_agentic_required_arm_without_attempt_is_policy_invalid():
    out, _ = _invoke_agentic("FULL", [])

    assert out["worker_results"]["RAG"]["validity"] == "policy-invalid"
    assert out["worker_results"]["RAG"]["status"] == "FAIL"
    assert out["status_flag"] == "SUCCESS"


def test_agentic_model_mismatch_is_invalid():
    artifact = {
        "tool": "semantic_search",
        "source": "semantic",
        "args": {"query": "q"},
        "chunks": [],
        "error": "",
    }
    agent = _FakeRagAgent([artifact])
    agent.model = SimpleNamespace(model="different-model", temperature=0.0)
    wf = build_retrieve_wf(agents={"RAG": agent}, resources=None)
    state = {
        "messages": [HumanMessage(content="q")],
        "user_query": "q",
        "worker_results": {},
        "language": "eng",
    }
    context = resolve_retrieval_context(
        Retrieve_param.from_preset("RAG", harness_id=RetrievalHarnessId.AGENTIC_V1)
    )

    out = asyncio.run(wf.ainvoke(state, context=context))

    assert out["worker_results"]["RAG"]["validity"] == "invalid"
    assert "model mismatch" in out["worker_results"]["RAG"]["errors"][0]


def test_harnesses_return_identical_rag_envelopes():
    agentic, _ = _invoke_agentic("RAG", [])
    fanout, _ = _invoke("RAG")

    assert set(agentic["worker_results"]["RAG"]) == set(fanout["worker_results"]["RAG"])


def test_agentic_v2_seeds_full_then_allows_tool_only_expansion():
    extra = {
        "tool": "semantic_search",
        "source": "semantic",
        "args": {"query": "focused"},
        "chunks": [],
        "error": "",
    }
    aggregator = _FakeAggregator([extra])
    resources = {
        "milvus_db": _SeedMilvus(),
        "graph_db": _SeedGraph(),
        "embedder": _SeedEmbedder(),
    }
    wf = build_retrieve_wf(agents={"RAG_AGGREGATOR": aggregator}, resources=resources)
    state = {
        "messages": [HumanMessage(content="q")],
        "user_query": "q",
        "worker_results": {},
        "language": "eng",
    }
    context = resolve_retrieval_context(
        Retrieve_param.from_preset("FULL", harness_id=RetrievalHarnessId.AGENTIC_V2)
    )
    tracer = AgentTracer(session_id="agentic-v2")
    chat_id = tracer.begin_chat("q")

    out = asyncio.run(
        wf.ainvoke(
            state,
            context=context,
            config={"configurable": {"tracer": tracer, "chat_id": chat_id}},
        )
    )

    assert aggregator.calls[0][0]["current_node"] == "Retrieval_Aggregator"
    assert aggregator.calls[0][0]["retrieval_call_count"] == 2
    assert "semantic" in aggregator.calls[0][0]["messages"][0][1]
    assert "textbook" in aggregator.calls[0][0]["messages"][0][1]
    assert aggregator.calls[0][1]["recursion_limit"] == 20
    assert out["worker_results"]["RAG"]["retrieval_attempts"] == 3
    assert out["worker_results"]["RAG"]["validity"] == "valid"
    assert out["worker_results"]["RAG"]["status"] == "SUCCESS"
    aggregation = next(step for step in tracer._active_chats[chat_id].agent if step.node == "Retrieval_Aggregator")
    assert aggregation.tool_result["seed_calls"] == 2
    assert aggregation.tool_result["aggregator_calls"] == 1
    assert aggregation.latency_ms >= 0


def test_agentic_v2_accepts_seed_without_extra_model_tool_calls():
    aggregator = _FakeAggregator()
    resources = {
        "milvus_db": _SeedMilvus(),
        "graph_db": _SeedGraph(),
        "embedder": _SeedEmbedder(),
    }
    wf = build_retrieve_wf(agents={"RAG_AGGREGATOR": aggregator}, resources=resources)
    state = {
        "messages": [HumanMessage(content="q")],
        "user_query": "q",
        "worker_results": {},
        "language": "eng",
    }
    context = resolve_retrieval_context(
        Retrieve_param.from_preset("FULL", harness_id=RetrievalHarnessId.AGENTIC_V2)
    )

    out = asyncio.run(wf.ainvoke(state, context=context))

    assert out["worker_results"]["RAG"]["retrieval_attempts"] == 2
    assert out["worker_results"]["RAG"]["validity"] == "valid"


def test_agentic_v3_keeps_seed_and_followup_gold_in_append_only_ledger():
    first_gold = "Bench_MuSiQue/paragraph/d587"
    second_gold = "Bench_MuSiQue/paragraph/b7fdd"
    followup = {
        "tool": "retrieve_more",
        "source": "parallel",
        "args": {"query": "Scottish Parliament official home since 2004", "sources": ["textbook"]},
        "requested_sources": ["textbook"],
        "executed_sources": ["textbook"],
        "invalid_sources": [],
        "source_artifacts": [{
            "tool": "textbook_search",
            "source": "textbook",
            "args": {"query": "Scottish Parliament official home since 2004"},
            "chunks": [
                {"id": second_gold, "uri": second_gold, "text": "Official home is at Holyrood.", "score": 1.0, "source": "textbook"},
                {"id": first_gold, "uri": first_gold, "text": "Matters devolve to the Scottish Parliament.", "score": 0.9, "source": "textbook"},
            ],
            "latency_ms": 2.0,
            "error": "",
        }],
        "chunks": [
            {"id": second_gold, "uri": second_gold, "text": "Official home is at Holyrood.", "score": 1.0,
             "source": "textbook", "rrf_score": 0.2, "source_ranks": {"textbook": 1}, "source_scores": {"textbook": 1.0}},
            {"id": first_gold, "uri": first_gold, "text": "Matters devolve to the Scottish Parliament.", "score": 0.9,
             "source": "textbook", "rrf_score": 0.1, "source_ranks": {"textbook": 2}, "source_scores": {"textbook": 0.9}},
        ],
        "latency_ms": 2.0,
        "error": "",
    }
    aggregator = _FakeLedgerAggregator([followup])
    resources = {
        "milvus_db": _SeedMilvus(),
        "graph_db": type("_Graph", (), {"passage_search": lambda self, *_a, **_k: [
            {"id": first_gold, "uri": first_gold, "text": "Matters devolve to the Scottish Parliament.", "score": 1.0}
        ]})(),
        "embedder": _SeedEmbedder(),
    }
    wf = build_retrieve_wf(
        agents={"RAG_LEDGER_AGGREGATOR": aggregator},
        resources=resources,
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "FULL",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )
    state = {
        "messages": [HumanMessage(content="question")],
        "user_query": "question",
        "worker_results": {},
        "language": "eng",
    }

    out = asyncio.run(wf.ainvoke(state, context=context))
    rag = out["worker_results"]["RAG"]

    assert first_gold in rag["entity_ids"]
    assert second_gold in rag["entity_ids"]
    assert "Holyrood" in rag["content"]
    assert rag["thought"].startswith("Scottish Parliament")
    assert rag["validity"] == "valid"
    assert aggregator.calls[0][0]["current_node"] == "Retrieval_Ledger_Aggregator"


def test_agentic_v3_traces_blocked_parallel_tool_calls():
    aggregator = _FakeLedgerAggregator(blocked_calls=2)
    wf = build_retrieve_wf(
        agents={"RAG_LEDGER_AGGREGATOR": aggregator},
        resources={"milvus_db": _SeedMilvus(), "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V4,
            harness_id=RetrievalHarnessId.AGENTIC_V3,
        )
    )
    tracer = AgentTracer(session_id="blocked-calls")
    chat_id = tracer.begin_chat("question")
    state = {
        "messages": [HumanMessage(content="question")],
        "user_query": "question",
        "worker_results": {},
        "language": "eng",
    }

    asyncio.run(
        wf.ainvoke(
            state,
            config={"configurable": {"tracer": tracer, "chat_id": chat_id}},
            context=context,
        )
    )

    final = next(
        step
        for step in tracer._active_chats[chat_id].agent
        if step.node == "Agentic_Retrieve"
    )
    assert final.tool_result["blocked_tool_calls"] == 2


def test_agentic_v4_runs_typed_grounded_loop_and_preserves_ledger():
    from TA.retrieval.schema import ClaimSupport, FinalChain, HopAction, HopDecision, StopReason

    planner = _StructuredPlanner(
        decisions=[
            HopDecision(
                action=HopAction.RETRIEVE,
                sub_question="bridge fact",
                query="focused bridge",
                sources=[RetrievalToolId.SEMANTIC],
                basis_uris=["semantic-seed"],
            ),
            HopDecision(
                action=HopAction.STOP,
                supported_claims=[
                    ClaimSupport(claim="seed reaches bridge", evidence_uris=["semantic-seed", "semantic-bridge"])
                ],
                stop_reason=StopReason.CHAIN_COMPLETE,
            ),
        ],
        final_chain=FinalChain(
            claims=[
                ClaimSupport(claim="seed reaches bridge", evidence_uris=["semantic-seed", "semantic-bridge"])
            ],
            answerable=True,
        ),
    )
    milvus = _V4Milvus()
    wf = build_retrieve_wf(
        agents={"RETRIEVAL_PLANNER": planner},
        resources={"milvus_db": milvus, "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )
    tracer = AgentTracer(session_id="agentic-v4")
    chat_id = tracer.begin_chat("question")
    state = {
        "messages": [HumanMessage(content="question")],
        "user_query": "question",
        "worker_results": {},
        "language": "eng",
    }

    out = asyncio.run(
        wf.ainvoke(
            state,
            config={"configurable": {"tracer": tracer, "chat_id": chat_id}},
            context=context,
        )
    )

    rag = out["worker_results"]["RAG"]
    assert milvus.queries == ["question", "focused bridge"]
    assert rag["entity_ids"] == ["semantic-seed", "semantic-bridge"]
    assert "seed evidence" in rag["content"] and "bridge evidence" in rag["content"]
    assert "seed reaches bridge" in rag["thought"]
    assert rag["validity"] == "valid"
    assert [schema.__name__ for schema, *_ in planner.calls] == [
        "HopDecision",
        "HopDecision",
        "FinalChain",
    ]
    assert planner.calls[0][1][0][1] != planner.calls[-1][1][0][1]
    nodes = [step.node for step in tracer._active_chats[chat_id].agent]
    assert "Seed_Retrieve" in nodes
    assert "Plan_Hop" in nodes
    assert "Retrieve_Hop" in nodes
    assert "Finalize_Chain" in nodes


def test_agentic_v4_envelope_compiles_cited_answer_context_before_ledger_tail():
    from core.schema.retrieval import RetrievalValidation, RetrievalValidity
    from TA.retrieval.controller import _envelope
    from TA.retrieval.schema import ClaimSupport, FinalChain

    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )
    chain = FinalChain(
        claims=[ClaimSupport(claim="bridge", evidence_uris=["bridge"])],
        answerable=True,
    )
    ledger = [
        {"uri": "seed", "text": "seed evidence"},
        {"uri": "bridge", "text": "bridge evidence"},
        {"uri": "tail", "text": "tail evidence"},
    ]

    result = _envelope(
        ledger,
        chain,
        RetrievalValidation(RetrievalValidity.VALID, 1, ()),
        context,
        {},
    )

    assert result["answer_context"]["uris"] == ["bridge", "seed", "tail"]
    assert result["content"].splitlines()[0] == "- [bridge] bridge evidence"
    assert result["content"] != ""


def test_agentic_v4_repairs_contextually_invalid_source_before_retrieval():
    from TA.retrieval.schema import FinalChain, HopAction, HopDecision, StopReason

    planner = _StructuredPlanner(
        decisions=[
            HopDecision(
                action=HopAction.RETRIEVE,
                sub_question="bad source",
                query="wrong source",
                sources=[RetrievalToolId.TEXTBOOK],
                basis_uris=["semantic-seed"],
            ),
            HopDecision(action=HopAction.STOP, stop_reason=StopReason.INSUFFICIENT_EVIDENCE),
        ],
        final_chain=FinalChain(answerable=False, remaining_uncertainty="missing"),
    )
    milvus = _V4Milvus()
    wf = build_retrieve_wf(
        agents={"RETRIEVAL_PLANNER": planner},
        resources={"milvus_db": milvus, "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )
    state = {
        "messages": [HumanMessage(content="question")],
        "user_query": "question",
        "worker_results": {},
        "language": "eng",
    }

    out = asyncio.run(wf.ainvoke(state, context=context))

    assert milvus.queries == ["question"]
    assert out["worker_results"]["RAG"]["validity"] == "valid"
    assert out["worker_results"]["RAG"]["schema_repairs"] == 1


def test_agentic_v4_duplicate_query_consumes_round_without_repository_call():
    from TA.retrieval.schema import FinalChain, HopAction, HopDecision, StopReason

    planner = _StructuredPlanner(
        decisions=[
            HopDecision(
                action=HopAction.RETRIEVE,
                sub_question="bridge",
                query="Focused Bridge",
                sources=[RetrievalToolId.SEMANTIC],
                basis_uris=["semantic-seed"],
            ),
            HopDecision(
                action=HopAction.RETRIEVE,
                sub_question="same bridge",
                query="  focused   bridge ",
                sources=[RetrievalToolId.SEMANTIC],
                basis_uris=["semantic-seed"],
            ),
            HopDecision(action=HopAction.STOP, stop_reason=StopReason.INSUFFICIENT_EVIDENCE),
        ],
        final_chain=FinalChain(answerable=False, remaining_uncertainty="missing"),
    )
    milvus = _V4Milvus()
    wf = build_retrieve_wf(
        agents={"RETRIEVAL_PLANNER": planner},
        resources={"milvus_db": milvus, "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )
    tracer = AgentTracer(session_id="agentic-v4-duplicate")
    chat_id = tracer.begin_chat("question")

    out = asyncio.run(wf.ainvoke(
        {
            "messages": [HumanMessage(content="question")],
            "user_query": "question",
            "worker_results": {},
            "language": "eng",
        },
        config={"configurable": {"tracer": tracer, "chat_id": chat_id}},
        context=context,
    ))

    assert milvus.queries == ["question", "Focused Bridge"]
    assert out["worker_results"]["RAG"]["stop_reason"] == "insufficient_evidence"
    repeated = [
        step for step in tracer._active_chats[chat_id].agent
        if step.node == "Retrieve_Hop" and step.tool_result.get("repeated_query")
    ]
    assert len(repeated) == 1
    assert repeated[0].tool_result["round"] == 2


def test_agentic_v4_four_followups_finalize_with_budget_reason():
    from TA.retrieval.schema import FinalChain, HopAction, HopDecision

    decisions = [
        HopDecision(
            action=HopAction.RETRIEVE,
            sub_question=f"link {index}",
            query=f"followup {index}",
            sources=[RetrievalToolId.SEMANTIC],
            basis_uris=["semantic-seed"],
        )
        for index in range(4)
    ]
    planner = _StructuredPlanner(
        decisions=decisions,
        final_chain=FinalChain(answerable=False, remaining_uncertainty="budget ended"),
    )
    milvus = _V4Milvus()
    wf = build_retrieve_wf(
        agents={"RETRIEVAL_PLANNER": planner},
        resources={"milvus_db": milvus, "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )

    out = asyncio.run(wf.ainvoke(
        {
            "messages": [HumanMessage(content="question")],
            "user_query": "question",
            "worker_results": {},
            "language": "eng",
        },
        context=context,
    ))

    rag = out["worker_results"]["RAG"]
    assert milvus.queries == ["question", "followup 0", "followup 1", "followup 2", "followup 3"]
    assert rag["stop_reason"] == "budget_exhausted"
    assert rag["validity"] == "valid"


def test_agentic_v4_rejects_second_ungrounded_decision_without_retrieval():
    from TA.retrieval.schema import FinalChain, HopAction, HopDecision

    bad = HopDecision(
        action=HopAction.RETRIEVE,
        sub_question="invented basis",
        query="follow invented entity",
        sources=[RetrievalToolId.SEMANTIC],
        basis_uris=["not-in-ledger"],
    )
    planner = _StructuredPlanner(
        decisions=[bad, bad],
        final_chain=FinalChain(answerable=False, remaining_uncertainty="invalid decision"),
    )
    milvus = _V4Milvus()
    wf = build_retrieve_wf(
        agents={"RETRIEVAL_PLANNER": planner},
        resources={"milvus_db": milvus, "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )

    out = asyncio.run(wf.ainvoke(
        {
            "messages": [HumanMessage(content="question")],
            "user_query": "question",
            "worker_results": {},
            "language": "eng",
        },
        context=context,
    ))

    rag = out["worker_results"]["RAG"]
    assert milvus.queries == ["question"]
    assert rag["validity"] == "invalid"
    assert rag["schema_repairs"] == 1
    assert any("basis URIs not in ledger" in error for error in rag["errors"])


def test_agentic_v4_never_retries_or_finalizes_after_provider_429():
    class _QuotaPlanner:
        model = "nvidia/nemotron-3-super-120b-a12b:free"
        temperature = 0.0

        def __init__(self):
            self.calls = 0

        def with_structured_output(self, *_args, **_kwargs):
            planner = self

            class _Invocation:
                async def ainvoke(self, *_args, **_kwargs):
                    planner.calls += 1
                    error = RuntimeError("Ollama status code: 429 session usage limit reached")
                    error.status_code = 429
                    raise error

            return _Invocation()

    planner = _QuotaPlanner()
    wf = build_retrieve_wf(
        agents={"RETRIEVAL_PLANNER": planner},
        resources={"milvus_db": _V4Milvus(), "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )

    out = asyncio.run(wf.ainvoke(
        {
            "messages": [HumanMessage(content="question")],
            "user_query": "question",
            "worker_results": {},
            "language": "eng",
        },
        context=context,
    ))

    assert planner.calls == 1
    assert out["worker_results"]["RAG"]["validity"] == "invalid"
    assert any("429" in error for error in out["worker_results"]["RAG"]["errors"])


def test_agentic_v4_propagates_hard_provider_budget_before_model_call():
    from TA.helper.model_call import provider_request_gate
    from TA.retrieval.schema import FinalChain, HopAction, HopDecision, StopReason
    from test.eval.provider_budget import ProviderBudgetExceeded

    planner = _StructuredPlanner(
        decisions=[HopDecision(action=HopAction.STOP, stop_reason=StopReason.CHAIN_COMPLETE)],
        final_chain=FinalChain(answerable=False, remaining_uncertainty="unused"),
    )
    wf = build_retrieve_wf(
        agents={"RETRIEVAL_PLANNER": planner},
        resources={"milvus_db": _V4Milvus(), "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )

    def exhausted(_profile):
        raise ProviderBudgetExceeded("provider request budget exhausted")

    with provider_request_gate(exhausted):
        with pytest.raises(ProviderBudgetExceeded):
            asyncio.run(wf.ainvoke(
                {
                    "messages": [HumanMessage(content="question")],
                    "user_query": "question",
                    "worker_results": {},
                    "language": "eng",
                },
                context=context,
            ))

    assert planner.calls == []


def test_agentic_v4_three_hop_chain_keeps_scottish_evidence_for_answerer():
    from TA.retrieval.schema import ClaimSupport, FinalChain, HopAction, HopDecision, StopReason
    from TA.workflow.smart_edu import SmartEdu

    evidence = {
        "question": ("scottish", "Matters devolve to the Scottish Parliament."),
        "holyrood": ("holyrood", "The Scottish Parliament meets at Holyrood."),
        "architect": ("architect", "Enric Miralles designed the Holyrood building."),
        "birthplace": ("birthplace", "Enric Miralles was born in Barcelona."),
    }

    class _ChainMilvus:
        def __init__(self):
            self.queries = []

        def search(self, query, **_kwargs):
            self.queries.append(query)
            uri, text = evidence[query]
            return [{"id": uri, "uri": uri, "text": text, "score": 1.0}]

    claims = [ClaimSupport(
        claim="The chain reaches Barcelona through Holyrood and its architect.",
        evidence_uris=["scottish", "holyrood", "architect", "birthplace"],
    )]
    planner = _StructuredPlanner(
        decisions=[
            HopDecision(action=HopAction.RETRIEVE, sub_question="meeting place", query="holyrood", sources=[RetrievalToolId.SEMANTIC], basis_uris=["scottish"]),
            HopDecision(action=HopAction.RETRIEVE, sub_question="architect", query="architect", sources=[RetrievalToolId.SEMANTIC], basis_uris=["holyrood"]),
            HopDecision(action=HopAction.RETRIEVE, sub_question="birthplace", query="birthplace", sources=[RetrievalToolId.SEMANTIC], basis_uris=["architect"]),
            HopDecision(action=HopAction.STOP, supported_claims=claims, stop_reason=StopReason.CHAIN_COMPLETE),
        ],
        final_chain=FinalChain(claims=claims, answerable=True),
    )
    milvus = _ChainMilvus()
    wf = build_retrieve_wf(
        agents={"RETRIEVAL_PLANNER": planner},
        resources={"milvus_db": milvus, "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )

    out = asyncio.run(wf.ainvoke(
        {
            "messages": [HumanMessage(content="question")],
            "user_query": "question",
            "worker_results": {},
            "language": "eng",
        },
        context=context,
    ))

    rag = out["worker_results"]["RAG"]
    answer_prompt = SmartEdu._benchmark_answer_prompt("question", out["worker_results"])
    assert milvus.queries == ["question", "holyrood", "architect", "birthplace"]
    assert rag["entity_ids"] == ["scottish", "holyrood", "architect", "birthplace"]
    assert "Scottish Parliament" in answer_prompt
    assert "Barcelona" in answer_prompt
    assert rag["stop_reason"] == "chain_complete"


def test_agentic_v4_context_guard_finalizes_before_next_repository_call():
    from dataclasses import replace
    from TA.retrieval.schema import FinalChain, HopAction, HopDecision

    planner = _StructuredPlanner(
        decisions=[HopDecision(
            action=HopAction.RETRIEVE,
            sub_question="too much evidence",
            query="would overflow",
            sources=[RetrievalToolId.SEMANTIC],
            basis_uris=["semantic-seed"],
        )],
        final_chain=FinalChain(answerable=False, remaining_uncertainty="context limit"),
    )
    milvus = _V4Milvus()
    wf = build_retrieve_wf(
        agents={"RETRIEVAL_PLANNER": planner},
        resources={"milvus_db": milvus, "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )
    context = replace(context, harness=replace(context.harness, max_context_chars=1))

    out = asyncio.run(wf.ainvoke(
        {
            "messages": [HumanMessage(content="question")],
            "user_query": "question",
            "worker_results": {},
            "language": "eng",
        },
        context=context,
    ))

    assert milvus.queries == ["question"]
    assert out["worker_results"]["RAG"]["stop_reason"] == "context_limit"


def test_agentic_v4_repairs_malformed_schema_once():
    from TA.retrieval.schema import FinalChain, HopAction, HopDecision, StopReason

    planner = _StructuredPlanner(
        decisions=[
            {"action": "retrieve", "query": "missing required fields"},
            HopDecision(action=HopAction.STOP, stop_reason=StopReason.INSUFFICIENT_EVIDENCE),
        ],
        final_chain=FinalChain(answerable=False, remaining_uncertainty="missing"),
    )
    wf = build_retrieve_wf(
        agents={"RETRIEVAL_PLANNER": planner},
        resources={"milvus_db": _V4Milvus(), "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )

    out = asyncio.run(wf.ainvoke(
        {
            "messages": [HumanMessage(content="question")],
            "user_query": "question",
            "worker_results": {},
            "language": "eng",
        },
        context=context,
    ))

    rag = out["worker_results"]["RAG"]
    assert rag["validity"] == "valid"
    assert rag["schema_repairs"] == 1
    assert [schema.__name__ for schema, *_ in planner.calls] == [
        "HopDecision", "HopDecision", "FinalChain"
    ]


def _run_v4(planner, milvus=None):
    milvus = milvus or _V4Milvus()
    wf = build_retrieve_wf(
        agents={"RETRIEVAL_PLANNER": planner},
        resources={"milvus_db": milvus, "embedder": _SeedEmbedder()},
    )
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(
            "RAG",
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
        )
    )
    out = asyncio.run(wf.ainvoke(
        {
            "messages": [HumanMessage(content="question")],
            "user_query": "question",
            "worker_results": {},
            "language": "eng",
        },
        context=context,
    ))
    return out, milvus


def test_agentic_v4_missing_sources_triggers_single_repair():
    from TA.retrieval.schema import FinalChain, HopAction, HopDecision, StopReason

    planner = _StructuredPlanner(
        decisions=[
            {"action": "retrieve", "query": "q", "sub_question": "gap", "basis_uris": ["semantic-seed"]},
            HopDecision(action=HopAction.STOP, stop_reason=StopReason.INSUFFICIENT_EVIDENCE),
        ],
        final_chain=FinalChain(answerable=False, remaining_uncertainty="missing"),
    )

    out, _ = _run_v4(planner)

    rag = out["worker_results"]["RAG"]
    assert rag["schema_repairs"] == 1
    assert [schema.__name__ for schema, *_ in planner.calls] == [
        "HopDecision", "HopDecision", "FinalChain"
    ]


def test_agentic_v4_missing_basis_uris_triggers_single_repair():
    from TA.retrieval.schema import FinalChain, HopAction, HopDecision, StopReason

    planner = _StructuredPlanner(
        decisions=[
            {"action": "retrieve", "query": "q", "sub_question": "gap", "sources": ["semantic"]},
            HopDecision(action=HopAction.STOP, stop_reason=StopReason.INSUFFICIENT_EVIDENCE),
        ],
        final_chain=FinalChain(answerable=False, remaining_uncertainty="missing"),
    )

    out, _ = _run_v4(planner)

    rag = out["worker_results"]["RAG"]
    assert rag["schema_repairs"] == 1
    assert [schema.__name__ for schema, *_ in planner.calls] == [
        "HopDecision", "HopDecision", "FinalChain"
    ]


def test_agentic_v4_second_malformed_decision_stays_invalid_with_concrete_error():
    from TA.retrieval.schema import FinalChain

    bad = {"action": "retrieve", "query": "still missing fields", "sub_question": "gap"}
    planner = _StructuredPlanner(
        decisions=[bad, bad],
        final_chain=FinalChain(answerable=False, remaining_uncertainty="unreachable"),
    )

    out, _ = _run_v4(planner)

    rag = out["worker_results"]["RAG"]
    assert rag["validity"] == "invalid"
    assert rag["schema_repairs"] == 1
    # no third planner call after the single permitted repair fails again
    assert [schema.__name__ for schema, *_ in planner.calls] == ["HopDecision", "HopDecision"]
    # the concrete cause reaches _error instead of a generic placeholder
    assert out["status_flag"] == "FAIL"
    assert "requires sources and basis URIs" in out["_error"]


def test_agentic_v4_never_invents_basis_uris_when_model_omits_them():
    from TA.retrieval.schema import FinalChain

    missing_basis = {
        "action": "retrieve",
        "query": "follow up",
        "sub_question": "gap",
        "sources": ["semantic"],
    }
    planner = _StructuredPlanner(
        decisions=[missing_basis, missing_basis],
        final_chain=FinalChain(answerable=False, remaining_uncertainty="unreachable"),
    )
    milvus = _V4Milvus()

    out, milvus = _run_v4(planner, milvus)

    rag = out["worker_results"]["RAG"]
    # old behavior silently grounded basis_uris in the last ledger entry and
    # retried the retrieval; the fixed controller must never do that
    assert milvus.queries == ["question"]
    assert rag["validity"] == "invalid"
    assert any("sources and basis URIs" in error for error in rag["errors"])
