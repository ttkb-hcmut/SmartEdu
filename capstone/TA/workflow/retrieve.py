import os
import time
import logging
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langchain_core.messages import ToolMessage
from typing import Dict, Any, List, Optional

from core.schema.retrieval import (
    RetrievalHarnessId,
    RetrievalRunContext,
    RetrievalToolId,
    RetrievalValidation,
    RetrievalValidity,
)
from core.schema.wf_state import AgentState, ConceptNode
import TA.helper.prompt as prompt_lib
from TA.helper.schema import RAGCore, RAGDeep, DeepDecision
from TA.helper.few_shot import get_language_instruction
from TA.helper.utils import parse_student_state, safe_parse_structured, extract_llm_raw_text, extract_agent_result


from TA.tracing.tracer import AgentTracer
from TA.retrieval.policy import get_tool_spec, validate_retrieval_artifacts

logger = logging.getLogger(__name__)


def _tracer_ctx(config):
    c = config.get("configurable", {})
    return c.get("tracer"), c.get("chat_id", "")


def _norm_chunk(r: Dict, source: str) -> Dict:
    return {
        "id": r.get("id"),
        "uri": r.get("uri") or r.get("id"),
        "text": r.get("text", ""),
        "score": float(r.get("score", 0.0) or 0.0),
        "source": source,
    }


def rrf_merge(pools: Dict[str, List[Dict]], rrf_k: int, top_k: int) -> List[Dict]:
    scores: Dict[str, float] = {}
    seen: Dict[str, Dict] = {}
    for chunks in pools.values():
        for rank, c in enumerate(chunks):
            key = str(c["uri"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            seen.setdefault(key, c)
    merged = sorted(seen.values(), key=lambda c: scores[str(c["uri"])], reverse=True)[:top_k]
    return [{**c, "score": scores[str(c["uri"])]} for c in merged]


def _rag_envelope(
    *,
    thought: str,
    entity_ids: List[str],
    content: str,
    validation: RetrievalValidation,
    agent_status: str = "",
) -> Dict[str, Any]:
    return {
        "thought": thought,
        "entity_ids": entity_ids,
        "content": content,
        "status": "SUCCESS" if validation.validity is RetrievalValidity.VALID else "FAIL",
        "validity": validation.validity.value,
        "retrieval_attempts": validation.attempted_calls,
        "errors": list(validation.errors),
        "agent_status": agent_status,
    }


def _retrieval_artifacts(messages, context: RetrievalRunContext) -> List[Dict[str, Any]]:
    names = {get_tool_spec(tool_id).name for tool_id in context.policy.allowed_tools}
    artifacts = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name not in names:
            continue
        if isinstance(message.artifact, dict):
            artifacts.append(message.artifact)
        else:
            artifacts.append({
                "tool": message.name,
                "source": "",
                "args": {},
                "chunks": [],
                "error": "missing retrieval artifact",
            })
    return artifacts


def _model_contract_error(rag_agent, context: RetrievalRunContext) -> str:
    model = getattr(rag_agent, "model", None)
    name = getattr(model, "model", None) or getattr(model, "model_name", None)
    temperature = getattr(model, "temperature", None)
    if name != context.policy.model_name:
        return f"model mismatch: expected {context.policy.model_name}, got {name or 'unknown'}"
    if temperature is None or float(temperature) != context.policy.temperature:
        return (
            f"temperature mismatch: expected {context.policy.temperature}, "
            f"got {temperature if temperature is not None else 'unknown'}"
        )
    return ""


def build_agentic_retrieve_wf(agents):
    builder = StateGraph(AgentState, context_schema=RetrievalRunContext)
    rag_agent = agents.get("RAG")

    async def run_agentic(state, config, runtime: Runtime[RetrievalRunContext]):
        context = runtime.context
        query = state.get("user_query", state["messages"][-1].content)
        prompt = f"Question: {query}"
        artifacts: List[Dict[str, Any]] = []
        execution_error = ""
        structured = RAGCore(thought="", content="", status="FAIL")
        result = {}
        if rag_agent is None:
            execution_error = "RAG agent unavailable"
        elif model_error := _model_contract_error(rag_agent, context):
            execution_error = model_error
        else:
            try:
                result = await rag_agent.ainvoke(
                    {"messages": [("user", prompt)], "current_node": "RAG_Core"},
                    config={**config, "recursion_limit": context.harness.recursion_limit},
                    context=context,
                )
                artifacts = _retrieval_artifacts(result.get("messages", []), context)
                structured = extract_agent_result(result, RAGCore, "rag_core")
            except Exception as exc:
                execution_error = f"{type(exc).__name__}: {exc}"
                logger.warning("[agentic_retrieve] failed: %s", execution_error)

        validation = validate_retrieval_artifacts(context, artifacts)
        if execution_error:
            validation = RetrievalValidation(
                RetrievalValidity.INVALID,
                validation.attempted_calls,
                (*validation.errors, execution_error),
            )
        rag_result = _rag_envelope(
            thought=structured.thought,
            entity_ids=structured.entity_ids,
            content=structured.content,
            validation=validation,
            agent_status=structured.status,
        )

        tracer, chat_id = _tracer_ctx(config)
        if tracer and chat_id:
            chunks = [chunk for artifact in artifacts for chunk in artifact.get("chunks", [])]
            for artifact in artifacts:
                tracer.log_step(
                    chat_id=chat_id,
                    node=f"Comp_{artifact.get('source', 'Unknown').title()}",
                    tool_result={
                        "tool": artifact.get("tool", ""),
                        "args": artifact.get("args", {}),
                        "error": artifact.get("error", ""),
                    },
                    chunks=artifact.get("chunks", []),
                    latency_ms=artifact.get("latency_ms", 0.0),
                )
            tracer.log_step(
                chat_id=chat_id,
                node="Agentic_Retrieve",
                tool_result={
                    "status": rag_result["status"],
                    "validity": rag_result["validity"],
                    "attempts": rag_result["retrieval_attempts"],
                    "errors": rag_result["errors"],
                },
                chunks=chunks,
            )

        current = state.get("worker_results", {})
        return {
            "worker_results": {**current, "RAG": rag_result},
            "status_flag": rag_result["status"],
        }

    builder.add_node("Agentic_Retrieve", run_agentic)
    builder.set_entry_point("Agentic_Retrieve")
    builder.add_edge("Agentic_Retrieve", END)
    return builder.compile()


def build_fanout_retrieve_wf(resources: Optional[Dict] = None):
    """Fan-out retrieval: dispatch -> enabled components (parallel) -> fusion."""
    builder = StateGraph(AgentState, context_schema=RetrievalRunContext)
    resources = resources or {}

    async def dispatch(state, config, runtime: Runtime[RetrievalRunContext]):
        allow = set(runtime.context.policy.allowed_tools)
        flags = {
            "semantic": RetrievalToolId.SEMANTIC in allow,
            "textbook": RetrievalToolId.TEXTBOOK in allow,
        }
        tracer, chat_id = _tracer_ctx(config)
        if tracer and chat_id:
            tracer.log_step(chat_id=chat_id, node="Retrieve_Dispatch",
                            tool_result={"flags": flags, "preset": runtime.context.preset.value})
        return {"_retrieve_flags": flags}

    async def comp_semantic(state, config, runtime: Runtime[RetrievalRunContext]):
        tracer, chat_id = _tracer_ctx(config)
        start = time.time()
        chunks: List[Dict] = []
        error = ""
        query = state.get("user_query", state["messages"][-1].content)
        milvus, embedder = resources.get("milvus_db"), resources.get("embedder")
        if milvus and embedder:
            try:
                scope = runtime.context.scope.course
                hits = milvus.search(
                    query=query,
                    embedder=embedder,
                    top_k=runtime.context.harness.per_source_k,
                    course_scope=scope,
                )
                chunks = [_norm_chunk(h, "semantic") for h in hits]
            except Exception as e:
                logger.warning(f"[comp_semantic] search failed: {e}")
                error = str(e)
        else:
            logger.warning("[comp_semantic] milvus/embedder resources missing, component no-op")
            error = "resources missing"
        if tracer and chat_id:
            ## error in trace -> eval can tell crash from empty
            tracer.log_step(chat_id=chat_id, node="Comp_Semantic",
                            tool_result={"error": error} if error else {},
                            chunks=chunks, latency_ms=(time.time() - start) * 1000)
        artifact = {
            "tool": get_tool_spec(RetrievalToolId.SEMANTIC).name,
            "source": "semantic",
            "args": {"query": query},
            "chunks": chunks,
            "latency_ms": (time.time() - start) * 1000,
            "error": error,
        }
        return {
            "retrieval_pool": {"semantic": chunks},
            "retrieval_artifacts": {"semantic": artifact},
        }

    async def comp_textbook(state, config, runtime: Runtime[RetrievalRunContext]):
        tracer, chat_id = _tracer_ctx(config)
        start = time.time()
        chunks: List[Dict] = []
        error = ""
        query = state.get("user_query", state["messages"][-1].content)
        graph_db, embedder = resources.get("graph_db"), resources.get("embedder")
        if graph_db and embedder:
            try:
                emb = embedder.get_embedding(query)
                scope = runtime.context.scope.course
                prefix = f"{scope}/" if scope else None
                hits = graph_db.passage_search(emb, query_text=query,
                                               top_k=runtime.context.harness.per_source_k,
                                               uri_prefix=prefix)
                chunks = [_norm_chunk(h, "textbook") for h in hits]
            except Exception as e:
                logger.warning(f"[comp_textbook] search failed: {e}")
                error = str(e)
        else:
            logger.warning("[comp_textbook] graph_db/embedder resources missing, component no-op")
            error = "resources missing"
        if tracer and chat_id:
            tracer.log_step(chat_id=chat_id, node="Comp_Textbook",
                            tool_result={"error": error} if error else {},
                            chunks=chunks, latency_ms=(time.time() - start) * 1000)
        artifact = {
            "tool": get_tool_spec(RetrievalToolId.TEXTBOOK).name,
            "source": "textbook",
            "args": {"query": query},
            "chunks": chunks,
            "latency_ms": (time.time() - start) * 1000,
            "error": error,
        }
        return {
            "retrieval_pool": {"textbook": chunks},
            "retrieval_artifacts": {"textbook": artifact},
        }

    async def fusion(state, config, runtime: Runtime[RetrievalRunContext]):
        tracer, chat_id = _tracer_ctx(config)
        start = time.time()
        pools = state.get("retrieval_pool") or {}
        merged = rrf_merge(
            pools,
            runtime.context.harness.rrf_k,
            runtime.context.harness.top_k,
        )
        content = "\n".join(f"- [{c['source']}] {c['text']}" for c in merged)

        by_source = state.get("retrieval_artifacts") or {}
        artifacts = [by_source[name] for name in ("semantic", "textbook") if name in by_source]
        validation = validate_retrieval_artifacts(runtime.context, artifacts)
        rag_result = _rag_envelope(
            thought=f"fusion over {list(pools.keys())} -> {len(merged)} chunks",
            entity_ids=[c["id"] for c in merged if c.get("id")],
            content=content,
            validation=validation,
        )
        status = rag_result["status"]

        session_context = config.get("configurable", {}).get("session_context")
        if session_context and chat_id and content:
            session_context.store_tool_result(
                chat_id=chat_id, tool_name="retrieval_fusion",
                args={"query": state.get("user_query", "")}, output=content, node="Fusion",
            )

        if tracer and chat_id:
            tracer.log_step(chat_id=chat_id, node="Fusion",
                            tool_result={"sources": state.get("_retrieve_flags", {}), "status": status},
                            chunks=merged, latency_ms=(time.time() - start) * 1000)

        current = state.get("worker_results", {})
        return {"worker_results": {**current, "RAG": rag_result}, "status_flag": status}

    def route_components(state):
        flags = state.get("_retrieve_flags") or {}
        targets = []
        if flags.get("semantic"):
            targets.append("Comp_Semantic")
        if flags.get("textbook"):
            targets.append("Comp_Textbook")
        return targets or ["Fusion"]

    builder.add_node("Retrieve_Dispatch", dispatch)
    builder.add_node("Comp_Semantic", comp_semantic)
    builder.add_node("Comp_Textbook", comp_textbook)
    builder.add_node("Fusion", fusion)

    builder.set_entry_point("Retrieve_Dispatch")
    builder.add_conditional_edges(
        "Retrieve_Dispatch",
        route_components,
        {"Comp_Semantic": "Comp_Semantic", "Comp_Textbook": "Comp_Textbook", "Fusion": "Fusion"},
    )
    builder.add_edge("Comp_Semantic", "Fusion")
    builder.add_edge("Comp_Textbook", "Fusion")
    builder.add_edge("Fusion", END)

    return builder.compile()


def build_retrieve_wf(agents, resources: Optional[Dict] = None):
    builder = StateGraph(AgentState, context_schema=RetrievalRunContext)
    agentic = build_agentic_retrieve_wf(agents)
    fanout = build_fanout_retrieve_wf(resources)

    async def select_harness(state, runtime: Runtime[RetrievalRunContext]):
        return {"_retrieve_harness": runtime.context.harness.id.value}

    def route_harness(state):
        return state["_retrieve_harness"]

    builder.add_node("Retrieve_Harness", select_harness)
    builder.add_node(RetrievalHarnessId.AGENTIC_V1.value, agentic)
    builder.add_node(RetrievalHarnessId.FANOUT_V1.value, fanout)
    builder.set_entry_point("Retrieve_Harness")
    builder.add_conditional_edges(
        "Retrieve_Harness",
        route_harness,
        {
            RetrievalHarnessId.AGENTIC_V1.value: RetrievalHarnessId.AGENTIC_V1.value,
            RetrievalHarnessId.FANOUT_V1.value: RetrievalHarnessId.FANOUT_V1.value,
        },
    )
    builder.add_edge(RetrievalHarnessId.AGENTIC_V1.value, END)
    builder.add_edge(RetrievalHarnessId.FANOUT_V1.value, END)
    return builder.compile()


## unwired legacy chain below (deep_decision/rag_deep), kept for rollback; bridge proposals dormant
async def deep_decision(state: AgentState, ta_agent, config):
    """ Classify DEEP vs SKIP, no tools, direct structured output"""
    rag_data = state.get("worker_results", {}).get("RAG", {})

    if state.get("status_flag") == "FAIL" or not rag_data.get("entity_ids"):
        return {"_deep_route": "skip"}

    query = state.get("user_query", state["messages"][-1].content)
    student_state = state.get("student_state", {})

    student_state = parse_student_state(student_state)

    language = state.get("language", "vn")
    language_instruction = get_language_instruction(language)

    prompt = prompt_lib.DEEP_CHECK_PROMPT.format(
        language_instruction=language_instruction,
        query=query,
        rag_summary=str(rag_data.get("content", ""))[:300],
        current_pos=student_state
    )

    ## -- Direct structured output, no tool loop needed
    structured_llm = ta_agent.model.with_structured_output(DeepDecision)
    try:
        decision: DeepDecision = await structured_llm.ainvoke(
            [("user", prompt)], config=config
        )
        route = decision.decision.lower()
    except Exception as e:
        logger.warning(f"[deep_decision] Structured output failed: {e}. Attempting json_repair.")
        raw_text = extract_llm_raw_text(e)
        try:
            decision = safe_parse_structured(raw_text, DeepDecision)
            route = decision.decision.lower()
        except Exception:
            # Last resort: keyword scan
            upper = raw_text.upper()
            route = "deep" if "DEEP" in upper else "skip"
            logger.info(f"[deep_decision] Keyword fallback → route='{route}'")

    log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
    if log_filename:
        AgentTracer.logging({
            "agent_name": ta_agent.name,
            "node": "Deep_Decision",
            "thought": decision.thought if 'decision' in locals() and hasattr(decision, "thought") else "",
            "prompt": prompt[:300],
            "output": decision.model_dump() if 'decision' in locals() and hasattr(decision, "model_dump") else {"decision": route}
        }, type="info", file_name=log_filename)

    return {"_deep_route": route}



async def rag_core(state: AgentState, rag_agent, config):
    """ Invoke global RAG agent with current_node=RAG_Core for schema routing"""
    import os
    import traceback
    
    start = time.time()
    log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
    
    try:
        query = state["messages"][-1].content
        language = state.get("language", "vn")
        language_instruction = get_language_instruction(language)
        
        core_prompt = prompt_lib.RESEARCH_STRATEGY_PROMPT['core'].format(
            language_instruction=language_instruction
        )
        prompt = f"{core_prompt}\nQuery: {query}"
        
        try:
            result = await rag_agent.ainvoke(
                {"messages": [("user", prompt)], "current_node": "RAG_Core"},
                config={"recursion_limit": 40, **config},
            )
        except Exception as e:
            logger.warning(f"[rag_core] Structured output failed: {e}. Attempting json_repair.")
            structured = safe_parse_structured(extract_llm_raw_text(e), RAGCore)
        else:
            structured = extract_agent_result(result, RAGCore, "rag_core")

        
        report_lines = [
            f"========================= Agent \"{rag_agent.name}\" Runtime Report ==========================",
            f"[SMART_EDU_LOG] Node: rag_core | Action: global_agent.ainvoke | Time: {time.time() - start:.4f}s",
            f"Status: {structured.status}",
            f"Reasoning: {structured.thought}",
            f"Entity IDs: {structured.entity_ids}",
            f"Content length: {len(structured.content) if structured.content else 0}",
            f"*************** END Report ***************"
        ]
        report_str = "\n" + "\n".join(report_lines) + "\n"
        logger.info(report_str)
        
        if log_filename:
            log_dict = {
                "agent_name": rag_agent.name,
                "node": "rag_core",
                "thought": structured.thought,
                "prompt": prompt[:300],
                "output": structured.model_dump()
            }
            AgentTracer.logging(log_dict, type="info", file_name=log_filename)

        current_worker_results = state.get("worker_results", {})
        result_state = {
            "worker_results": {**current_worker_results, "RAG": structured.model_dump()},
            "status_flag": structured.status,
        }

        # Store in session context (in-memory, for within-request recall by TA tools)
        session_context = config.get("configurable", {}).get("session_context")
        chat_id = config.get("configurable", {}).get("chat_id", "")
        if session_context and chat_id:
            session_context.store_tool_result(
                chat_id=chat_id,
                tool_name="rhetorical_retriever",
                args={"query": query},
                output=structured.content or "",
                node="RAG_Core",
            )

        # Persist to MongoDB atomically (background, concurrent-safe $push)
        student_id = config.get("configurable", {}).get("student_id", "")
        tracker = config.get("configurable", {}).get("student_tracker")
        if tracker and student_id and chat_id:
            import asyncio
            asyncio.create_task(
                asyncio.to_thread(
                    tracker.mongodb.push_chat_message,
                    student_id,
                    config.get("configurable", {}).get("session_id", ""),
                    chat_id,
                    {
                        "role": rag_agent.name,
                        "heading": "RAG Core Search",
                        "message": structured.thought
                    }
                )
            )

        return result_state
    except Exception as e:
        if log_filename:
            AgentTracer.logging(traceback.format_exc(), type="err", file_name=log_filename)
        raise e


async def rag_deep(state: AgentState, rag_agent, config):
    """ Invoke global RAG agent with current_node=RAG_Deep for schema routing"""
    import os
    import traceback
    
    start = time.time()
    log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
    
    try:
        rag_data: Dict[str, Any] = state.get("worker_results", {}).get("RAG", {})
        student_state = state.get("student_state", {})
        current_pos_obj = student_state.get("current_pos")

        query = state.get("user_query", state["messages"][-1].content)
        language = state.get("language", "vn")
        language_instruction = get_language_instruction(language)

        prompt = prompt_lib.RESEARCH_STRATEGY_PROMPT["deep"].format(
            language_instruction=language_instruction,
            query=query,
            current_pos=current_pos_obj.name if current_pos_obj else "None",
            entity_ids=str(rag_data.get("entity_ids", []))
        )
        
        try:
            result = await rag_agent.ainvoke(
                {"messages": [("user", f"{prompt}\nContext: {rag_data.get('content', '')}")], "current_node": "RAG_Deep"},
                config={"recursion_limit": 15, **config},
            )
        except Exception as e:
            logger.warning(f"[rag_deep] Structured output failed: {e}. Attempting json_repair.")
            structured = safe_parse_structured(extract_llm_raw_text(e), RAGDeep)
        else:
            structured = extract_agent_result(result, RAGDeep, "rag_deep")

        
        report_lines = [
            f"========================= Agent \"{rag_agent.name}\" Runtime Report ==========================",
            f"[SMART_EDU_LOG] Node: rag_deep | Action: global_agent.ainvoke | Time: {time.time() - start:.4f}s",
            f"Is Deep: {structured.is_deep}",
            f"Reasoning: {structured.thought}",
            f"Bridge Concepts: {len(structured.bridge_concepts) if structured.bridge_concepts else 0}",
            f"*************** END Report ***************"
        ]
        report_str = "\n" + "\n".join(report_lines) + "\n"
        logger.info(report_str)
        
        if log_filename:
            log_dict = {
                "agent_name": rag_agent.name,
                "node": "rag_deep",
                "thought": structured.thought,
                "prompt": prompt[:300],
                "output": structured.model_dump()
            }
            AgentTracer.logging(log_dict, type="info", file_name=log_filename)

        bridge_nodes = [
            ConceptNode(name=c["name"]) if isinstance(c, dict) else ConceptNode(name=c.name)
            for c in structured.bridge_concepts
            if (isinstance(c, dict) and c.get("name")) or (hasattr(c, "name") and c.name)
        ]

        proposal = None
        if structured.is_deep and bridge_nodes:
            proposal = {
                "type": "bridge",
                "new_current": bridge_nodes[0].model_dump(),
                "new_upcoming": [n.model_dump() for n in bridge_nodes[1:]],
                "reason": f"Knowledge gap: {structured.knowledge_gap_score:.1f}",
                "source_wf": "Retrieve",
                "auto_apply": True,
            }
        elif bridge_nodes and current_pos_obj:
            proposal = {
                "type": "bridge",
                "new_current": current_pos_obj.model_dump(),
                "new_upcoming": [n.model_dump() for n in bridge_nodes],
                "reason": f"Bridge concepts needed",
                "source_wf": "Retrieve",
                "auto_apply": True,
            }

        merged_rag_data = {**rag_data, **structured.model_dump()}
        current_worker_results = state.get("worker_results", {})

        result_dict = {"worker_results": {**current_worker_results, "RAG": merged_rag_data}}
        if proposal:
            result_dict["pending_proposal"] = proposal
        return result_dict

    except Exception as e:
        if log_filename:
            AgentTracer.logging(traceback.format_exc(), type="err", file_name=log_filename)
        raise e
