import os
import time
import logging
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, Optional

from core.config import retrieve_param as _default_rp, Retrieve_param
from core.schema.wf_state import AgentState, ConceptNode
import TA.edu.helper.prompt as prompt_lib
from TA.edu.helper.schema import RAGCore, RAGDeep, DeepDecision
from TA.edu.helper.few_shot import get_language_instruction
from TA.edu.helper.utils import parse_student_state, safe_parse_structured, extract_llm_raw_text, extract_agent_result


from TA.tracing.tracer import AgentTracer

logger = logging.getLogger(__name__)


def _get_rp(config) -> Retrieve_param:
    return config.get("configurable", {}).get("retrieve_param") or _default_rp


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


def build_retrieve_wf(agents, resources: Optional[Dict] = None):
    """Fan-out retrieval: dispatch -> enabled components (parallel) -> fusion."""
    builder = StateGraph(AgentState)
    resources = resources or {}

    async def dispatch(state, config):
        rp = _get_rp(config)
        flags = rp.flag_set()
        tracer, chat_id = _tracer_ctx(config)
        if tracer and chat_id:
            tracer.log_step(chat_id=chat_id, node="Retrieve_Dispatch",
                            tool_result={"flags": flags, "preset": rp.preset})
        return {"_retrieve_flags": flags}

    async def comp_rag(state, config):
        rp = _get_rp(config)
        tracer, chat_id = _tracer_ctx(config)
        start = time.time()
        chunks: List[Dict] = []
        error = ""
        query = state.get("user_query", state["messages"][-1].content)
        milvus, embedder = resources.get("milvus_db"), resources.get("embedder")
        if milvus and embedder:
            try:
                ## assumes ingest writes course into community field
                expr = f'community == "{rp.benchmark_course}"' if rp.benchmark_course else None
                hits = milvus.search(query=query, embedder=embedder, top_k=rp.per_component_k, expr=expr)
                chunks = [_norm_chunk(h, "rag") for h in hits]
            except Exception as e:
                logger.warning(f"[comp_rag] search failed: {e}")
                error = str(e)
        else:
            logger.warning("[comp_rag] milvus/embedder resources missing, component no-op")
            error = "resources missing"
        if tracer and chat_id:
            ## error in trace -> eval can tell crash from empty
            tracer.log_step(chat_id=chat_id, node="Comp_RAG",
                            tool_result={"error": error} if error else {},
                            chunks=chunks, latency_ms=(time.time() - start) * 1000)
        return {"retrieval_pool": {"rag": chunks}}

    async def comp_graphrag(state, config):
        rp = _get_rp(config)
        tracer, chat_id = _tracer_ctx(config)
        start = time.time()
        chunks: List[Dict] = []
        error = ""
        query = state.get("user_query", state["messages"][-1].content)
        graph_db, embedder = resources.get("graph_db"), resources.get("embedder")
        if graph_db and embedder:
            try:
                emb = embedder.get_embedding(query)
                prefix = f"{rp.benchmark_course}/" if rp.benchmark_course else None
                hits = graph_db.passage_search(emb, query_text=query,
                                               top_k=rp.per_component_k, uri_prefix=prefix)
                chunks = [_norm_chunk(h, "graphrag") for h in hits]
            except Exception as e:
                logger.warning(f"[comp_graphrag] search failed: {e}")
                error = str(e)
        else:
            logger.warning("[comp_graphrag] graph_db/embedder resources missing, component no-op")
            error = "resources missing"
        if tracer and chat_id:
            tracer.log_step(chat_id=chat_id, node="Comp_GraphRAG",
                            tool_result={"error": error} if error else {},
                            chunks=chunks, latency_ms=(time.time() - start) * 1000)
        return {"retrieval_pool": {"graphrag": chunks}}

    async def fusion(state, config):
        rp = _get_rp(config)
        tracer, chat_id = _tracer_ctx(config)
        start = time.time()
        pools = state.get("retrieval_pool") or {}
        merged = rrf_merge(pools, rp.rrf_k, rp.top_k)
        content = "\n".join(f"- [{c['source']}] {c['text']}" for c in merged)

        status = "SUCCESS" if merged else ("PLAIN" if not any(rp.flag_set().values()) else "FAIL")
        rag_result = {
            "thought": f"fusion over {list(pools.keys())} -> {len(merged)} chunks",
            "entity_ids": [c["id"] for c in merged if c.get("id")],
            "content": content,
            "status": status,
        }

        session_context = config.get("configurable", {}).get("session_context")
        if session_context and chat_id and content:
            session_context.store_tool_result(
                chat_id=chat_id, tool_name="retrieval_fusion",
                args={"query": state.get("user_query", "")}, output=content, node="Fusion",
            )

        if tracer and chat_id:
            tracer.log_step(chat_id=chat_id, node="Fusion",
                            tool_result={"sources": rp.flag_set(), "status": status},
                            chunks=merged, latency_ms=(time.time() - start) * 1000)

        current = state.get("worker_results", {})
        return {"worker_results": {**current, "RAG": rag_result}, "status_flag": status}

    def route_components(state):
        flags = state.get("_retrieve_flags") or {}
        targets = []
        if flags.get("rag"):
            targets.append("Comp_RAG")
        if flags.get("graphrag"):
            targets.append("Comp_GraphRAG")
        return targets or ["Fusion"]

    builder.add_node("Retrieve_Dispatch", dispatch)
    builder.add_node("Comp_RAG", comp_rag)
    builder.add_node("Comp_GraphRAG", comp_graphrag)
    builder.add_node("Fusion", fusion)

    builder.set_entry_point("Retrieve_Dispatch")
    builder.add_conditional_edges(
        "Retrieve_Dispatch",
        route_components,
        {"Comp_RAG": "Comp_RAG", "Comp_GraphRAG": "Comp_GraphRAG", "Fusion": "Fusion"},
    )
    builder.add_edge("Comp_RAG", "Fusion")
    builder.add_edge("Comp_GraphRAG", "Fusion")
    builder.add_edge("Fusion", END)

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