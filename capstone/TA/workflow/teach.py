import json
import logging
from langgraph.graph import StateGraph, END
from core.schema.wf_state import AgentState, ConceptNode
from TA.helper.schema import TeachEvalOutput, TeachLectureOutput, NextTopicOutput
import TA.helper.prompt as prompt_lib
from TA.helper.few_shot import get_language_instruction
from TA.helper.utils import safe_parse_structured, extract_llm_raw_text, extract_agent_result
from TA.helper.context import extract_ta_context
from TA.tools.tool_config import PREREQUISITE_WEIGHT
from core.repo.graph.cypher.tools.course import CYPHER_get_recommendations
import os
from TA.tracing.tracer import AgentTracer

logger = logging.getLogger(__name__)


def build_teach_wf(agents):
    builder = StateGraph(AgentState)

    async def _teach_understand(state, config):
        return await teach_understand(state, agents["TA"], config)

    async def _teach_lookup(state, config):
        return await teach_lookup(state, config)

    async def _teach_rag(state, config):
        return await teach_rag(state, agents["RAG"], config)

    async def _teach_lecture(state, config):
        return await teach_lecture(state, agents["TA"], config)

    async def _teach_evaluate(state, config):
        return await teach_evaluate(state, agents["TA"], config)

    async def _next_topic(state, config):
        return await next_topic(state, agents["TA"], config)

    builder.add_node("Teach_Understand", _teach_understand)
    builder.add_node("Teach_Lookup", _teach_lookup)
    builder.add_node("Teach_RAG", _teach_rag)
    builder.add_node("Teach_Lecture", _teach_lecture)
    builder.add_node("Teach_Evaluate", _teach_evaluate)
    builder.add_node("Next_Topic", _next_topic)

    builder.set_entry_point("Teach_Understand")

    builder.add_conditional_edges(
        "Teach_Understand",
        lambda state: state.get("_teach_mode", "continue"),
        {
            "review": "Teach_Lookup",
            "continue": "Teach_Lookup",
            "evaluate": "Teach_Evaluate",
        },
    )

    builder.add_conditional_edges(
        "Teach_Lookup",
        _route_after_lookup,
        {
            "has_content": "Teach_Lecture",
            "no_content": "Teach_RAG",
        },
    )

    builder.add_edge("Teach_RAG", "Teach_Lecture")
    builder.add_edge("Teach_Lecture", END)
    builder.add_edge("Teach_Evaluate", "Next_Topic")
    builder.add_edge("Next_Topic", END)

    return builder.compile()


def _route_after_lookup(state: AgentState) -> str:
    ctx = state.get("_teach_context", {})
    return "has_content" if ctx.get("source") == "PDF" else "no_content"


# ─── Node 1: Intent Classification ─────────────────────────────────────────

async def teach_understand(state: AgentState, ta_agent, config):
    """ Raw text classification, no schema needed"""
    query = state.get("user_query", "")
    sid = config["configurable"]["session_id"]
    tracker = config["configurable"]["student_tracker"]
    tracer = config["configurable"].get("tracer")
    chat_id = config["configurable"].get("chat_id", "")
    history = tracker.get_chat_history(sid)

    language = state.get("language", "vn")
    language_instruction = get_language_instruction(language)

    prompt = prompt_lib.TEACH_UNDERSTAND_PROMPT.format(
        language_instruction=language_instruction,
        history=history,
        query=query
    )

    ## -- Simple classification: raw invoke, parse single word
    res = await ta_agent.ainvoke(
        [("user", prompt)], config=config
    )

    mode = res.content.strip().lower()
    if mode not in ("review", "continue", "evaluate"):
        mode = "continue"

    log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
    if log_filename:
        AgentTracer.logging({
            "agent_name": ta_agent.name,
            "node": "Teach_Understand",
            "prompt": prompt[:300],
            "output": {"mode": mode}
        }, type="info", file_name=log_filename)

    if tracer and chat_id:
        tracer.log_step(
            chat_id=chat_id,
            node="Teach_Understand",
            prompt=prompt,
            state=tracker.get_student_state(sid),
            output=mode,
        )

    return {"_teach_mode": mode}


# ─── Node 2: Deterministic PDF Lookup (No LLM) ─────────────────────────────

async def teach_lookup(state: AgentState, config):
    """ Deterministic PDF lookup, no LLM needed"""
    sid = config["configurable"]["session_id"]
    tracker = config["configurable"]["student_tracker"]
    tracer = config["configurable"].get("tracer")
    chat_id = config["configurable"].get("chat_id", "")
    teach_tools = config["configurable"].get("teach_tools", {})
    session = tracker.get_session(sid)
    student_state = session.student_state

    current_node = student_state.get("current_pos")
    mode = state.get("_teach_mode", "continue")
    active_resource = student_state.get("active_resource")

    no_content = {
        "_teach_context": {"source": "NO_CONTENT", "content": "", "page": None, "mode": mode}
    }

    def _log_and_return(result):
        log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
        if log_filename:
            ctx = result.get("_teach_context", {})
            AgentTracer.logging({
                "agent_name": "Teach_Lookup",
                "node": "Teach_Lookup",
                "prompt": f"Lookup concept: {current_node.name if current_node else 'None'}",
                "output": {
                    "source": ctx.get("source", "NO_CONTENT"),
                    "page": ctx.get("page"),
                    "storage_uri": ctx.get("storage_uri")
                }
            }, type="info", file_name=log_filename)
        return result

    if not current_node:
        logger.info("[Teach_Lookup] No current_pos, routing to RAG")
        return _log_and_return(no_content)

    concept_tool = teach_tools.get("get_concept")
    pages_tool = teach_tools.get("get_pages")

    if not concept_tool or not pages_tool:
        logger.warning("[Teach_Lookup] teach_tools not injected, routing to RAG")
        return _log_and_return(no_content)

    ## -- Step 1: Find pages containing the concept
    try:
        concept_result = concept_tool._run(concept=current_node.name)
        parsed = json.loads(concept_result)
    except Exception as e:
        logger.warning(f"[Teach_Lookup] GetConcept failed: {e}")
        return _log_and_return(no_content)

    if isinstance(parsed, dict) and "error" in parsed:
        logger.info(f"[Teach_Lookup] Concept '{current_node.name}' not in PDF, routing to RAG")
        return _log_and_return(no_content)

    if not parsed or not isinstance(parsed, list) or len(parsed) == 0:
        return _log_and_return(no_content)

    ## -- Step 2: Read page content
    hard_ref_str = parsed[0].get("hard_ref")
    if not hard_ref_str:
        return _log_and_return(no_content)

    try:
        ref_data = json.loads(hard_ref_str)
        p_list = ref_data.get("p_num", [])
        page_num = p_list[0] if isinstance(p_list, list) and p_list else p_list
        storage_uri = ref_data.get("id")
    except Exception as e:
        logger.warning(f"[Teach_Lookup] Failed to parse hard_ref: {e}")
        return _log_and_return(no_content)

    if not storage_uri or not page_num:
        return _log_and_return(no_content)

    try:
        page_content = pages_tool._run(
            pages=[page_num, page_num + 1],
            destination=storage_uri,
        )
    except Exception as e:
        logger.warning(f"[Teach_Lookup] GetPages failed: {e}")
        return _log_and_return(no_content)

    if "error" in page_content.lower() or len(page_content.strip()) < 50:
        logger.info("[Teach_Lookup] PDF content too short, routing to RAG")
        return _log_and_return(no_content)

    if tracer and chat_id:
        tracer.log_step(
            chat_id=chat_id,
            node="Teach_Lookup",
            prompt=f"Lookup concept: {current_node.name}",
            state=tracker.get_student_state(sid),
            output=f"PDF source: {ref_data.get('name')}, page {page_num}, {len(page_content)} chars",
        )

    return _log_and_return({
        "_teach_context": {
            "source": "PDF",
            "content": page_content,
            "page": page_num,
            "storage_uri": storage_uri,
            "mode": mode,
        },
        "ui_action": {"navigate_page": page_num, "document": storage_uri},
    })


# ─── Node 3: RAG Fallback ──────────────────────────────────────────────────

async def teach_rag(state: AgentState, rag_agent, config):
    """ Reuse rag_core from retrieve.py as fallback"""
    from TA.workflow.retrieve import rag_core

    sid = config["configurable"]["session_id"]
    tracker = config["configurable"]["student_tracker"]
    tracer = config["configurable"].get("tracer")
    chat_id = config["configurable"].get("chat_id", "")
    current_node = tracker.get_student_state(sid).get("current_pos")
    mode = state.get("_teach_mode", "continue")

    concept_name = current_node.name if current_node else "the current topic"

    mock_message = {"role": "user", "content": f"Explain the concept of {concept_name} in detail."}
    temp_state = {
        **state,
        "messages": state["messages"] + [mock_message],
        "user_query": f"Explain {concept_name}",
    }

    rag_result = await rag_core(temp_state, rag_agent, config)
    rag_content = rag_result.get("worker_results", {}).get("RAG", {}).get("content", "")

    log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
    if log_filename:
        AgentTracer.logging({
            "agent_name": rag_agent.name,
            "node": "Teach_RAG",
            "prompt": f"RAG fallback for: {concept_name}",
            "output": {"content_len": len(rag_content)}
        }, type="info", file_name=log_filename)

    if tracer and chat_id:
        tracer.log_step(
            chat_id=chat_id,
            node="Teach_RAG",
            prompt=f"RAG fallback for: {concept_name}",
            state=tracker.get_student_state(sid),
            output=f"RAG retrieved {len(rag_content)} chars",
        )

    return {
        "worker_results": rag_result.get("worker_results", {}),
        "_teach_context": {
            "source": "RAG",
            "content": rag_content,
            "page": None,
            "mode": mode,
        },
    }


# ─── Node 4: LLM Lecture Generation ────────────────────────────────────────

async def teach_lecture(state: AgentState, ta_agent, config):
    """
    TA agent (tool-calling) reads PDF content then generates lecture.
    - If Teach_Lookup found a PDF page: inject page ref into prompt, TA calls get_pdf_pages itself
    - If RAG fallback: content already in _teach_context
    TA must read the source material via tool before generating lecture.
    """
    sid = config["configurable"]["session_id"]
    tracker = config["configurable"]["student_tracker"]
    tracer = config["configurable"].get("tracer")
    chat_id = config["configurable"].get("chat_id", "")
    session = tracker.get_session(sid)
    student_state = session.student_state
    history = tracker.get_chat_history(sid)

    ctx = state.get("_teach_context", {})
    source = ctx.get("source", "unknown")
    mode = ctx.get("mode", "continue")

    current_node = student_state.get("current_pos")
    previous_nodes = student_state.get("previous_nodes", [])
    current_str = current_node.name if current_node else "None"

    language = state.get("language", "vn")
    language_instruction = get_language_instruction(language)

    prev_str = "\n".join(
        f"  {i+1}. {n.name} ({n.type})" for i, n in enumerate(previous_nodes[:3])
    ) or "  (no previous nodes)"

    if source == "PDF":
        page_num = ctx.get("page")
        storage_uri = ctx.get("storage_uri", "")
        # TA receives page ref and must call get_pdf_pages to read before lecturing
        source_ref = f"PDF page {page_num} at '{storage_uri}'" if page_num else "the PDF document"
        content_hint = f"Call get_pdf_pages(pages=[{page_num}, {page_num + 1 if page_num else ''}], destination='{storage_uri}') to read the content first."
    else:
        # RAG fallback: content already provided
        rag_content = ctx.get("content", "")
        source_ref = "RAG knowledge base"
        content_hint = f"Source content (RAG):\n{rag_content[:2500]}"

    if mode == "review":
        prev_str_long = "\n".join(
            f"  {i+1}. {n.name} ({n.type})" for i, n in enumerate(previous_nodes[:6])
        ) or "  (no previous nodes)"
        prompt = prompt_lib.TEACH_REVIEW_PROMPT.format(
            language_instruction=language_instruction,
            previous_nodes=prev_str_long,
            current_node=current_str,
            source=source_ref,
            content=content_hint,
            history=history,
        )
    else:
        prompt = prompt_lib.TEACH_CONTINUE_PROMPT.format(
            language_instruction=language_instruction,
            previous_nodes=prev_str,
            current_node=current_str,
            source=source_ref,
            content=content_hint,
            history=history,
        )

    ## -- Inject prior TA messages for coherence
    ta_context = extract_ta_context(state)
    if ta_context:
        prompt = f"[Prior TA reasoning]:\n{ta_context}\n\n{prompt}"

    ## -- TA agent invokes as tool-calling agent, reads PDF via get_pdf_pages then lectures
    try:
        result = await ta_agent.ainvoke(
            {"messages": [("user", prompt)], "current_node": "Teach_Lecture"},
            config=config,
        )
    except Exception as e:
        logger.warning(f"[teach_lecture] Agent invoke failed: {e}. Attempting json_repair.")
        res = safe_parse_structured(extract_llm_raw_text(e), TeachLectureOutput)
    else:
        res = extract_agent_result(result, TeachLectureOutput, "teach_lecture")

    lecture_text = res.lecture
    if res.challenge_question:
        lecture_text += f"\n\n**C\u00e2u h\u1ecfi ki\u1ec3m tra:** {res.challenge_question}"

    log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
    if log_filename:
        AgentTracer.logging({
            "agent_name": ta_agent.name,
            "node": "Teach_Lecture",
            "thought": res.thought if hasattr(res, "thought") else "",
            "prompt": prompt[:300],
            "output": res.model_dump() if hasattr(res, "model_dump") else res
        }, type="info", file_name=log_filename)

    if tracer and chat_id:
        tracer.log_step(
            chat_id=chat_id,
            node=f"Teach_Lecture ({mode})",
            prompt=prompt[:500],
            state=tracker.get_student_state(sid),
            output=lecture_text[:500],
        )

    result_key = "Teach_Review" if mode == "review" else "Teach_Lecture"
    current_results = state.get("worker_results", {})
    return {
        "worker_results": {**current_results, result_key: lecture_text},
        "messages": [{"role": "assistant", "content": lecture_text}],
        "_teach_context": {**ctx, "lecture_output": res.model_dump()},
        "status_flag": "SUCCESS",
    }


# ─── Node 5: Evaluation ────────────────────────────────────────────────────

async def teach_evaluate(state: AgentState, ta_agent, config):
    """ No-tool node: raw_model with TeachEvalOutput schema"""
    sid = config["configurable"]["session_id"]
    tracker = config["configurable"]["student_tracker"]
    history = tracker.get_chat_history(sid)

    language = state.get("language", "vn")
    language_instruction = get_language_instruction(language)

    prompt = prompt_lib.TEACH_EVAL_PROMPT_V2.format(
        language_instruction=language_instruction,
        history=history
    )

    ## -- Direct structured output
    structured_llm = ta_agent.model.with_structured_output(TeachEvalOutput)
    try:
        eval_res: TeachEvalOutput = await structured_llm.ainvoke(
            [("user", prompt)], config=config
        )
    except Exception as e:
        logger.warning(f"[teach_evaluate] Structured output failed: {e}. Attempting json_repair.")
        eval_res = safe_parse_structured(extract_llm_raw_text(e), TeachEvalOutput)


    log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
    if log_filename:
        AgentTracer.logging({
            "agent_name": ta_agent.name,
            "node": "Teach_Evaluate",
            "thought": eval_res.thought if hasattr(eval_res, "thought") else "",
            "prompt": prompt[:300],
            "output": eval_res.model_dump() if hasattr(eval_res, "model_dump") else eval_res
        }, type="info", file_name=log_filename)

    current_results = state.get("worker_results", {})
    return {
        "worker_results": {**current_results, "Teach_Eval": eval_res.model_dump()},
        "_teach_context": {"eval_result": eval_res.model_dump()},
    }


# ─── Node 6: Next Topic Selection ──────────────────────────────────────────

async def next_topic(state: AgentState, ta_agent, config):
    """ No-tool node: raw_model with NextTopicOutput schema"""
    sid = config["configurable"]["session_id"]
    tracker = config["configurable"]["student_tracker"]
    session = tracker.get_session(sid)
    student_state = session.student_state
    eval_data = state.get("_teach_context", {}).get("eval_result", {})

    current_node = student_state.get("current_pos")
    current_str = current_node.name if current_node else "None"

    if not eval_data.get("passed", False):
        log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
        if log_filename:
            AgentTracer.logging({
                "agent_name": "Next_Topic",
                "node": "Next_Topic",
                "prompt": "Evaluation stay check",
                "output": {"action": "STAY", "reason": "Evaluation not passed"}
            }, type="info", file_name=log_filename)
        return {
            "worker_results": {
                **state.get("worker_results", {}),
                "Next_Topic": {"action": "STAY", "reason": "Evaluation not passed"},
            }
        }

    recommend_result = _get_recommendations(tracker.graphdb, current_node)

    language = state.get("language", "vn")
    language_instruction = get_language_instruction(language)

    prompt = prompt_lib.NEXT_TOPIC_PROMPT.format(
        language_instruction=language_instruction,
        passed=eval_data.get("passed"),
        user_eval=eval_data.get("user_eval", ""),
        current_node=current_str,
        recommend_list=recommend_result,
    )

    ## -- Direct structured output
    structured_llm = ta_agent.model.with_structured_output(NextTopicOutput)
    try:
        topic_res: NextTopicOutput = await structured_llm.ainvoke(
            [("user", prompt)], config=config
        )
    except Exception as e:
        logger.warning(f"[next_topic] Structured output failed: {e}. Attempting json_repair.")
        topic_res = safe_parse_structured(extract_llm_raw_text(e), NextTopicOutput)


    selected_names = topic_res.selected_nodes
    next_nodes = [ConceptNode(name=name) for name in selected_names if isinstance(name, str)]

    if current_node:
        tracker.add_finished_community(sid, current_node)

    proposal = None
    if next_nodes:
        proposal = {
            "type": "advance",
            "new_current": next_nodes[0].model_dump(),
            "new_upcoming": [n.model_dump() for n in next_nodes[1:]],
            "reason": f"Eval passed. Next: {next_nodes[0].name}",
            "source_wf": "Teach",
            "auto_apply": True,
        }

    log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
    if log_filename:
        AgentTracer.logging({
            "agent_name": ta_agent.name,
            "node": "Next_Topic",
            "thought": topic_res.thought if 'topic_res' in locals() and hasattr(topic_res, "thought") else "",
            "prompt": prompt[:300],
            "output": topic_res.model_dump() if 'topic_res' in locals() and hasattr(topic_res, "model_dump") else {}
        }, type="info", file_name=log_filename)

    result = {
        "worker_results": {
            **state.get("worker_results", {}),
            "Next_Topic": {
                "action": "ADVANCE" if next_nodes else "NO_CANDIDATES",
                "selected_nodes": [n.name for n in next_nodes],
            },
        },
        "student_state": student_state,
    }
    if proposal:
        result["pending_proposal"] = proposal
    return result


def _get_recommendations(graphdb, current_node) -> str:
    if not current_node:
        return "No current position."

    results = graphdb.run_query(graphdb.db_name, CYPHER_get_recommendations, {"name": current_node.name, "prereq_weight": PREREQUISITE_WEIGHT})

    if not results:
        return "No neighboring nodes found."

    lines = []
    for i, r in enumerate(results, 1):
        desc = r.get("content", "")[:50] if r.get("content") else "N/A"
        lines.append(
            f"{i}. [{r.get('type', '')}] {r['name']} "
            f"| connections: {r.get('out_degree', 0)} | {desc}"
        )
    return "\n".join(lines)
