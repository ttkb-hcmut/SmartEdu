import logging
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from core.schema.wf_state import AgentState, ConceptNode
from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId, RetrievalRunContext
from core.config import Retrieve_param
from TA.helper.schema import TeachEvalOutput, TeachLectureOutput, NextTopicOutput
import TA.helper.prompt as prompt_lib
from TA.helper.few_shot import get_language_instruction
from TA.helper.utils import safe_parse_structured, extract_llm_raw_text
from TA.helper.context import build_ui_citations, extract_ta_context
from TA.helper.model_call import is_transient, ta_ainvoke
from TA.retrieval.policy import resolve_retrieval_context
from TA.tools.tool_config import PREREQUISITE_WEIGHT
from core.repo.graph.cypher.tools.course import CYPHER_get_recommendations
import os
from TA.tracing.tracer import AgentTracer

logger = logging.getLogger(__name__)


def build_teach_wf(agents, retrieve_wf, graph_db=None):
    builder = StateGraph(AgentState, context_schema=RetrievalRunContext)

    async def _teach_understand(state, config):
        return await teach_understand(state, agents["TA"], config)

    async def _teach_rag(state, config, runtime: Runtime[RetrievalRunContext]):
        return await teach_rag(state, retrieve_wf, graph_db, config, runtime)

    async def _teach_lecture(state, config):
        return await teach_lecture(state, agents["TA"], config)

    async def _teach_evaluate(state, config):
        return await teach_evaluate(state, agents["TA"], config)

    async def _next_topic(state, config):
        return await next_topic(state, agents["TA"], config)

    builder.add_node("Teach_Understand", _teach_understand)
    builder.add_node("Teach_RAG", _teach_rag)
    builder.add_node("Teach_Lecture", _teach_lecture)
    builder.add_node("Teach_Evaluate", _teach_evaluate)
    builder.add_node("Next_Topic", _next_topic)

    builder.set_entry_point("Teach_Understand")

    builder.add_conditional_edges(
        "Teach_Understand",
        lambda state: state.get("_teach_mode", "continue"),
        {
            "review": "Teach_RAG",
            "continue": "Teach_RAG",
            "evaluate": "Teach_Evaluate",
        },
    )

    builder.add_edge("Teach_RAG", "Teach_Lecture")
    builder.add_edge("Teach_Lecture", END)
    builder.add_edge("Teach_Evaluate", "Next_Topic")
    builder.add_edge("Next_Topic", END)

    return builder.compile()


def _bounded_history(tracker, session_id: str, chat_id: str) -> str:
    return tracker.get_chat_history(
        session_id,
        mode="skim",
        recent_turns=4,
        exclude_chat_id=chat_id,
        max_chars=4_000,
    )


# ─── Node 1: Intent Classification ─────────────────────────────────────────

async def teach_understand(state: AgentState, ta_agent, config):
    """ Raw text classification, no schema needed"""
    query = state.get("user_query", "")
    sid = config["configurable"]["session_id"]
    tracker = config["configurable"]["student_tracker"]
    tracer = config["configurable"].get("tracer")
    chat_id = config["configurable"].get("chat_id", "")
    history = _bounded_history(tracker, sid, chat_id)

    language = state.get("language", "vn")
    language_instruction = get_language_instruction(language)

    prompt = prompt_lib.TEACH_UNDERSTAND_PROMPT.format(
        language_instruction=language_instruction,
        history=history,
        query=query
    )

    ## -- Simple classification: raw invoke, parse single word
    model = ta_agent.model
    if model.__class__.__module__.startswith("langchain_openrouter"):
        model = model.bind(max_tokens=512)
    res = await ta_ainvoke(model, [("user", prompt)], config)

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


# ─── Node 3: RAG Fallback ──────────────────────────────────────────────────

async def teach_rag(state: AgentState, retrieve_wf, graph_db, config, runtime: Runtime[RetrievalRunContext]):
    sid = config["configurable"]["session_id"]
    tracker = config["configurable"]["student_tracker"]
    tracer = config["configurable"].get("tracer")
    chat_id = config["configurable"].get("chat_id", "")
    current_node = tracker.get_student_state(sid).get("current_pos")
    mode = state.get("_teach_mode", "continue")
    user_query = str(state.get("user_query", "")).strip()
    continue_only = user_query.casefold() in {"continue", "more", "tiếp tục", "dạy tiếp"}
    query = (
        f"Explain the concept of {current_node.name} in detail."
        if current_node and (not user_query or continue_only)
        else user_query
    )
    request_context = runtime.context
    teach_context = resolve_retrieval_context(
        Retrieve_param(
            preset=request_context.preset,
            policy_id=RetrievalPolicyId.BASELINE_V5,
            harness_id=RetrievalHarnessId.AGENTIC_V4,
            course_scope=request_context.scope.course,
        ),
        request_context.case,
        request_context.code,
    )
    rag_result = {}
    try:
        retrieved = await retrieve_wf.ainvoke(
            {**state, "messages": [{"role": "user", "content": query}],
             "user_query": query, "worker_results": {}},
            config=config,
            context=teach_context,
        )
        rag_result = retrieved.get("worker_results", {}).get("RAG", {})
    except Exception:
        logger.exception("[Teach_RAG] V4 retrieval failed")

    grounded = rag_result.get("status") == "SUCCESS" and rag_result.get("answerable")
    rag_content = rag_result.get("content", "") if grounded else ""
    citations = await build_ui_citations(rag_result, graph_db) if grounded else []

    if tracer and chat_id:
        tracer.log_step(
            chat_id=chat_id,
            node="Teach_RAG",
            prompt=query,
            state=tracker.get_student_state(sid),
            tool_result={
                "harness_id": teach_context.harness.id.value,
                "policy_id": teach_context.policy.id.value,
                "retrieval_status": rag_result.get("status", "FAIL"),
                "answerable": bool(grounded),
            },
            execution_config={
                "harness_id": teach_context.harness.id.value,
                "harness_digest": teach_context.harness.digest,
                "policy_id": teach_context.policy.id.value,
                "policy_digest": teach_context.policy.digest,
            },
            output=f"Retrieved {len(rag_content)} chars",
        )
    return {
        "worker_results": {**state.get("worker_results", {}), "RAG": rag_result},
        "_teach_context": {
            "source": "RAG" if grounded else "GENERAL",
            "content": rag_content,
            "page": None,
            "mode": mode,
        },
        "ui_action": {"citations": citations} if citations else None,
    }

# ─── Node 4: LLM Lecture Generation ────────────────────────────────────────

async def teach_lecture(state: AgentState, ta_agent, config):
    """Generate a lecture from retrieved evidence or general knowledge."""
    sid = config["configurable"]["session_id"]
    tracker = config["configurable"]["student_tracker"]
    tracer = config["configurable"].get("tracer")
    chat_id = config["configurable"].get("chat_id", "")
    session = tracker.get_session(sid)
    student_state = session.student_state
    history = _bounded_history(tracker, sid, chat_id)

    ctx = state.get("_teach_context", {})
    source = ctx.get("source", "unknown")
    mode = ctx.get("mode", "continue")

    current_node = student_state.get("current_pos")
    previous_nodes = student_state.get("previous_nodes", [])
    current_str = current_node.name if current_node else "None"
    user_query = str(state.get("user_query", "")).strip()

    language = state.get("language", "vn")
    language_instruction = get_language_instruction(language)

    prev_str = "\n".join(
        f"  {i+1}. {n.name} ({n.type})" for i, n in enumerate(previous_nodes[:3])
    ) or "  (no previous nodes)"

    rag_content = ctx.get("content", "")
    source_ref = "RAG knowledge base" if source == "RAG" else "general knowledge"
    content_hint = (
        f"Source content (RAG):\n{rag_content[:2500]}"
        if source == "RAG" else
        "No retrieved source available. Explain from general knowledge; do not claim a PDF source."
    )

    if mode == "review":
        prev_str_long = "\n".join(
            f"  {i+1}. {n.name} ({n.type})" for i, n in enumerate(previous_nodes[:6])
        ) or "  (no previous nodes)"
        prompt = prompt_lib.TEACH_REVIEW_PROMPT.format(
            language_instruction=language_instruction,
            query=user_query,
            previous_nodes=prev_str_long,
            current_node=current_str,
            source=source_ref,
            content=content_hint,
            history=history,
        )
    else:
        prompt = prompt_lib.TEACH_CONTINUE_PROMPT.format(
            language_instruction=language_instruction,
            query=user_query,
            previous_nodes=prev_str,
            current_node=current_str,
            source=source_ref,
            content=content_hint,
            history=history,
        )

    request_text = user_query or f"Continue learning about {current_str}."
    lesson_contract = (
        "Review the requested material and test recall without repeating questions from chat history."
        if mode == "review" else
        "For a full lesson, write 4–6 titled sections with substantive explanations. Cover each requested part; use concrete examples, a comparison, a common misconception, and a short recap when relevant. Do not compress this into a benchmark-style short answer. Respect an explicit request for brevity."
    )
    prompt += (
        "\n\nCURRENT TEACHING REQUEST — follow this over older prompt directions:\n"
        f"{request_text}\n"
        f"{lesson_contract}\n"
        "Use the source material already supplied above; do not call PDF lookup tools. "
        "If the source is general knowledge, do not claim the PDF supports the explanation. "
        "End with one question that checks understanding."
    )

    ## -- Inject prior TA messages for coherence
    ta_context = extract_ta_context(state)
    if ta_context:
        prompt = f"[Prior TA reasoning]:\n{ta_context}\n\n{prompt}"

    try:
        lecture_model = ta_agent.model.with_structured_output(TeachLectureOutput)
        messages = []
        system_prompt = getattr(ta_agent, "system_prompt_text", "")
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("user", prompt))
        res = await ta_ainvoke(lecture_model, messages, config)
    except Exception as e:
        if is_transient(e):
            raise
        logger.warning(f"[teach_lecture] Agent invoke failed: {e}. Attempting json_repair.")
        res = safe_parse_structured(extract_llm_raw_text(e), TeachLectureOutput)

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
    chat_id = config["configurable"].get("chat_id", "")
    history = _bounded_history(tracker, sid, chat_id)

    language = state.get("language", "vn")
    language_instruction = get_language_instruction(language)

    prompt = prompt_lib.TEACH_EVAL_PROMPT_V2.format(
        language_instruction=language_instruction,
        history=history
    )

    ## -- Direct structured output
    structured_llm = ta_agent.model.with_structured_output(TeachEvalOutput)
    try:
        eval_res: TeachEvalOutput = await ta_ainvoke(structured_llm, [("user", prompt)], config)
    except Exception as e:
        if is_transient(e):
            raise
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
        topic_res: NextTopicOutput = await ta_ainvoke(structured_llm, [("user", prompt)], config)
    except Exception as e:
        if is_transient(e):
            raise
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
