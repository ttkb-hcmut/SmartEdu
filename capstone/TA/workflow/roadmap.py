import time
import logging
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from core.schema.wf_state import AgentState
from TA.helper.schema import RoadmapExplore, RoadmapCritique, RoadmapFinal
from TA.helper.prompt import ROADMAP_PROMPT
from TA.helper.few_shot import get_language_instruction
from TA.helper.utils import filter_mastery, safe_parse_structured, extract_llm_raw_text, extract_agent_result, extract_kg_context
from TA.helper.context import extract_ta_context


from TA.tracing.tracer import AgentTracer
import os

logger = logging.getLogger(__name__)

# Cap on explore→evaluate cycles. 2 = one initial pass + one infeasibility retry.
MAX_ROADMAP_ATTEMPTS = 2


def _explore_router(state: AgentState):
    wr = state.get("worker_results", {})
    r = wr.get("Roadmap", {})
    if not r.get("steps") and not (r.get("goal") or "").strip():
        return END
    return "Roadmap_Evaluator"


def _evaluator_router(state: AgentState):
    """Close the loop: an infeasible verdict reruns Explore (where the KG tools live)
    instead of being narrated around by TA_Advice. Capped by MAX_ROADMAP_ATTEMPTS.

    ``is_feasible is not True`` deliberately catches a None left by a parse-fallback
    critique — an unknown verdict is treated as "not yet feasible" and retried, not
    silently shipped as final.
    """
    r = state.get("worker_results", {}).get("Roadmap", {})
    critique = r.get("critique", {})
    is_feasible = critique.get("is_feasible") if isinstance(critique, dict) else None
    attempts = state.get("roadmap_attempts", 0)

    if is_feasible is not True and attempts < MAX_ROADMAP_ATTEMPTS:
        return "Roadmap_Explore"
    return "TA_Advice"


def build_roadmap_wf(agents):
    builder = StateGraph(AgentState)

    async def _roadmap_explore(state, config):
        return await roadmap_explore_logic(state, agents["RAG"], config)

    async def _roadmap_evaluator(state, config):
        return await roadmap_evaluator_logic(state, agents["TA"], config)

    async def _ta_advice(state, config):
        return await ta_advice_logic(state, agents["TA"], config)

    builder.add_node("Roadmap_Explore", _roadmap_explore)
    builder.add_node("Roadmap_Evaluator", _roadmap_evaluator)
    builder.add_node("TA_Advice", _ta_advice)

    builder.set_entry_point("Roadmap_Explore")
    builder.add_conditional_edges("Roadmap_Explore", _explore_router)
    builder.add_conditional_edges("Roadmap_Evaluator", _evaluator_router)
    builder.add_edge("TA_Advice", END)

    return builder.compile()


async def roadmap_explore_logic(state: AgentState, rag_agent, config):
    """ Invoke global RAG agent with current_node=Roadmap_Explore for schema routing"""
    import os
    import traceback
    
    start = time.time()
    log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
    
    try:
        query = state.get("user_query", state["messages"][-1].content)
        student_state = state.get("student_state", {})
        current_pos = student_state.get("current_pos")

        language = state.get("language", "vn")
        language_instruction = get_language_instruction(language)

        if current_pos is None:
            instruction = ROADMAP_PROMPT['explore_new'].format(
                language_instruction=language_instruction,
                student_query=query
            )
        else:
            ## prepend frontier render when course tree + learning tree are cached
            tree_block = ""
            try:
                c = config.get("configurable", {})
                mongo = c.get("mongo_db")
                uid = c.get("student_id") or c.get("session_id", "")
                course = getattr(current_pos, "course_name", "") or ""
                if mongo and course:
                    cached = mongo.get_course_tree(course.title())
                    lt = mongo.get_learning_tree(uid, course)
                    if cached.get("tree"):
                        from TA.helper.tree_render import render_frontier
                        learning = {n["name"]: n for n in lt.get("nodes", [])}
                        tree_block = render_frontier(cached["tree"], learning,
                                                     current_pos.name, char_budget=1800)
            except Exception as e:
                logger.warning(f"[roadmap_explore_logic] render skipped: {e}")
            instruction = ROADMAP_PROMPT['explore_existing'].format(
                language_instruction=language_instruction,
                current_pos=current_pos.name if hasattr(current_pos, 'name') else str(current_pos),
                student_query=query
            )
            if tree_block:
                instruction = f"[Student's learning tree]\n{tree_block}\n\n{instruction}"

        current_results = state.get("worker_results", {})

        ##    back in so re-exploration corrects the gap instead of repeating it.
        prior_critique = current_results.get("Roadmap", {}).get("critique") or {}
        if prior_critique.get("is_feasible") is False:
            gap = prior_critique.get("thought") or prior_critique.get("ai_message") or ""
            if gap:
                instruction = (
                    f"[Previous roadmap was judged INFEASIBLE]:\n{gap}\n\n"
                    f"Revise the path to address this — favour prerequisite/foundation nodes "
                    f"the student is missing.\n\n{instruction}"
                )

        raw_context = ""
        try:
            result = await rag_agent.ainvoke(
                {"messages": [("user", instruction)], "current_node": "Roadmap_Explore"},
                config={"recursion_limit": 30, **config},
            )
        except Exception as e:
            logger.warning(f"[roadmap_explore_logic] Agent invoke failed: {e}. Attempting json_repair.")
            structured = safe_parse_structured(extract_llm_raw_text(e), RoadmapExplore)
        else:
            structured = extract_agent_result(result, RoadmapExplore, "roadmap_explore_logic")
            ## -- Capture KG tool output (course_relevance/backbone) from the agent trace.
            ##    It dies in intermediate messages otherwise, leaving the Evaluator blind.
            raw_context = extract_kg_context(result.get("messages", []))

        ## -- raw_context is set programmatically (the LLM never fills it); fall back to
        ##    any value the agent happened to emit so a retry doesn't clobber prior capture.
        structured.raw_context = raw_context or structured.raw_context

        report_lines = [
            f"========================= Agent \"{rag_agent.name}\" Runtime Report ==========================",
            f"[SMART_EDU_LOG] Node: Roadmap_Explore | Action: global_agent.ainvoke | Time: {time.time() - start:.4f}s",
            f"Start node: {structured.start_node.name if structured.start_node else 'None'}",
            f"Goal: {structured.goal}",
            f"Steps: {[node.name for node in structured.steps] if structured.steps else []}",
            f"Reasoning: {structured.thought}",
            f"*************** END Report ***************"
        ]
        report_str = "\n" + "\n".join(report_lines) + "\n"
        logger.info(report_str)
        
        if log_filename:
            log_dict = {
                "agent_name": rag_agent.name,
                "node": "Roadmap_Explore",
                "thought": structured.thought,
                "prompt": instruction[:300],
                "output": structured.model_dump()
            }
            AgentTracer.logging(log_dict, type="info", file_name=log_filename)

        ## -- Empty explore: no steps and no goal means the router will END here.
        ##    Flag it so the dead-end carries a reason instead of failing silently.
        is_empty = not structured.steps and not (structured.goal or "").strip()
        out = {
            "worker_results": {**current_results, "Roadmap": structured.model_dump()},
            "messages": [{"role": "assistant", "content": structured.ai_message}],
            "status_flag": "EMPTY" if is_empty else "SUCCESS",
        }
        if is_empty:
            out["error"] = "roadmap_explore_empty: no hub nodes matched in the knowledge graph"
            logger.warning("[roadmap_explore_logic] Empty exploration — routing to END with error flag.")
        return out
    except Exception as e:
        if log_filename:
            AgentTracer.logging(traceback.format_exc(), type="err", file_name=log_filename)
        raise e


async def roadmap_evaluator_logic(state: AgentState, ta_agent, config: RunnableConfig):
    """ No-tool node: direct structured output with RoadmapCritique"""
    current_results = state.get("worker_results", {})
    roadmap_data = current_results.get("Roadmap", {})
    student_state = state.get("student_state", {})
    mastery_map = student_state.get("mastery_map", {})

    relevant_mastery = filter_mastery(roadmap_data, mastery_map)

    language = state.get("language", "vn")
    language_instruction = get_language_instruction(language)

    instruction = ROADMAP_PROMPT['evaluate'].format(
        language_instruction=language_instruction,
        proposed_steps=roadmap_data.get("steps", []),
        student_mastery=relevant_mastery,
        course_relevance=roadmap_data.get("raw_context", "")
    )

    ## -- Direct structured output, no tool loop
    structured_llm = ta_agent.model.with_structured_output(RoadmapCritique)
    try:
        critique: RoadmapCritique = await structured_llm.ainvoke(
            [("user", instruction)], config=config
        )
    except Exception as e:
        logger.warning(f"[roadmap_evaluator_logic] Structured output failed: {e}. Attempting json_repair.")
        critique = safe_parse_structured(extract_llm_raw_text(e), RoadmapCritique)

    log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
    if log_filename:
        AgentTracer.logging({
            "agent_name": ta_agent.name,
            "node": "Roadmap_Evaluator",
            "thought": critique.thought if hasattr(critique, "thought") else "",
            "prompt": instruction[:300],
            "output": critique.model_dump() if hasattr(critique, "model_dump") else critique
        }, type="info", file_name=log_filename)

    updated_roadmap = {**roadmap_data, "critique": critique.model_dump()}
    return {
        "worker_results": {**current_results, "Roadmap": updated_roadmap},
        "messages": [{"role": "assistant", "content": critique.ai_message}],
        ## -- Bumped here (not in the router, which can't mutate state) so the
        ##    explore→evaluate loop is bounded by MAX_ROADMAP_ATTEMPTS.
        "roadmap_attempts": state.get("roadmap_attempts", 0) + 1,
    }


async def ta_advice_logic(state: AgentState, ta_agent, config: RunnableConfig):
    """ No-tool node: direct structured output with RoadmapFinal, TA history injected"""
    current_results = state.get("worker_results", {})
    roadmap_data = current_results.get("Roadmap", {})
    critique = roadmap_data.get("critique", {})

    ## -- 'thought' is the real reasoning field on RoadmapCritique; 'reasoning' never
    ##    existed on the schema and always piped "None" into this prompt.
    critique_str = f"Feasible: {critique.get('is_feasible')}\nReasoning: {critique.get('thought')}"

    language = state.get("language", "vn")
    language_instruction = get_language_instruction(language)

    instruction = ROADMAP_PROMPT['advice'].format(
        language_instruction=language_instruction,
        backbone=roadmap_data.get('steps', []),
        critique=critique_str
    )

    ## -- Inject prior TA messages for coherence (TA model only)
    ta_context = extract_ta_context(state)
    if ta_context:
        instruction = f"[Prior TA reasoning]:\n{ta_context}\n\n{instruction}"

    ## -- Direct structured output, no tool loop
    structured_llm = ta_agent.model.with_structured_output(RoadmapFinal)
    try:
        advice: RoadmapFinal = await structured_llm.ainvoke(
            [("user", instruction)], config=config
        )
    except Exception as e:
        logger.warning(f"[ta_advice_logic] Structured output failed: {e}. Attempting json_repair.")
        advice = safe_parse_structured(extract_llm_raw_text(e), RoadmapFinal)

    log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
    if log_filename:
        AgentTracer.logging({
            "agent_name": ta_agent.name,
            "node": "TA_Advice",
            "thought": advice.thought if hasattr(advice, "thought") else "",
            "prompt": instruction[:300],
            "output": advice.model_dump() if hasattr(advice, "model_dump") else advice
        }, type="info", file_name=log_filename)

    updated_roadmap = {**roadmap_data, "advice": advice.model_dump()}

    ## seed Mongo learning tree from final backbone (ADR-0003); never kills the reply
    try:
        c = config.get("configurable", {})
        mongo = c.get("mongo_db")
        uid = c.get("student_id") or c.get("session_id", "")
        course = (advice.final_steps[0].course_name if advice.final_steps else "") or ""
        if mongo and uid and course:
            mongo.seed_learning_tree(uid, course,
                                     [{"name": n.name, "type": n.type} for n in advice.final_steps])
    except Exception as seed_err:
        logger.warning(f"[ta_advice_logic] learning-tree seed skipped: {seed_err}")

    return {
        "worker_results": {**current_results, "Roadmap": updated_roadmap},
        "messages": [{"role": "assistant", "content": advice.ai_message}],
        "status_flag": "SUCCESS",
    }
