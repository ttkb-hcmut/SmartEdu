import re
import time
import logging
from datetime import datetime
from typing import List, Optional, Any, Dict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

import TA.edu.helper.prompt as prompt_lib
from TA.edu.helper.few_shot import format_few_shot, get_language_instruction
from TA.edu.helper.schema import RouterDecision
from core.schema.wf_state import AgentState, ConceptNode, TAOutput

from TA.edu.workflow.retrieve import build_retrieve_wf, _get_rp
from TA.edu.workflow.roadmap import build_roadmap_wf
from TA.edu.workflow.teach import build_teach_wf

from TA.edu.helper.utils import parse_student_state
from TA.edu.helper.context import extract_ta_context
import os
from TA.tracing.tracer import AgentTracer



from core.config import TEST_LOG

logger = logging.getLogger(__name__)


## node->label for SSE steps; (vn, eng). Arrives post-node so label trails by one
_STEP_LABELS = {
    "TA_Router": ("Đang phân tích câu hỏi…", "Analyzing your question…"),
    "WF_Retrieve": ("Đang tra cứu kiến thức…", "Searching the knowledge base…"),
    "WF_Roadmap": ("Đang lập lộ trình…", "Planning your roadmap…"),
    "WF_Teach": ("Đang chuẩn bị bài giảng…", "Preparing the lesson…"),
    "Apply_Proposal": ("Đang áp dụng lộ trình…", "Applying the roadmap…"),
}
_FINISH_LABEL = ("Đang viết câu trả lời…", "Writing the answer…")


def _humanize(node: str, lang: str) -> str:
    i = 1 if lang == "eng" else 0
    if node.startswith("TA_") and node.endswith("_Finish"):
        return _FINISH_LABEL[i]
    return _STEP_LABELS.get(node, ("Đang xử lý…", "Working…"))[i]


def _resource_ctx(config: RunnableConfig):
    """Get session_id, student_id, student_tracker, session_context from graph config."""
    c = config["configurable"]
    return c["session_id"], c.get("student_id", ""), c["student_tracker"], c.get("session_context")


def _tracer_ctx(config: RunnableConfig):
    c = config["configurable"]
    return c.get("tracer"), c.get("chat_id", "")





class SmartEdu:
    def __init__(self, agents, teach_tools: Dict = None, retrieve_res: Dict = None):
        self.agents = agents
        self.teach_tools = teach_tools or {}
        self.retrieve_res = retrieve_res or {}
        self.app = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("TA_Router", self.ta_router_node)

        builder.add_node("WF_Retrieve", build_retrieve_wf(agents=self.agents, resources=self.retrieve_res))
        builder.add_node("WF_Roadmap", build_roadmap_wf(agents=self.agents))
        builder.add_node("WF_Teach", build_teach_wf(agents=self.agents))

        builder.add_node("TA_Retrieve_Finish", self.ta_retrieve_finish)
        builder.add_node("TA_Roadmap_Finish", self.ta_roadmap_finish)
        builder.add_node("TA_Teach_Finish", self.ta_teach_finish)
        builder.add_node("Apply_Proposal", self.apply_proposal_node)
        builder.add_node("TA_Unknown_Finish", self.ta_unknown_finish)

        builder.set_entry_point("TA_Router")

        builder.add_conditional_edges(
            "TA_Router",
            lambda x: x["intent"],
            {
                "retrieve": "WF_Retrieve",
                "roadmap": "WF_Roadmap",
                "teaching": "WF_Teach",
                "confirm": "Apply_Proposal",
                "unknown": "TA_Unknown_Finish",
            }
        )

        builder.add_edge("WF_Retrieve", "TA_Retrieve_Finish")
        builder.add_edge("WF_Roadmap", "TA_Roadmap_Finish")
        builder.add_edge("WF_Teach", "TA_Teach_Finish")

        builder.add_edge("TA_Retrieve_Finish", END)
        builder.add_edge("TA_Roadmap_Finish", END)
        builder.add_edge("TA_Teach_Finish", END)
        builder.add_edge("Apply_Proposal", END)
        builder.add_edge("TA_Unknown_Finish", END)

        return builder.compile()

    async def ta_router_node(self, state: AgentState, config: RunnableConfig):
        """ No-tool node: direct structured output with RouterDecision"""
        sid, _uid, tracker, session_context = _resource_ctx(config)
        tracer, chat_id = _tracer_ctx(config)
        ta = self.agents["TA"]
        S_state = parse_student_state(tracker.get_student_state(sid))
        history = tracker.get_chat_history(sid, mode="skim")
        language = state.get("language", "vn")

        few_shot = format_few_shot(language)
        language_instruction = get_language_instruction(language)

        prompt = prompt_lib.ROUTER_PROMPT.format(
            state=S_state,
            query=state.get("user_query", ""),
            history=history,
            few_shot=few_shot,
            language_instruction=language_instruction,
        )

        start_invoke = time.time()
        ## -- Router stays as lightweight raw LLM call (speed matters here)
        ## -- num_predict must exceed gpt-oss hidden-reasoning budget (~150-320 tok);
        ##    100 truncated the model before the final-channel word → empty content → 'unknown'.
        llm = ta.model.bind(options={"temperature": 0, "num_predict": 256})
        res = await llm.ainvoke(
            [("user", prompt)],
            config=config
        )
        ## -- Take the LAST allowed token: prompt reasons first, then emits the word.
        raw = (res.content or "").lower()
        matches = re.findall(r"\b(retrieve|roadmap|teaching|confirm|unknown)\b", raw)
        intent = matches[-1] if matches else "unknown"
        logger.info(f"[SMART_EDU_LOG] Node: TA_Router | Time: {time.time() - start_invoke:.4f}s |")
        logger.info(f"[SMART_EDU_LOG] Intent: {intent} | Response: {res}")
        if not matches:
            logger.warning(
                f"[ta_router_node] No intent token in LLM response; defaulting to 'unknown'. "
                f"query={state.get('user_query', '')!r} raw={res.content!r}"
            )

        log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
        if log_filename:
            AgentTracer.logging({
                "agent_name": "TA_Router",
                "node": "TA_Router",
                "prompt": prompt[:300],
                "output": {"intent": intent}
            }, type="info", file_name=log_filename)

        if tracer and chat_id:
            tracer.log_step(
                chat_id=chat_id,
                node="TA_Router",
                prompt=prompt,
                state=tracker.get_student_state(sid),
                output=intent,
            )

        return {"intent": intent}

    async def ta_retrieve_finish(self, state: AgentState, config: RunnableConfig):
        """ TA synthesis node — tool-calling agent reads context then synthesizes"""
        sid, uid, tracker, session_context = _resource_ctx(config)
        tracer, chat_id = _tracer_ctx(config)
        ta = self.agents["TA"]
        language = state.get("language", "vn")
        language_instruction = get_language_instruction(language)

        proposal = tracker.get_session(sid).student_state.get("pending_proposal")
        if proposal and proposal.get("auto_apply"):
            tracker.apply_proposal(sid, proposal)

        results = state.get("worker_results", {})
        rp = _get_rp(config)
        if rp.preset == "PLAIN":
            ## PLAIN floor: prompt must not point at retrieval data
            refine_prompt = prompt_lib.RETRIEVE_PLAIN_PROMPT.format(
                language_instruction=language_instruction
            )
            prompt = f"{refine_prompt}\nQuestion: {state.get('user_query', '')}"
        else:
            refine_prompt = prompt_lib.RETRIEVE_REFINE_PROMPT.format(
                language_instruction=language_instruction
            )
            prompt = f"{refine_prompt}\nData: {results}"

        ## -- Inject prior TA messages for coherence
        ta_context = extract_ta_context(state)
        if ta_context:
            prompt = f"[Prior TA reasoning]:\n{ta_context}\n\n{prompt}"

        start_invoke = time.time()
        message = await self._stream_answer(ta, prompt, config)
        ta_output = TAOutput(summary=self._derive_summary(message), message=message)

        log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
        if log_filename:
            AgentTracer.logging({
                "agent_name": ta.name,
                "node": "TA_Retrieve_Finish",
                "thought": ta_output.summary,
                "prompt": prompt[:300],
                "output": ta_output.model_dump()
            }, type="info", file_name=log_filename)

        # --- Background: memoize & persist (non-blocking, atomic $push) ---
        async def _bg_retrieve(ta_out: TAOutput, cid: str):
            if uid and cid:
                tracker.mongodb.push_chat_message(
                    uid, sid, cid,
                    {"role": ta.name, "heading": "Synthesizing Retrieval", "message": ta_out.summary}
                )
            await self._save_ta_memo(sid, cid, tracker, ta_out)
            tracker.save_state(sid)

        await _bg_retrieve(ta_output, chat_id)  ## await -> persist before turn returns, no race/lost write

        logger.info(f"[SMART_EDU_LOG] Node: TA_Retrieve_Finish | Time: {time.time() - start_invoke:.4f}s")

        if tracer and chat_id:
            tracer.log_step(
                chat_id=chat_id,
                node="TA_Retrieve_Finish",
                prompt=prompt,
                state=tracker.get_student_state(sid),
                tool_result=results,
                output=ta_output.message,
            )

        return {"messages": [AIMessage(content=ta_output.message)], "pending_proposal": None}

    async def ta_roadmap_finish(self, state: AgentState, config: RunnableConfig):
        """ TA synthesis node — tool-calling agent"""
        sid, uid, tracker, session_context = _resource_ctx(config)
        tracer, chat_id = _tracer_ctx(config)
        ta = self.agents["TA"]
        language = state.get("language", "vn")
        language_instruction = get_language_instruction(language)

        roadmap_data = state.get("worker_results", {}).get("Roadmap", {})
        steps = roadmap_data.get("steps", [])
        advice = roadmap_data.get("advice", {})
        start_node = roadmap_data.get("start_node")

        proposal = None
        if steps:
            new_upcoming = [ConceptNode(name=s) if isinstance(s, str) else ConceptNode(**s) for s in steps]
            new_current = ConceptNode(name=start_node) if start_node else (new_upcoming[0] if new_upcoming else None)

            proposal = {
                "type": "roadmap",
                "new_current": new_current.model_dump() if new_current else None,
                "new_upcoming": [n.model_dump() for n in new_upcoming],
                "reason": advice.get("pedagogical_advice", "") if isinstance(advice, dict) else str(advice),
                "source_wf": "Roadmap",
                "auto_apply": False,
            }
            session = tracker.get_session(sid)
            session.student_state["pending_proposal"] = proposal

        if proposal:
            prompt = prompt_lib.PROPOSAL_PRESENT_PROMPT.format(
                language_instruction=language_instruction,
                type=proposal["type"],
                reason=proposal["reason"][:200],
                new_current=proposal["new_current"]["name"] if proposal["new_current"] else "N/A",
                new_upcoming=", ".join(n["name"] for n in proposal["new_upcoming"][:5]),
                source_wf=proposal["source_wf"],
            )
        else:
            prompt = f"{language_instruction}\n\nFriendly narrative response for this roadmap and pedagogical advice:\n{advice}"

        ## -- Inject prior TA messages for coherence
        ta_context = extract_ta_context(state)
        if ta_context:
            prompt = f"[Prior TA reasoning]:\n{ta_context}\n\n{prompt}"

        start_invoke = time.time()
        message = await self._stream_answer(ta, prompt, config)
        ta_output = TAOutput(summary=self._derive_summary(message), message=message)

        log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
        if log_filename:
            AgentTracer.logging({
                "agent_name": ta.name,
                "node": "TA_Roadmap_Finish",
                "thought": ta_output.summary,
                "prompt": prompt[:300],
                "output": ta_output.model_dump()
            }, type="info", file_name=log_filename)

        # --- Background: memoize & persist (non-blocking, atomic $push) ---
        async def _bg_roadmap(ta_out: TAOutput, cid: str):
            if uid and cid:
                tracker.mongodb.push_chat_message(
                    uid, sid, cid,
                    {"role": ta.name, "heading": "Planning Roadmap", "message": ta_out.summary}
                )
            await self._save_ta_memo(sid, cid, tracker, ta_out)
            tracker.save_state(sid)

        await _bg_roadmap(ta_output, chat_id)

        logger.info(f"[SMART_EDU_LOG] Node: TA_Roadmap_Finish | Time: {time.time() - start_invoke:.4f}s")

        if tracer and chat_id:
            tracer.log_step(
                chat_id=chat_id,
                node="TA_Roadmap_Finish",
                prompt=prompt,
                state=tracker.get_student_state(sid),
                tool_result=roadmap_data,
                output=ta_output.message,
            )

        return {"messages": [AIMessage(content=ta_output.message)]}  ## proposal owned by session.student_state, not the graph channel

    async def ta_teach_finish(self, state: AgentState, config: RunnableConfig):
        """ TA synthesis node — tool-calling agent"""
        sid, uid, tracker, session_context = _resource_ctx(config)
        tracer, chat_id = _tracer_ctx(config)
        ta = self.agents["TA"]
        language = state.get("language", "vn")
        language_instruction = get_language_instruction(language)

        proposal = tracker.get_session(sid).student_state.get("pending_proposal")
        if proposal and proposal.get("auto_apply"):
            tracker.apply_proposal(sid, proposal)

        worker_results = state.get("worker_results", {})
        teach_res = (
            worker_results.get("Teach_Lecture")
            or worker_results.get("Teach_Review")
            or {}
        )

        prompt = prompt_lib.TEACH_PRESENT_PROMPT.format(
            language_instruction=language_instruction,
            teach_res=teach_res
        )

        ## -- Inject prior TA messages for coherence
        ta_context = extract_ta_context(state)
        if ta_context:
            prompt = f"[Prior TA reasoning]:\n{ta_context}\n\n{prompt}"

        start_invoke = time.time()
        message = await self._stream_answer(ta, prompt, config)
        ta_output = TAOutput(summary=self._derive_summary(message), message=message)

        log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
        if log_filename:
            AgentTracer.logging({
                "agent_name": ta.name,
                "node": "TA_Teach_Finish",
                "thought": ta_output.summary,
                "prompt": prompt[:300],
                "output": ta_output.model_dump()
            }, type="info", file_name=log_filename)

        # --- Background: memoize & persist (non-blocking, atomic $push) ---
        async def _bg_teach(ta_out: TAOutput, cid: str):
            if uid and cid:
                tracker.mongodb.push_chat_message(
                    uid, sid, cid,
                    {"role": ta.name, "heading": "Teaching & Evaluating", "message": ta_out.summary}
                )
            await self._save_ta_memo(sid, cid, tracker, ta_out)
            tracker.save_state(sid)

        await _bg_teach(ta_output, chat_id)

        logger.info(f"[SMART_EDU_LOG] Node: TA_Teach_Finish | Time: {time.time() - start_invoke:.4f}s")

        ## -- Auto FE navigation: prefer state ui_action (from teach_lookup), else ta_output.ui_action
        ui_action = state.get("ui_action") or ta_output.ui_action
        teach_ctx = state.get("_teach_context", {})
        if not ui_action and teach_ctx.get("page"):
            ui_action = {"navigate_page": teach_ctx["page"], "document": teach_ctx.get("storage_uri")}

        if tracer and chat_id:
            tracer.log_step(
                chat_id=chat_id,
                node="TA_Teach_Finish",
                prompt=prompt,
                state=tracker.get_student_state(sid),
                tool_result=worker_results,
                output=ta_output.message,
            )

        return {
            "messages": [AIMessage(content=ta_output.message)],
            "pending_proposal": None,
            "ui_action": ui_action,
        }

    async def apply_proposal_node(self, state: AgentState, config: RunnableConfig):
        sid, _uid, tracker, _ctx = _resource_ctx(config)
        tracer, chat_id = _tracer_ctx(config)
        session = tracker.get_session(sid)
        proposal = session.student_state.get("pending_proposal")

        if not proposal:
            msg = "Không có đề xuất nào đang chờ xác nhận."
            ta_output = TAOutput(summary="No pending proposal", message=msg)
            await self._save_ta_memo(sid, chat_id, tracker, ta_output)
            return {"messages": [AIMessage(content=msg)]}

        tracker.apply_proposal(sid, proposal)

        log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
        if log_filename:
            AgentTracer.logging({
                "agent_name": "Apply_Proposal",
                "node": "Apply_Proposal",
                "prompt": "Apply pending proposal",
                "output": proposal if proposal else {}
            }, type="info", file_name=log_filename)

        current = session.student_state.get("current_pos")
        upcoming = session.student_state.get("upcoming_nodes", [])

        msg = f"Đã áp dụng lộ trình mới.\n"
        if current:
            msg += f"Vị trí hiện tại: {current.name}\n"
        if upcoming:
            msg += f"Tiếp theo: {', '.join(n.name for n in upcoming[:3])}"

        ta_output = TAOutput(summary=f"Applied proposal: {current.name if current else 'N/A'}", message=msg)

        if tracer and chat_id:
            tracer.log_step(
                chat_id=chat_id,
                node="Apply_Proposal",
                prompt="",
                state=tracker.get_student_state(sid),
                tool_result=proposal,
                output=msg,
            )

        await self._save_ta_memo(sid, chat_id, tracker, ta_output)
        tracker.save_state(sid)

        return {"messages": [AIMessage(content=ta_output.message)], "pending_proposal": None}

    async def ta_unknown_finish(self, state: AgentState, config: RunnableConfig):
        """ Fallback node for unclassifiable queries — guides student on how to interact. """
        sid, _uid, tracker, _ctx = _resource_ctx(config)
        tracer, chat_id = _tracer_ctx(config)
        ta = self.agents["TA"]
        language = state.get("language", "vn")
        language_instruction = get_language_instruction(language)

        S_state = parse_student_state(tracker.get_student_state(sid))
        history = tracker.get_chat_history(sid, mode="skim")

        prompt = prompt_lib.UNKNOWN_PROMPT.format(
            language_instruction=language_instruction,
            query=state.get("user_query", ""),
            state=S_state,
            history=history,
        )

        start_invoke = time.time()
        message = await self._stream_answer(ta, prompt, config)
        ta_output = TAOutput(summary=self._derive_summary(message), message=message)

        log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
        if log_filename:
            AgentTracer.logging({
                "agent_name": ta.name,
                "node": "TA_Unknown_Finish",
                "thought": ta_output.summary,
                "prompt": prompt[:300],
                "output": ta_output.model_dump()
            }, type="info", file_name=log_filename)

        logger.info(f"[SMART_EDU_LOG] Node: TA_Unknown_Finish | Time: {time.time() - start_invoke:.4f}s")

        if tracer and chat_id:
            tracer.log_step(
                chat_id=chat_id,
                node="TA_Unknown_Finish",
                prompt=prompt,
                state=tracker.get_student_state(sid),
                output=ta_output.message,
            )

        # --- Background: memoize & persist (non-blocking) ---
        await self._bg_save(sid, chat_id, tracker, ta_output)

        return {"messages": [AIMessage(content=ta_output.message)], "pending_proposal": None}



    @staticmethod
    def _derive_summary(message: str) -> str:
        """memo heading, was LLM-made, now first non-empty line"""
        for line in message.splitlines():
            s = line.strip().lstrip("#").strip()
            if s:
                return s[:120]
        return message[:120]

    async def _stream_answer(self, ta, prompt: str, config: RunnableConfig) -> str:
        """raw model stream, no tool loop (finish prompts self-contained); sys prompt bypassed by agent so prepend"""
        emit = config.get("configurable", {}).get("emit")
        msgs = [("system", ta.system_prompt_text), ("user", prompt)]
        parts = []
        async for chunk in ta.model.astream(msgs, config=config):
            text = getattr(chunk, "content", "") or ""
            if text:
                parts.append(text)
                if emit:
                    await emit({"type": "token", "text": text})
        return "".join(parts)

    @staticmethod
    async def _save_ta_memo(session_id: str, chat_id: str, tracker, ta_output: TAOutput):
        """Standard memo save for all finish nodes — uses TAOutput."""
        session = tracker.get_session(session_id)
        session.student_state["summary"] = ta_output.summary
        
        # Append to DB directly
        msg = {
            "role": "TA",
            "heading": ta_output.summary,
            "message": ta_output.message,
        }
        tracker.mongodb.push_chat_message(session.student_id, session_id, chat_id, msg)
        
        # Optional: append to in-memory memo if needed, but not required if get_chat_history uses DB.
        # Since get_chat_history uses self.session.chats in memo, we should append in memory too.
        for chat in session.memo.session.chats:
            if chat.id == chat_id:
                from student.memo import ChatMessage
                import datetime
                chat.messages.append(ChatMessage(**msg, timestamp=datetime.datetime.utcnow().isoformat()))
                break

    async def _bg_save(self, session_id: str, chat_id: str, tracker, ta_output: TAOutput):
        """Background-safe save: memo + student state persistence."""
        await self._save_ta_memo(session_id, chat_id, tracker, ta_output)
        tracker.save_state(session_id)

    async def execute(
        self,
        initial_state: AgentState,
        session_id: str,
        student_id: str = "",
        student_tracker=None,
        session_context=None,
        tracer=None,
        chat_id: str = "",
        callbacks: Optional[List[Any]] = None,
        update_callback=None,
        emit=None,
        retrieve_param=None,
    ):

        log_f = f"wf/wf_v0_{datetime.now().strftime('%H%M%S')}_{datetime.now().strftime('%d%m')}.json"
        lang = initial_state.get("language", "vn")


        run_config: RunnableConfig = {
            "configurable": {
                "session_id": session_id,
                "student_id": student_id,
                "student_tracker": student_tracker,
                "mongo_db": getattr(student_tracker, "mongodb", None),
                "session_context": session_context,
                "tracer": tracer,
                "chat_id": chat_id,
                "teach_tools": self.teach_tools,
                "log_filename": log_f,
                "emit": emit,  ## finish nodes pull this to stream tokens
                "retrieve_param": retrieve_param,
            },
            "callbacks": callbacks or [],
            "recursion_limit": 50,
        }
        try:
            final_state = dict(initial_state)
            async for chunk in self.app.astream(initial_state, config=run_config, stream_mode="updates"):
                for node_name, state_update in chunk.items():
                    _loggable_keys = [k for k in state_update if k not in ("messages", "worker_results")]
                    if _loggable_keys:
                        logger.debug("[astream] Node: %s | keys: %s", node_name, _loggable_keys)
                    else:
                        logger.debug("[astream] Node: %s done", node_name)
                    if update_callback:
                        await update_callback(node_name, state_update)
                    if emit:
                        await emit({"type": "step", "node": node_name, "label": _humanize(node_name, lang)})
                    # Manually merge the state_update into final_state
                    for key, val in state_update.items():
                        if key == "messages":
                            if "messages" not in final_state:
                                final_state["messages"] = []
                            final_state["messages"].extend(val)
                        elif isinstance(val, dict) and isinstance(final_state.get(key), dict):
                            final_state[key].update(val)
                        else:
                            final_state[key] = val
            return final_state
        except Exception as e:
            logger.exception(
                "SmartEdu execution FAILED | session=%s | input=%r",
                session_id,
                initial_state.get("user_query", "")[:120],
            )
            if tracer and chat_id:
                tracer.log_step(
                    chat_id=chat_id,
                    node="__error__",
                    prompt="",
                    state={},
                    output=str(e),
                )
            # Ensure return dict has the expected keys for TAModule.run
            return {
                **final_state, 
                "status_flag": "FAIL", 
                "_error": str(e),
                "messages": final_state.get("messages", [])
            }