import asyncio
import json
import re
import time
import logging
from datetime import datetime
from typing import List, Optional, Any, Dict
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

import TA.helper.prompt as prompt_lib
from TA.helper.few_shot import format_few_shot, get_language_instruction
from TA.helper.model_call import bounded_ainvoke, is_transient, ta_ainvoke, text_content
from TA.retrieval.policy import BENCHMARK_ANSWER_PROMPT
from TA.helper.schema import BenchmarkAnswer, RouterDecision
from core.schema.wf_state import AgentState, ConceptNode, TAOutput

from TA.workflow.retrieve import build_retrieve_wf
from core.schema.retrieval import RetrievalCaseKind, RetrievalRunContext
from TA.workflow.roadmap import build_roadmap_wf
from TA.workflow.teach import build_teach_wf

from TA.helper.utils import parse_student_state
from TA.helper.context import build_ui_citations, extract_ta_context
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
    def __init__(self, agents, retrieve_res: Dict = None):
        self.agents = agents
        self.retrieve_res = retrieve_res or {}
        self.app = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState, context_schema=RetrievalRunContext)

        builder.add_node("TA_Router", self.ta_router_node)

        retrieve_wf = build_retrieve_wf(agents=self.agents, resources=self.retrieve_res)
        builder.add_node("WF_Retrieve", retrieve_wf)
        builder.add_node("WF_Roadmap", build_roadmap_wf(agents=self.agents))
        builder.add_node("WF_Teach", build_teach_wf(
            agents=self.agents, retrieve_wf=retrieve_wf,
            graph_db=self.retrieve_res.get("graph_db"),
        ))

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

    @staticmethod
    def _bind_generation(model, temperature: float, max_tokens: int):
        if model.__class__.__module__.startswith("langchain_ollama"):
            return model.bind(
                options={"temperature": temperature, "num_predict": max_tokens},
                reasoning=False,
            )
        if model.__class__.__module__.startswith("langchain_openrouter"):
            return model.bind(temperature=temperature, max_tokens=max_tokens)
        return model.bind(temperature=temperature, max_output_tokens=max_tokens)

    async def ta_router_node(
        self,
        state: AgentState,
        config: RunnableConfig,
        runtime: Runtime[RetrievalRunContext],
    ):
        """ No-tool node: direct structured output with RouterDecision"""
        forced = runtime.context.case.forced_route
        if forced is not None:
            tracer, chat_id = _tracer_ctx(config)
            if tracer and chat_id:
                tracer.log_step(
                    chat_id=chat_id,
                    node="TA_Router",
                    tool_result={"forced": forced.value},
                    output=forced.value,
                )
            return {"intent": forced.value}

        sid, _uid, tracker, session_context = _resource_ctx(config)
        tracer, chat_id = _tracer_ctx(config)
        ta = self.agents["TA"]
        S_state = parse_student_state(tracker.get_student_state(sid))
        history = tracker.get_chat_history(
            sid,
            mode="skim",
            recent_turns=4,
            exclude_chat_id=chat_id,
            max_chars=4_000,
        )
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
        llm = self._bind_generation(ta.model, 0, 1024)
        res = await ta_ainvoke(llm, [("user", prompt)], config)
        ## -- Take the LAST allowed token: prompt reasons first, then emits the word.
        raw = text_content(getattr(res, "content", res)).lower()
        matches = re.findall(r"\b(retrieve|roadmap|teaching|confirm|unknown)\b", raw)
        intent = matches[-1] if matches else ""
        if not intent:
            query = str(state.get("user_query", "")).casefold()
            lesson_markers = (
                "bài giảng", "bài học tương tác", "câu hỏi ôn tập",
                "interactive lesson", "full lesson", "step-by-step lesson",
            )
            intent = "teaching" if any(marker in query for marker in lesson_markers) else "unknown"
        logger.info(f"[SMART_EDU_LOG] Node: TA_Router | Time: {time.time() - start_invoke:.4f}s |")
        logger.info(f"[SMART_EDU_LOG] Intent: {intent} | Response: {res}")
        if not matches:
            logger.warning(
                f"[ta_router_node] No intent token in LLM response; using {intent!r} fallback. "
                f"query={state.get('user_query', '')!r} raw={getattr(res, 'content', res)!r}"
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

    async def ta_retrieve_finish(
        self,
        state: AgentState,
        config: RunnableConfig,
        runtime: Runtime[RetrievalRunContext],
    ):
        """ TA synthesis node — tool-calling agent reads context then synthesizes"""
        sid, _uid, tracker, session_context = _resource_ctx(config)
        tracer, chat_id = _tracer_ctx(config)
        ta = self.agents["TA"]
        language = state.get("language", "vn")
        language_instruction = get_language_instruction(language)

        proposal = tracker.get_session(sid).student_state.get("pending_proposal")
        if proposal and proposal.get("auto_apply"):
            tracker.apply_proposal(sid, proposal)

        results = state.get("worker_results", {})
        refine_prompt = prompt_lib.RETRIEVE_REFINE_PROMPT.format(
            language_instruction=language_instruction
        )
        prompt = f"{refine_prompt}\nData: {results}"

        ## -- Inject prior TA messages for coherence
        ta_context = extract_ta_context(state)
        if ta_context:
            prompt = f"[Prior TA reasoning]:\n{ta_context}\n\n{prompt}"

        start_invoke = time.time()
        if runtime.context.case.kind is RetrievalCaseKind.BENCHMARK:
            message = await self._benchmark_answer(ta, state, results, config, runtime)
        else:
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

        await self._bg_save(sid, chat_id, tracker, ta_output)

        logger.info(f"[SMART_EDU_LOG] Node: TA_Retrieve_Finish | Time: {time.time() - start_invoke:.4f}s")

        if tracer and chat_id:
            tracer.log_step(
                chat_id=chat_id,
                node="TA_Retrieve_Finish",
                prompt=prompt,
                state=tracker.get_student_state(sid),
                tool_result=results,
                output=ta_output.message,
                latency_ms=(time.time() - start_invoke) * 1000,
            )

        citations = await build_ui_citations(results.get("RAG", {}), self.retrieve_res.get("graph_db"))
        return {
            "messages": [AIMessage(content=ta_output.message)],
            "pending_proposal": None,
            "ui_action": {"citations": citations} if citations else None,
        }

    async def ta_roadmap_finish(self, state: AgentState, config: RunnableConfig):
        """ TA synthesis node — tool-calling agent"""
        sid, _uid, tracker, session_context = _resource_ctx(config)
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

        await self._bg_save(sid, chat_id, tracker, ta_output)

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
        sid, _uid, tracker, session_context = _resource_ctx(config)
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

        start_invoke = time.time()
        if not teach_res:  ## str({}) is truthy "{}", checked pre-stringify so an empty fallback still raises
            raise RuntimeError("Teach finished without a lecture")
        message = str(teach_res).strip()
        ta_output = TAOutput(summary=self._derive_summary(message), message=message)

        log_filename = config.get("configurable", {}).get("log_filename") or os.getenv("TEST_LOG_FILENAME")
        if log_filename:
            AgentTracer.logging({
                "agent_name": ta.name,
                "node": "TA_Teach_Finish",
                "thought": ta_output.summary,
                "prompt": state.get("user_query", "")[:300],
                "output": ta_output.model_dump()
            }, type="info", file_name=log_filename)

        await self._bg_save(sid, chat_id, tracker, ta_output)

        logger.info(f"[SMART_EDU_LOG] Node: TA_Teach_Finish | Time: {time.time() - start_invoke:.4f}s")

        ui_action = state.get("ui_action") or ta_output.ui_action

        if tracer and chat_id:
            tracer.log_step(
                chat_id=chat_id,
                node="TA_Teach_Finish",
                prompt=state.get("user_query", ""),
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
        await asyncio.to_thread(tracker.save_state, sid)

        return {
            "messages": [AIMessage(content=ta_output.message)],
            "pending_proposal": None,
            "status_flag": "SUCCESS",
        }

    async def ta_unknown_finish(self, state: AgentState, config: RunnableConfig):
        """ Fallback node for unclassifiable queries — guides student on how to interact. """
        sid, _uid, tracker, _ctx = _resource_ctx(config)
        tracer, chat_id = _tracer_ctx(config)
        ta = self.agents["TA"]
        language = state.get("language", "vn")
        language_instruction = get_language_instruction(language)

        S_state = parse_student_state(tracker.get_student_state(sid))
        history = tracker.get_chat_history(
            sid,
            mode="skim",
            recent_turns=4,
            exclude_chat_id=chat_id,
            max_chars=4_000,
        )

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

        # --- Memoize & persist (offloaded off the event loop) ---
        await self._bg_save(sid, chat_id, tracker, ta_output)

        return {
            "messages": [AIMessage(content=ta_output.message)],
            "pending_proposal": None,
            "status_flag": "SUCCESS",
        }



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
        tracer, chat_id = _tracer_ctx(config)
        msgs = [("system", ta.system_prompt_text), ("user", prompt)]
        attempt = 0
        while True:
            parts = []
            try:
                async for chunk in ta.model.astream(msgs, config=config):
                    text = text_content(getattr(chunk, "content", chunk))
                    if text:
                        if tracer and chat_id:
                            tracer.mark_first_token(chat_id)
                        parts.append(text)
                        if emit:
                            await emit({"type": "token", "text": text})
                return "".join(parts)
            except Exception as e:
                ## retry only before any token reached the client, else the stream would repeat
                if parts or attempt == 2 or not is_transient(e):
                    raise
                await asyncio.sleep(2**attempt)
                attempt += 1

    async def _benchmark_answer(
        self,
        ta,
        state: AgentState,
        results: Dict[str, Any],
        config: RunnableConfig,
        runtime: Runtime[RetrievalRunContext],
    ) -> str:
        policy = runtime.context.policy
        answer_model = self._benchmark_answer_model(ta, runtime.context)
        model_name = getattr(answer_model, "model", None) or getattr(answer_model, "model_name", None)
        if model_name != policy.answer_model_name:
            raise RuntimeError(
                f"benchmark answer model mismatch: expected {policy.answer_model_name}, got {model_name or 'unknown'}"
            )
        answerer = self._bind_generation(
            answer_model, policy.answer_temperature, 256
        ).with_structured_output(
            BenchmarkAnswer,
            method="json_mode",
            include_raw=True,
        )
        prompt = self._benchmark_answer_prompt(
            state.get("user_query", ""),
            results,
            policy.answer_prompt or BENCHMARK_ANSWER_PROMPT,
        )
        result, _ = await bounded_ainvoke(
            answerer,
            [("user", prompt)],
            config=config,
            timeout_s=policy.answer_timeout_s,
            retries=policy.answer_transport_retries,
            request_key=policy.answer_model_profile,
        )
        return self._benchmark_answer_text(result)

    def _benchmark_answer_model(self, ta, context: RetrievalRunContext):
        if context.policy.answer_model_profile == "retrieval_answerer":
            answerer = self.agents.get("RETRIEVAL_ANSWERER")
            if answerer is None:
                raise RuntimeError("dedicated retrieval answerer unavailable")
            return answerer
        return ta.model

    @staticmethod
    def _benchmark_answer_prompt(
        question: str,
        results: Dict[str, Any],
        template: str = BENCHMARK_ANSWER_PROMPT,
    ) -> str:
        rag = results.get("RAG", {})
        return template.format(
            question=question,
            synthesis=rag.get("thought", ""),
            evidence=rag.get("content", ""),
        )

    @staticmethod
    def _benchmark_answer_text(result: Dict[str, Any]) -> str:
        parsed = result.get("parsed")
        if isinstance(parsed, BenchmarkAnswer):
            return parsed.answer.strip()
        raw = text_content(getattr(result.get("raw"), "content", "unknown")) or "unknown"
        try:
            value = json.loads(raw).get("answer", raw)
        except (TypeError, json.JSONDecodeError):
            value = raw
        return BenchmarkAnswer(answer=str(value).strip() or "unknown").answer

    @staticmethod
    async def _save_ta_memo(session_id: str, chat_id: str, tracker, ta_output: TAOutput):
        """Standard memo save for all finish nodes — uses TAOutput."""
        session = await asyncio.to_thread(tracker.get_session, session_id)
        session.student_state["summary"] = ta_output.summary

        # Append to DB directly
        msg = {
            "role": "TA",
            "heading": ta_output.summary,
            "message": ta_output.message,
        }
        ## sync pymongo, to_thread or it stalls every concurrent SSE stream
        await asyncio.to_thread(tracker.mongodb.push_chat_message, session.student_id, session_id, chat_id, msg)

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
        await asyncio.to_thread(tracker.save_state, session_id)

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
        retrieval_context: RetrievalRunContext | None = None,
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
                "log_filename": log_f,
                "emit": emit,  ## finish nodes pull this to stream tokens
            },
            "callbacks": callbacks or [],
            "recursion_limit": 50,
        }
        try:
            final_state = dict(initial_state)
            async for chunk in self.app.astream(
                initial_state,
                config=run_config,
                context=retrieval_context,
                stream_mode="updates",
            ):
                for node_name, state_update in chunk.items():
                    if tracer and chat_id:
                        tracer.ensure_node_step(chat_id, node_name)
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
