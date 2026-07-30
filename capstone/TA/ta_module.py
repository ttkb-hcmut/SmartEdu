from TA.agent.injector import AgentInjector
from TA.tools.factory import ToolFactory
from TA.tools.student.context_tools import _current_session_context, _current_tracker
from TA.workflow.smart_edu import SmartEdu
from TA.tracing import AgentTracer
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from core.schema.wf_state import AgentState, TAOutput
from core.config import retrieve_param as _default_rp
from core.schema.retrieval import RetrievalCase, RetrievalCodeState
from TA.retrieval.policy import resolve_retrieval_context
from student.Student_Tracker import Student_Tracker
from student.memo import Chat, ChatMessage
from collections import OrderedDict
from datetime import datetime, timezone
import asyncio

RECENT_TURNS = 4  # last N turns full, older skim


class TAModule:
    def __init__(self, llm, graph_db, milvus_db, embedder, minio, student_tracker: Student_Tracker, config):
        self.student_tracker = student_tracker

        self.tools_factory = ToolFactory(graph_db, milvus_db, embedder, minio, tracker=student_tracker)

        self.agents = AgentInjector.initialize_all_agents(
            llm_engine=llm, tools_factory=self.tools_factory, config=config.AGENTS
        )
        self.engine: SmartEdu = SmartEdu(
            agents=self.agents,
            teach_tools=self.tools_factory.get_teach_lookup_tools(),
            retrieve_res={"graph_db": graph_db, "milvus_db": milvus_db, "embedder": embedder},
        )

        # tracer per session, LRU bound -> no unbounded leak
        self._tracers: "OrderedDict[str, AgentTracer]" = OrderedDict()
        self._max_tracers = 100

    def _get_tracer(self, session_id: str) -> AgentTracer:
        tracer = self._tracers.get(session_id)
        if tracer is None:
            tracer = AgentTracer(session_id=session_id)
            self._tracers[session_id] = tracer
            while len(self._tracers) > self._max_tracers:
                self._tracers.popitem(last=False)  # evict oldest
        else:
            self._tracers.move_to_end(session_id)
        return tracer

    async def run(self,
                    user_input: str,
                    session_id: str,
                    update_callback=None,
                    language: str = "vn",
                    chat_id: str = "",
                    emit=None,
                    retrieve_param=None,
                    retrieval_case: RetrievalCase | None = None,
                    code_state: RetrievalCodeState | None = None):

        params = retrieve_param or _default_rp
        retrieval_context = resolve_retrieval_context(
            params,
            retrieval_case,
            code_state,
        )

        session = self.student_tracker.get_session(session_id)
        tracer = self._get_tracer(session_id)

        ## reuse route chat_id -> trace/memo/mongo share one id
        chat_id = tracer.begin_chat(query=user_input, chat_id=chat_id or None)

        # Bug 3 fix: use session.context directly (loaded from MongoDB on session init)
        student_id = self.student_tracker._resolve(session_id)
        session_context = session.context

        history_str = session.memo.get_formatted_history(recent_turns=RECENT_TURNS)
        student_state = session.student_state

        ## mirror student msg into in-mem memo (mongo gets it via route.create_chat); same chat_id so TA reply appends here
        memo_chats = session.memo.session.chats
        if not any(c.id == chat_id for c in memo_chats):
            memo_chats.append(Chat(
                id=chat_id,
                invoke=user_input,
                messages=[ChatMessage(
                    role="student", heading="Student Query",
                    message=user_input, timestamp=datetime.now(timezone.utc).isoformat(),
                )],
            ))

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=f"Chat History Context:\n{history_str}"),
                HumanMessage(content=user_input)
            ],
            "student_state": student_state,
            "worker_results": {},
            "intent": user_input,
            "thought": "",
            "status_flag": "On-going",
            "user_query": user_input,
            "language": language,
        }

        callbacks = []
        if tracer.langfuse_handler:
            callbacks.append(tracer.langfuse_handler)

        # Bug 2 fix: inject session_context via ContextVar (async-safe, picked up by context tools)
        _ctx_token = _current_session_context.set(session_context)
        _tracker_token = _current_tracker.set(self.student_tracker)
        response = ""
        ui_action = None
        intent = ""
        status = "FAIL"
        errors = []
        try:
            final_state = await self.engine.execute(
                initial_state=initial_state,
                session_id=session_id,
                student_id=student_id,
                student_tracker=self.student_tracker,
                session_context=session_context,
                chat_id=chat_id,
                tracer=tracer,
                callbacks=callbacks,
                update_callback=update_callback,
                emit=emit,
                retrieval_context=retrieval_context,
            )
            status = final_state.get("status_flag", "SUCCESS")
            intent = final_state.get("intent", "")
            ui_action = final_state.get("ui_action")
            if status == "FAIL":
                errors.append(final_state.get("_error", "Unknown TA workflow error."))
            else:
                last_msg = final_state["messages"][-1]
                response = (
                    last_msg.content
                    if isinstance(last_msg, BaseMessage)
                    else last_msg.get("content", "")
                )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            _current_session_context.reset(_ctx_token)
            _current_tracker.reset(_tracker_token)
            if errors:
                status = "FAIL"
                response = f"System error encountered: {errors[0]}"
                ui_action = None
            try:
                await tracer.end_chat(
                    chat_id=chat_id,
                    final_output=response,
                    intent=intent,
                    status=status,
                    errors=errors,
                    retrieval_context=retrieval_context,
                )
            except Exception as exc:
                status = "FAIL"
                errors.append(f"trace finalization failed: {type(exc).__name__}: {exc}")

        return {
            "message": response,
            "ui_action": ui_action,
            "status": status,
            "errors": errors,
        }
