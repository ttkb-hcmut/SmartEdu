"""
AgentTracer — per-session in-memory tracer with async flush to JSON.

Usage:
    tracer = AgentTracer(session_id="student_123")
    chat_id = tracer.begin_chat(query="What is ML?")
    tracer.log_step(chat_id, node="TA_Router", prompt="...", state={}, output="retrieve")
    tracer.log_step(chat_id, node="Fusion", chunks=[...], latency_ms=12.5)
    await tracer.end_chat(chat_id, final_output="...", intent="retrieve", preset="FULL")

Langfuse integration:
    Activate by default if found env vars LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY. More docs on this shortly
    Otherwise silent skip (Consider it a disabled state, NOT an error).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal, TYPE_CHECKING

from TA.tracing.schema import ChatTrace, StepTrace, TraceSession
from TA.tracing.writer import TraceWriter
from core.config import langfuse_config

if TYPE_CHECKING:
    from core.schema.retrieval import RetrievalRunContext

logger = logging.getLogger(__name__)


def _serialize_student_state(state: Any) -> Dict[str, Any]:
    """
    Serialize and truncate StudentState (TypedDict) to plain dict for JSON storage.
    """
    if not state:
        return {}
    if not isinstance(state, dict):
        try:
            state = dict(state)
        except Exception:
            return {"raw": str(state)}

    result: Dict[str, Any] = {}

    # current_pos (ConceptNode)
    cp = state.get("current_pos")
    if cp is not None:
        try:
            result["current_pos"] = cp.model_dump() if hasattr(cp, "model_dump") else dict(cp)
        except Exception:
            result["current_pos"] = str(cp)

    # previous_nodes (List[ConceptNode]) — chỉ lấy 3 gần nhất
    prev = state.get("previous_nodes", [])
    result["previous_nodes"] = [
        (n.name if hasattr(n, "name") else str(n)) for n in list(prev)[:3]
    ]

    # upcoming_nodes
    upcoming = state.get("upcoming_nodes", [])
    result["upcoming_nodes"] = [
        (n.name if hasattr(n, "name") else str(n)) for n in list(upcoming)[:3]
    ]

    # mastery_map — chỉ giữ entries có value > 0
    mastery = state.get("mastery_map", {})
    if mastery:
        result["mastery_map"] = {k: v for k, v in mastery.items() if v and v > 0}

    # summary
    if state.get("summary"):
        result["summary"] = str(state["summary"])[:200]

    return result


class AgentTracer:
    """
    Per-session tracer. Mỗi TAModule.run() = 1 chat turn.
    Thread-safe: dùng in-memory dict cho buffer, flush async khi end_chat.
    """

    def __init__(self, session_id: str = "default", writer: Optional[TraceWriter] = None):
        self.session_id = session_id
        self._session = TraceSession(session_id=session_id)
        self._writer = writer or TraceWriter()
        self._active_chats: Dict[str, ChatTrace] = {}  # chat_id → ChatTrace buffer

        # Langfuse — optional, lazy init
        self._langfuse_handler = None
        self._init_langfuse()

    # ── Langfuse (optional) ───────────────────────────────────────────────

    def _init_langfuse(self):
        if not langfuse_config.enable:
            logger.debug("[AgentTracer] Langfuse disabled in config.")
            return

        secret = langfuse_config.key
        public = langfuse_config.public_key
        host = langfuse_config.host

        if not secret or not public:
            logger.warning("[AgentTracer] Langfuse enabled but keys missing!")
            return

        try:
            from langfuse.langchain import CallbackHandler
            self._langfuse_handler = CallbackHandler(
                secret_key=secret,
                public_key=public,
                host=host,
                session_id=self.session_id,
            )
            logger.info(f"[AgentTracer] Langfuse active → host={host}")
        except ImportError:
            logger.warning("[AgentTracer] langfuse not installed. Run: pip install langfuse")
        except Exception as e:
            logger.warning(f"[AgentTracer] Langfuse init failed: {e}")

    @property
    def langfuse_handler(self):
        return self._langfuse_handler

    # ── Chat lifecycle ────────────────────────────────────────────────────

    ## chat_id passed -> reuse caller id (align trace/memo/mongo). else datetime.
    def begin_chat(self, query: str, chat_id: Optional[str] = None) -> str:
        chat_id = chat_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        chat = ChatTrace(chat_id=chat_id, query=query)
        self._active_chats[chat_id] = chat
        logger.debug(f"[AgentTracer] begin_chat: session={self.session_id} chat_id={chat_id}")
        return chat_id

    def log_step(
        self,
        chat_id: str,
        node: str,
        prompt: str = "",
        state: Any = None,
        tool_result: Optional[Dict[str, Any]] = None,
        output: str = "",
        chunks: Optional[List[Dict[str, Any]]] = None,
        latency_ms: float = 0.0,
        tokens: Optional[Dict[str, int]] = None,
    ):
        """
        Append one reasoning step to the buffer.
        Thread-safe (in-memory append only, no I/O).
        """
        chat = self._active_chats.get(chat_id)
        if chat is None:
            logger.warning(f"[AgentTracer] log_step: unknown chat_id={chat_id}")
            return

        step = StepTrace(
            node=node,
            prompt=prompt[:3000] if prompt else "",  # cap, file size
            state=_serialize_student_state(state),
            tool_result=tool_result or {},
            output=str(output)[:2000] if output else "",
            chunks=[{**c, "text": str(c.get("text", ""))[:1000]} for c in (chunks or [])],
            latency_ms=latency_ms,
            tokens=tokens or {},
        )
        chat.agent.append(step)

    async def end_chat(
        self,
        chat_id: str,
        final_output: str = "",
        intent: str = "",
        status: str = "SUCCESS",
        retrieve_flags: Optional[Dict[str, bool]] = None,
        preset: str = "",
        errors: Optional[List[str]] = None,
        retrieval_context: Optional["RetrievalRunContext"] = None,
    ):
        """
        Close the chat turn: flush buffer to JSON file.
        Called after SmartEdu.execute() completes.
        """
        chat = self._active_chats.pop(chat_id, None)
        if chat is None:
            logger.warning(f"[AgentTracer] end_chat: unknown chat_id={chat_id}")
            return

        chat.final_output = str(final_output)[:2000]
        chat.intent = intent
        chat.status = status
        chat.retrieve_flags = retrieve_flags or {}
        chat.preset = preset
        chat.errors = errors or []
        if retrieval_context is not None:
            context = retrieval_context
            chat.retrieve_flags = {
                tool.value: True for tool in context.policy.allowed_tools
            }
            chat.preset = context.preset.name
            chat.policy_id = context.policy.id.value
            chat.policy_digest = context.policy.digest
            chat.harness_id = context.harness.id.value
            chat.run_id = context.case.run_id
            chat.question_id = context.case.question_id
            chat.warmup = context.case.kind.value == "warmup"
            chat.model_profile = context.policy.model_profile
            chat.model = context.policy.model_name
            chat.temperature = context.policy.temperature
            chat.code_revision = context.code.revision
            chat.dirty = context.code.dirty

        self._session.chat.append(chat)
        path = await self._writer.write_async(self._session)
        logger.info(f"[AgentTracer] Trace saved → {path} (turn {len(self._session.chat)})")

        # Flush Langfuse telemetry data immediately if active
        if self._langfuse_handler:
            try:
                self._langfuse_handler.langfuse.flush()
            except Exception as e:
                logger.warning(f"[AgentTracer] Langfuse flush failed: {e}")

        return path

    @staticmethod
    def logging(data: Any, type: Literal["err", "info"] = "info", file_name: Optional[str] = None):
        import json
        if not file_name:
            return

        log_dir = "test/TA/logs"
        if not file_name.endswith((".log", ".logs", ".json")):
            file_name = f"{file_name}.json"
        filepath = os.path.join(log_dir, file_name)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if isinstance(data, str):
            if len(data) > 400:
                processed_s = f"{data[:200]}......{data[-200:]}"
            else:
                processed_s = data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prefix = f"[{timestamp}] [{type.upper()}] "
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"{prefix}{processed_s}\n")
        elif isinstance(data, dict):
            existing_data = []
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            existing_data = json.loads(content)
                            if not isinstance(existing_data, list):
                                existing_data = [existing_data]
                except Exception:
                    existing_data = []
            
            existing_data.append(data)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
