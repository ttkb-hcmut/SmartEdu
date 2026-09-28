from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from datetime import datetime
import time
import os
from uuid_v7.base import uuid7

def generate_uuidv7() -> str:
    return str(uuid7())

class ChatMessage(BaseModel):
    role: str
    heading: str
    message: str
    timestamp: str

class Chat(BaseModel):
    id: str = Field(default_factory=generate_uuidv7)
    invoke: str = ""
    messages: List[ChatMessage] = Field(default_factory=list)

class Session(BaseModel):
    id: str = Field(default_factory=generate_uuidv7)
    name: str = "New Session"
    chats: List[Chat] = Field(default_factory=list)

class Memo:
    def __init__(self, session_id: str, session_data: Optional[Dict] = None, save_callback: Optional[Callable] = None):
        self.session_id = session_id
        self.save_callback = save_callback
        if session_data:
            self.session = Session(**session_data)
        else:
            self.session = Session(id=session_id)

    async def save(self, msg_dict: Dict[str, Any]):
        if "timestamp" not in msg_dict:
            msg_dict["timestamp"] = datetime.utcnow().isoformat()
        chat_msg = ChatMessage(**msg_dict)
        if not self.session.chats:
            self.session.chats.append(Chat(messages=[chat_msg]))
        else:
            self.session.chats[-1].messages.append(chat_msg)
        if self.save_callback:
            res = self.save_callback(self.session_id, self.session)
            if hasattr(res, "__await__"):
                await res

    ## recent_turns set -> last N turns full, older turns skim. bounds tokens.
    def get_formatted_history(
        self,
        mode: str = "full",
        recent_turns: Optional[int] = None,
        exclude_chat_id: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        if not self.session.chats:
            return "No prior context."

        chats = [chat for chat in self.session.chats if chat.id != exclude_chat_id]
        if not chats:
            return "No prior context."
        cutoff = (len(chats) - recent_turns) if recent_turns is not None else 0

        formatted_lines = []
        for i, chat in enumerate(chats):
            chat_mode = "skim" if (recent_turns is not None and i < cutoff) else mode
            for msg in chat.messages:
                if msg.role == "student":
                    formatted_lines.append(f"Student: {msg.message}")
                elif msg.role == "TA":
                    if chat_mode == "skim":
                        formatted_lines.append(f"TA: {msg.heading}")
                    else:
                        formatted_lines.append(f"TA: {msg.message}")

        history = "\n".join(formatted_lines)
        if max_chars is None or len(history) <= max_chars:
            return history
        if max_chars <= 0:
            return ""

        kept = []
        remaining = max_chars
        for line in reversed(formatted_lines):
            separator = 1 if kept else 0
            if len(line) + separator > remaining:
                if not kept:
                    kept.append(line[:remaining])
                break
            kept.append(line)
            remaining -= len(line) + separator
        return "\n".join(reversed(kept))
