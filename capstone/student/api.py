"""
student/api.py
──────────────
Student-facing REST endpoints:
  POST /register         → tạo account mới
  POST /login            → trả về JWT Access Token
  POST /session/start    → (Authenticated) Khởi tạo/lấy chat session, trả về session_id
  DELETE /session/end    → (Authenticated) Hủy chat session hiện tại khỏi memory

OAuth-like Standard flow:
  Client ──── POST /login ────► Backend ──── JWT(student_id) ────► Client
  Client ──── POST /session/start [Bearer JWT] ─► Backend ──── session_id ──► Client
  Client ──── POST /ta/chat [session_id] ────────────────────────────────────► TA
"""

import asyncio
import uuid
from fastapi import APIRouter, Request, HTTPException, Depends, Query, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

import os
from student.auth import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    get_current_student,
    User,
)

router = APIRouter(tags=["Student Auth"])

# admin ids from env, comma-separated; matching new accounts become admin
_ADMIN_IDS = {x.strip() for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip()}


################## Request / Response schemas
class RegisterRequest(BaseModel):
    student_id: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str          # proxy stores this in an httpOnly cookie
    is_admin: bool = False
    token_type: str = "bearer"

class RefreshRequest(BaseModel):
    refresh_token: str          # proxy forwards the cookie value here

class AccessResponse(BaseModel):
    access_token: str
    is_admin: bool = False
    token_type: str = "bearer"

class ProfileResponse(BaseModel):
    student_id: str
    is_admin: bool = False
    language: str = "vn"

class LanguageUpdate(BaseModel):
    language: str               # "vn" | "eng"

class SessionResponse(BaseModel):
    session_id: str          # UUID, opaque to the caller

class ResetRequest(BaseModel):
    student_id: str


################## Endpoints
@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(request: Request, payload: RegisterRequest):
    """
    Create a new student account.
    Stores hashed password in SQL; creates Mongo document for state.
    """
    tracker = request.app.state.student_tracker
    sql = tracker.sqldb
    mongo = tracker.mongodb

    if sql.get_student_by_id(payload.student_id) is not None:
        raise HTTPException(status_code=409, detail="Student ID already exists.")

    # admin if id is in the env allow-list
    is_admin = payload.student_id in _ADMIN_IDS
    sql.create_student(
        student_id=payload.student_id,
        username=payload.student_id,
        password=payload.password,
        is_admin=is_admin,
    )
    mongo.create_student(payload.student_id)
    return {"detail": "Registered successfully."}


@router.post("/login", response_model=TokenResponse)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login.
    Uses SQL_DB's built-in SHA256 authenticate method.
    Returns a SHORT access token (browser memory) + LONG refresh token (httpOnly cookie).
    """
    tracker = request.app.state.student_tracker
    sql = tracker.sqldb

    if not sql.authenticate(username=form_data.username, password=form_data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    # bake admin flag into access token claims
    is_admin = sql.is_admin(form_data.username)
    access = create_access_token(form_data.username, extra={"is_admin": is_admin})
    refresh = create_refresh_token(form_data.username)
    return TokenResponse(access_token=access, refresh_token=refresh, is_admin=is_admin)


@router.post("/refresh", response_model=AccessResponse)
async def refresh(request: Request, payload: RefreshRequest):
    """
    Trade a valid refresh token (from the httpOnly cookie, forwarded by the FE proxy)
    for a fresh short-lived access token. No IP binding.
    """
    # validate refresh token, learn student id
    student_id = decode_refresh_token(payload.refresh_token)

    tracker = request.app.state.student_tracker
    sql = tracker.sqldb
    # re-read admin flag fresh in case it changed
    is_admin = sql.is_admin(student_id)
    access = create_access_token(student_id, extra={"is_admin": is_admin})
    return AccessResponse(access_token=access, is_admin=is_admin)


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(
    request: Request,
    current_student: User = Depends(get_current_student),
):
    """Return account info + saved language preference for the logged-in student."""
    tracker = request.app.state.student_tracker
    mongo = tracker.mongodb
    # read saved language
    language = mongo.get_language(current_student.id)
    return ProfileResponse(
        student_id=current_student.id,
        is_admin=current_student.is_admin,
        language=language,
    )


@router.patch("/profile", response_model=ProfileResponse)
async def update_profile(
    payload: LanguageUpdate,
    request: Request,
    current_student: User = Depends(get_current_student),
):
    """Update the student's language preference (persisted in MongoDB)."""
    if payload.language not in ("vn", "eng"):
        raise HTTPException(status_code=400, detail="language must be 'vn' or 'eng'.")
    tracker = request.app.state.student_tracker
    mongo = tracker.mongodb
    # save new language
    mongo.set_language(current_student.id, payload.language)
    return ProfileResponse(
        student_id=current_student.id,
        is_admin=current_student.is_admin,
        language=payload.language,
    )


@router.post("/session/start", response_model=SessionResponse)
async def start_session(
    request: Request,
    current_student: User = Depends(get_current_student),
):
    """
    (Authenticated) Allocate a new Chat Session ID and register it in the tracker.
    Returns the session_id the client should use when calling /ta/chat.
    """
    tracker = request.app.state.student_tracker
    session_id = str(uuid.uuid4())
    tracker.create_chat_session(student_id=current_student.id, session_id=session_id)
    return SessionResponse(session_id=session_id)


@router.delete("/session/end", status_code=status.HTTP_204_NO_CONTENT)
async def end_session(
    session_id: str,
    request: Request,
    current_student: User = Depends(get_current_student),
):
    """
    (Authenticated) Drop a specific chat session from in-memory tracker.
    Student can only drop their own sessions.
    """
    tracker = request.app.state.student_tracker
    # Verify ownership before dropping
    owner = tracker.get_student_id_by_session(session_id)
    if owner != current_student.id:
        raise HTTPException(status_code=403, detail="Session does not belong to this student.")
    tracker.drop_session(session_id)


@router.get("/sessions")
async def list_sessions(
    request: Request,
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=200),
    current_student: User = Depends(get_current_student),
):
    mongo = request.app.state.student_tracker.mongodb
    return await asyncio.to_thread(mongo.list_sessions, current_student.id, limit, offset, q)


@router.get("/sessions/{session_id}")
async def read_session(
    session_id: str,
    request: Request,
    current_student: User = Depends(get_current_student),
):
    mongo = request.app.state.student_tracker.mongodb
    session = await asyncio.to_thread(mongo.get_session_data, current_student.id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = []
    for chat in session.get("chats", []):
        visible = [m for m in chat.get("messages", []) if m.get("role") in {"student", "TA"}]
        latest_ta = next((m for m in reversed(visible) if m["role"] == "TA"), None)
        visible = [m for m in visible if m["role"] == "student"]
        if latest_ta:
            visible.append(latest_ta)
        last_ta = len(visible) - 1 if latest_ta else -1
        for index, message in enumerate(visible):
            messages.append({
                "id": f"{chat['id']}:{index}",
                "role": "user" if message["role"] == "student" else "ta",
                "content": message.get("message", ""),
                "timestamp": message.get("timestamp"),
                "ui_action": chat.get("ui_action") if index == last_ta else None,
            })
    last_chat = (session.get("chats") or [None])[-1]
    pending_id = last_chat.get("id") if last_chat and not any(
        m.get("role") == "TA" for m in last_chat.get("messages", [])
    ) else None
    return {"id": session_id, "messages": messages, "pending_chat_id": pending_id}


@router.post("/sessions/{session_id}/resume", response_model=SessionResponse)
async def resume_session(
    session_id: str,
    request: Request,
    current_student: User = Depends(get_current_student),
):
    tracker = request.app.state.student_tracker
    session = await asyncio.to_thread(tracker.mongodb.get_session_data, current_student.id, session_id)
    if not session or tracker.get_student_id_by_session(session_id) not in {None, current_student.id}:
        raise HTTPException(status_code=404, detail="Session not found.")
    await asyncio.to_thread(tracker.create_chat_session, current_student.id, session_id)
    return SessionResponse(session_id=session_id)


@router.post("/admin/reset", status_code=status.HTTP_200_OK)
async def reset_student(
    request: Request,
    payload: ResetRequest
):
    """
    Reset a student's database states (MongoDB state) and clear memory.
    Used for testing/development.
    """
    tracker = request.app.state.student_tracker
    mongo = tracker.mongodb
    
    # 1. Reset MongoDB state by re-creating it
    mongo.students.delete_one({"_id": payload.student_id})
    mongo.create_student(payload.student_id)
    
    # 2. Clear from tracker memory
    tracker._student_states.pop(payload.student_id, None)
    
    # 3. Close active sessions in memory for this student
    sessions_to_drop = [sid for sid, uid in tracker._session_map.items() if uid == payload.student_id]
    for sid in sessions_to_drop:
        tracker.drop_session(sid)
        
    return {"detail": f"Student {payload.student_id} state reset successfully."}
