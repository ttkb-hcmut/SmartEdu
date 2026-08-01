import asyncio
import json
import logging
import time
import uuid
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from student.auth import get_current_student, User
from student.memo import generate_uuidv7

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str      # Chat Session UUID — opaque, issued by POST /student/session/start
    user_input: str
    language: str = "vn"  # "vn" | "eng" — bilingual toggle


class ChatResponse(BaseModel):
    message: str
    ui_action: dict | None = None


class ChatAcceptedResponse(BaseModel):
    task_id: str
    status: str = "received"
    agent: str = "TA"


class ChatStatusResponse(BaseModel):
    task_id: str
    status: str          # "working" | "finished" | "Fail"
    agent_name: str | None = None
    intent: str | None = None
    thought: str | None = None
    result: ChatResponse | None = None
    error: str | None = None


def _get_session_id(payload: ChatRequest, request: Request, current_student: User) -> str:
    """
    Verify that the Bearer token's student_id owns this session_id.
    This prevents a student from hijacking another student's session.
    """
    tracker = request.app.state.student_tracker
    owner = tracker.get_student_id_by_session(payload.session_id)
    if owner is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call POST /student/session/start first.",
        )
    if owner != current_student.id:
        raise HTTPException(status_code=403, detail="Session does not belong to this token.")

    return payload.session_id


async def _run_ta_task(app_state, task_id: str, user_input: str, session_id: str, language: str = "vn"):
    queue: asyncio.Queue = app_state.ta_tasks[task_id]["queue"]

    async def emit(event: dict):
        await queue.put(event)

    async def update_status(node_name: str, state_update: dict):
        current_status = app_state.ta_tasks.get(task_id, {})
        app_state.ta_tasks[task_id] = {
            **current_status,
            "status": "working",
            "agent_name": node_name,
            "intent": state_update.get("intent", current_status.get("intent", "")),
            "thought": state_update.get("thought", current_status.get("thought", "")),
        }

    try:
        ta_module = app_state.TA
        result = await ta_module.run(user_input=user_input, session_id=session_id, update_callback=update_status, language=language, chat_id=task_id, emit=emit)
        current_status = app_state.ta_tasks.get(task_id, {})
        app_state.ta_tasks[task_id] = {
            **current_status,
            "status": "finished",
            "result": {"message": result["message"], "ui_action": result.get("ui_action")},
        }
        await emit({"type": "done", "message": result["message"], "ui_action": result.get("ui_action")})
        logger.info("TA task %s completed successfully.", task_id)
    except Exception:
        logger.exception("TA background task %s failed.", task_id)
        current_status = app_state.ta_tasks.get(task_id, {})
        app_state.ta_tasks[task_id] = {
            **current_status,
            "status": "Fail",
            "error": "Internal TA workflow error — check server logs.",
        }
        await emit({"type": "error", "error": "Internal TA workflow error — check server logs."})
    finally:
        ## stamp terminal time so the sweeper can evict, entry stays readable for /chat/status
        entry = app_state.ta_tasks.get(task_id)
        if entry is not None:
            entry["done_at"] = time.monotonic()
            ## stream_chat binds queue to a local before iterating, popping here cannot starve it
            entry.pop("queue", None)


@router.post("/chat", response_model=ChatAcceptedResponse, status_code=202)
async def chat_with_ta(
    request: Request,
    payload: ChatRequest,
    current_student: User = Depends(get_current_student),
):
    """
    Fire-and-forget chat endpoint.
    - Validates ownership then immediately schedules the TA workflow as a background task.
    - Returns task_id + status='received' so the client can poll GET /chat/status/{task_id}.
    """
    try:
        session_id = _get_session_id(payload, request, current_student)

        ta_module = request.app.state.TA
        tracker = request.app.state.student_tracker
        if not ta_module:
            raise HTTPException(status_code=500, detail="TAModule is not initialized.")

        task_id = generate_uuidv7()
        chat_id = task_id  # They are the same
        tracker.mongodb.create_chat(current_student.id, session_id, chat_id, payload.user_input)

        # Mark as processing before scheduling so the status endpoint never sees a missing key.
        # Queue created here (not in the task) so /chat/stream never races a missing queue.
        request.app.state.ta_tasks[task_id] = {
            "status": "working",
            "agent_name": "TA_Router (Thinking...)",
            "queue": asyncio.Queue(),
        }

        asyncio.create_task(
            _run_ta_task(request.app.state, task_id, payload.user_input, session_id, payload.language)
        )
        logger.info("TA task %s scheduled for session %s.", task_id, session_id)
        return ChatAcceptedResponse(task_id=task_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to schedule TA task for session %s.", payload.session_id)
        raise HTTPException(status_code=500, detail="Failed to schedule TA task.")


@router.get("/chat/status/{task_id}", response_model=ChatStatusResponse)
async def get_chat_status(task_id: str, request: Request, _: User = Depends(get_current_student)):
    """
    Poll the result of a previously submitted /chat request.
    Returns status='processing' while the workflow is running,
    status='done' with the result when finished,
    or status='error' with an error message on failure.
    """
    ta_tasks: dict = request.app.state.ta_tasks
    entry = ta_tasks.get(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    status = entry["status"]
    if status == "working":
        return ChatStatusResponse(
            task_id=task_id, 
            status="working",
            agent_name=entry.get("agent_name"),
            intent=entry.get("intent"),
            thought=entry.get("thought"),
        )
    if status == "finished":
        raw = entry["result"]
        return ChatStatusResponse(
            task_id=task_id,
            status="finished",
            agent_name=entry.get("agent_name"),
            intent=entry.get("intent"),
            thought=entry.get("thought"),
            result=ChatResponse(message=raw["message"], ui_action=raw.get("ui_action")),
        )
    # error
    return ChatStatusResponse(task_id=task_id, status="Fail", error=entry.get("error"))


@router.get("/chat/stream/{task_id}")
async def stream_chat(task_id: str, request: Request, _: User = Depends(get_current_student)):
    """
    SSE stream of a submitted /chat task: `step` (progress) + `token` (answer text),
    terminated by `done` (full message + ui_action) or `error`.
    Auth via Bearer header — the client uses fetch+reader, not EventSource (which can't set headers).
    """
    entry = request.app.state.ta_tasks.get(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    queue: asyncio.Queue = entry.get("queue")
    if queue is None:
        raise HTTPException(status_code=409, detail="Task has no active stream.")

    async def gen():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=180)
            except asyncio.TimeoutError:
                ## graph hung; bail so the generator never zombies
                yield f"data: {json.dumps({'type': 'error', 'error': 'TA timed out.'})}\n\n"
                return
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") in ("done", "error"):
                return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},  ## X-Accel disables proxy buffering
    )
