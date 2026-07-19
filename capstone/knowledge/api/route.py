import json
import os
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Response
from pydantic import BaseModel, field_validator
from typing import List, Optional

from core.dependencies import get_ingestion_service
from knowledge.service.pdf_loader import topic_pdf_bytes
from student.auth import require_admin, get_current_student, User

router = APIRouter(tags=["Knowledge"])


######### schemas

def _canon_course(name: str) -> str:
    ## casing drift breaks uri prefix match downstream
    return " ".join(name.split()).title()


VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mp3", ".m4a", ".wav")


class CourseIngestionRequest(BaseModel):
    course_name: str = "Machine Learning"
    slide_files: List[str]                         # finished files
    textbook_files: List[str]
    video_files: List[str] = []
    reset: bool = True

    _canon = field_validator("course_name")(_canon_course)


class UploadUrlRequest(BaseModel):
    course_name: str
    file_names: List[str]                # files needing a presigned upload url

    _canon = field_validator("course_name")(_canon_course)


class PresignedTarget(BaseModel):
    file_name: str
    url: str                              # browser PUTs raw pdf bytes straight here


class UploadUrlResponse(BaseModel):
    targets: List[PresignedTarget]


######### upload: hand the browser presigned PUT urls (admin only)

@router.post("/upload-url", response_model=UploadUrlResponse)
async def get_upload_urls(
    req: UploadUrlRequest,
    service=Depends(get_ingestion_service),
    _: User = Depends(require_admin),
):
    targets = []
    for fn in req.file_names:
        if not fn.lower().endswith((".pdf",) + VIDEO_EXTS):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {fn}")
        url = service.minio_repo.presigned_put_url(course_name=req.course_name, file_name=fn)
        targets.append(PresignedTarget(file_name=fn, url=url))
    return UploadUrlResponse(targets=targets)


######### ingest: process the already-uploaded pdfs (admin only)

@router.post("/ingest-course")
async def ingest_course(
    req: CourseIngestionRequest,
    background_tasks: BackgroundTasks,
    service=Depends(get_ingestion_service),
    _: User = Depends(require_admin),
):
    # confirm files are in storage, then ingest in background and answer fast
    service.validate_files(
        req.course_name, req.slide_files + req.textbook_files, req.video_files
    )

    flow_run_id = None
    ## inline = parity escape hatch, prefect = worker path (ADR-0005)
    if os.getenv("INGEST_ORCHESTRATOR", "prefect") == "prefect":
        import inspect
        from prefect.deployments import run_deployment
        ## run_deployment is sync/async-dual — await only when it hands back a coroutine
        fr = run_deployment(
            name="course-flow/course-ingest",
            parameters={
                "course_name": req.course_name,
                "slide_files": req.slide_files,
                "textbook_files": req.textbook_files,
                "video_files": req.video_files,
                "reset": req.reset,
            },
            timeout=0,
        )
        if inspect.isawaitable(fr):
            fr = await fr
        flow_run_id = str(fr.id)
    else:
        background_tasks.add_task(service.run, req)

    return {
        "status": "accepted",
        "message": f"Course {req.course_name} ingestion started",
        "flow_run_id": flow_run_id,
        "details": {
            "slides_count": len(req.slide_files),
            "textbooks_count": len(req.textbook_files),
        },
    }


@router.get("/ingest-report")
async def ingest_report(
    course: Optional[str] = None,
    run_id: Optional[str] = None,
    service=Depends(get_ingestion_service),
    _: User = Depends(require_admin),
):
    ## worker-run reports live in minio, in-memory only covers inline mode
    if course:
        obj = f"{_canon_course(course)}/_reports/{run_id or 'latest'}.json"
        try:
            data = service.minio_repo.get_object_bytes(obj)
            return json.loads(data)
        except Exception:
            raise HTTPException(status_code=404, detail=f"No report found for {course}.")
    return service.last_report or {"status": "no ingestion run yet"}


######### discovery + viewer (any logged-in student)

@router.get("/courses")
async def list_courses(
    service=Depends(get_ingestion_service),
    _: User = Depends(get_current_student),
):
    # every course folder in storage
    return {"courses": service.minio_repo.list_courses()}


@router.get("/courses/{course_name}/topics")
async def list_topics(
    course_name: str,
    service=Depends(get_ingestion_service),
    _: User = Depends(get_current_student),
):
    # every topic folder inside one course
    return {"course": course_name, "topics": service.minio_repo.list_topics(course_name)}


@router.get("/pdf/{course_name}/{topic}")
async def get_topic_pdf(
    course_name: str,
    topic: str,
    service=Depends(get_ingestion_service),
    _: User = Depends(get_current_student),
):
    # stream the topic's small page pdf; 404 if missing
    data = topic_pdf_bytes(service.minio_repo, course_name, topic)
    if data is None:
        raise HTTPException(status_code=404, detail="Topic PDF not found.")
    return Response(content=data, media_type="application/pdf")
