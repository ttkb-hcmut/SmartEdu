from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Response
from pydantic import BaseModel, field_validator
from typing import List

from core.dependencies import get_ingestion_service
from knowledge.service.pdf_loader import topic_pdf_bytes
from student.auth import require_admin, get_current_student, User

router = APIRouter(tags=["Knowledge"])


######### schemas

def _canon_course(name: str) -> str:
    ## casing drift breaks uri prefix match downstream
    return " ".join(name.split()).title()


class CourseIngestionRequest(BaseModel):
    course_name: str = "Machine Learning"
    slide_files: List[str]                         # finished files
    textbook_files: List[str]
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
        if not fn.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF allowed: {fn}")
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
    service.validate_files(req.course_name, req.slide_files + req.textbook_files)

    background_tasks.add_task(service.run, req)

    return {
        "status": "accepted",
        "message": f"Course {req.course_name} ingestion started in background",
        "details": {
            "slides_count": len(req.slide_files),
            "textbooks_count": len(req.textbook_files),
        },
    }


@router.get("/ingest-report")
async def ingest_report(
    service=Depends(get_ingestion_service),
    _: User = Depends(require_admin),
):
    # outcome of the last background ingestion run
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
