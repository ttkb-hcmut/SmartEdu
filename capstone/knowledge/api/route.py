import asyncio
import json
import os
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Response
from pydantic import BaseModel, field_validator
from typing import List, Optional

from core.dependencies import get_ingestion_service, get_knowledge_module
from core.repo.storage.minio_repo import resolve_pdf_reference, validate_file_names
from knowledge.pipeline.readiness import capability_status, required_capabilities
from knowledge.pipeline.submit import course_submit
from knowledge.service.pdf_loader import topic_pdf_bytes
from core.security import require_admin_token, verify_token, TokenUser

router = APIRouter(tags=["Knowledge"])


######### schemas

def _canon_course(name: str) -> str:
    ## casing drift breaks uri prefix match downstream
    return " ".join(name.split()).title()


VID_EXT = (".mp4", ".mkv", ".webm", ".mp3", ".m4a", ".wav")


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
    url: str                              # browser PUTs raw source bytes straight here


class UploadUrlResponse(BaseModel):
    targets: List[PresignedTarget]


######### upload: hand the browser presigned PUT urls (admin only)

@router.post("/upload-url", response_model=UploadUrlResponse)
async def get_upload_urls(
    req: UploadUrlRequest,
    service=Depends(get_ingestion_service),
    _: TokenUser = Depends(require_admin_token),
):
    try:
        validate_file_names(req.file_names)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=f"Invalid file name: {err}")

    targets = []
    for fn in req.file_names:
        if not fn.lower().endswith((".pdf",) + VID_EXT):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {fn}")
        url = service.minio_repo.presigned_put_url(course_name=req.course_name, file_name=fn)
        targets.append(PresignedTarget(file_name=fn, url=url))
    return UploadUrlResponse(targets=targets)


######### ingest: process the already-uploaded pdfs (admin only)

@router.post("/ingest-course", status_code=202)
async def ingest_course(
    req: CourseIngestionRequest,
    background_tasks: BackgroundTasks,
    service=Depends(get_ingestion_service),
    _: TokenUser = Depends(require_admin_token),
):
    inline = os.getenv("INGEST_ORCH", "prefect") != "prefect"
    if inline and req.video_files:
        raise HTTPException(
            status_code=409,
            detail="Video ingestion requires the Prefect orchestrator.",
        )

    ## storage exist ---> ingest ---> response
    service.f_valid(
        req.course_name, req.slide_files + req.textbook_files, req.video_files
    )

    flow_run_id = None
    ## inline parity escape, Prefect submit seam per ADR-0007
    if not inline:
        capabilities = await capability_status(required_capabilities(
            req.slide_files, req.textbook_files, req.video_files, req.reset
        ))
        if capabilities.missing:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "ingest_capability_unavailable",
                    "missing": list(capabilities.missing),
                    "present": list(capabilities.present),
                },
            )
        flow_run_id = await course_submit({
            "course_name": req.course_name,
            "slide_files": req.slide_files,
            "textbook_files": req.textbook_files,
            "video_files": req.video_files,
            "reset": req.reset,
        })
    else:
        background_tasks.add_task(service.run, req)

    return {
        "status": "accepted",
        "message": f"Course {req.course_name} ingestion started",
        "flow_run_id": flow_run_id,
        "details": {
            "slides_count": len(req.slide_files),
            "textbooks_count": len(req.textbook_files),
            "videos_count": len(req.video_files),
        },
    }


@router.get("/ingest-report")
async def ingest_report(
    course: Optional[str] = None,
    run_id: Optional[str] = None,
    service=Depends(get_ingestion_service),
    _: TokenUser = Depends(require_admin_token),
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


def _search_payload(module, query: str, course: str | None) -> dict:
    vector = module.embedder.get_embedding(query)
    scope = _canon_course(course) if course else None
    expr = module.milvus_db._course_expr(scope) if scope else None
    concept_rows = module.milvus_db.search_vec(vector, top_k=5, expr=expr)
    passage_rows = module.graph_db.passage_search(
        vector, query_text=query, top_k=5, uri_prefix=f"{scope}/" if scope else None,
    )
    source_rows = module.graph_db.resolve_entity_sources(
        [row["id"] for row in concept_rows if row.get("id")]
    ) if concept_rows else []
    sources = {row["id"]: row for row in source_rows}
    concepts = []
    for row in concept_rows:
        document, page = resolve_pdf_reference(sources.get(row.get("id"), {}))
        concepts.append({
            "id": row.get("id"), "course": row.get("community") or scope,
            "title": row.get("name") or row.get("id"),
            "snippet": str(row.get("text") or "")[:240], "score": row.get("score"),
            "document": document, "page": page,
        })
    passages = []
    for row in passage_rows:
        document, page = resolve_pdf_reference({"uri": row.get("uri"), "p_lo": row.get("p_lo")})
        passages.append({
            "id": row.get("id"), "course": str(row.get("uri") or "").split("/", 1)[0],
            "title": str(row.get("uri") or row.get("id") or "").split("/")[-1],
            "snippet": str(row.get("text") or "")[:240], "score": row.get("score"),
            "document": document, "page": page,
        })
    return {"concepts": concepts, "passages": passages}


@router.get("/search")
async def search_materials(
    q: str = "",
    course: str | None = None,
    module=Depends(get_knowledge_module),
    _: TokenUser = Depends(verify_token),
):
    query = q.strip()
    if len(query) < 2 or len(query) > 200:
        raise HTTPException(status_code=400, detail="Query must contain 2 to 200 characters.")
    try:
        return await asyncio.to_thread(_search_payload, module, query, course)
    except Exception:
        raise HTTPException(status_code=503, detail="Search is temporarily unavailable.")

@router.get("/courses")
async def list_courses(
    service=Depends(get_ingestion_service),
    _: TokenUser = Depends(verify_token),
):
    # every course folder in storage
    return {"courses": service.minio_repo.list_courses()}


@router.get("/courses/{course_name}/topics")
async def list_topics(
    course_name: str,
    service=Depends(get_ingestion_service),
    _: TokenUser = Depends(verify_token),
):
    # every topic folder inside one course
    return {"course": course_name, "topics": service.minio_repo.list_topics(course_name)}


@router.get("/pdf/{course_name}/{topic}")
async def get_topic_pdf(
    course_name: str,
    topic: str,
    service=Depends(get_ingestion_service),
    _: TokenUser = Depends(verify_token),
):
    # stream the topic's small page pdf; 404 if missing
    data = topic_pdf_bytes(service.minio_repo, course_name, topic)
    if data is None:
        raise HTTPException(status_code=404, detail="Topic PDF not found.")
    return Response(content=data, media_type="application/pdf")


@router.get("/pdf/raw/{course_name}/{file_name}")
async def get_raw_pdf(
    course_name: str,
    file_name: str,
    service=Depends(get_ingestion_service),
    _: TokenUser = Depends(verify_token),
):
    try:
        validate_file_names([course_name])
        validate_file_names([file_name])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid PDF path.") from exc
    if course_name in {".", ".."} or file_name in {".", ".."} or not file_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid PDF path.")
    key = service.minio_repo.raw_object_name(course_name, file_name)
    if not service.minio_repo.object_exists(key):
        raise HTTPException(status_code=404, detail="PDF not found.")
    return Response(content=service.minio_repo.get_object_bytes(key), media_type="application/pdf")
