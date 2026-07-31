import io
import json
from typing import Dict, List

from core.schema.ingest import REPORT_RUNNING, TERMINAL_REPORT_STATUSES


def persist_slide_kg(graph_db, milvus_db, embedder, db_name: str, course: str,
                     nodes: List[Dict], edges: List[Dict], clusters: List[Dict]):
    ## dual-store write atomic-ish, one seam
    graph_db.import_data(db_name=db_name, nodes=nodes, edges=edges, clusters=clusters)
    milvus_db.insert_data(nodes=nodes, embedder=embedder, course=course)


def persist_report(minio_repo, course_name: str, run_id: str, report: Dict) -> str:
    ## worker split kills memory report
    if report.get("run_id") != run_id:
        raise ValueError("report run_id does not match object path")

    status = report.get("status")
    if status != REPORT_RUNNING and status not in TERMINAL_REPORT_STATUSES:
        raise ValueError(f"invalid report status: {status}")

    payload = json.dumps(report, ensure_ascii=False, default=str).encode("utf-8")
    report_name = f"{course_name}/_reports/{run_id}.json"
    latest_name = f"{course_name}/_reports/latest.json"

    if status in TERMINAL_REPORT_STATUSES:
        if minio_repo.object_exists(report_name):
            raise FileExistsError(f"immutable report already exists: {report_name}")
        _put_report(minio_repo, report_name, payload)

    if minio_repo.object_exists(latest_name):
        latest = json.loads(minio_repo.get_object_bytes(latest_name))
        latest_start = latest.get("started_at")
        if latest_start and latest_start > report["started_at"]:
            return f"minio://{minio_repo.bucket_name}/{report_name}"

    _put_report(minio_repo, latest_name, payload)
    return f"minio://{minio_repo.bucket_name}/{report_name}"


def _put_report(minio_repo, object_name: str, payload: bytes) -> None:
    minio_repo.client.put_object(
        minio_repo.bucket_name,
        object_name,
        io.BytesIO(payload),
        length=len(payload),
        content_type="application/json",
    )
