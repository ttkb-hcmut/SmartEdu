import io
import json
from typing import Dict, List


def persist_slide_kg(graph_db, milvus_db, embedder, db_name: str,
                     nodes: List[Dict], edges: List[Dict], clusters: List[Dict]):
    ## dual-store write atomic-ish, one seam
    graph_db.import_data(db_name=db_name, nodes=nodes, edges=edges, clusters=clusters)
    milvus_db.insert_data(nodes=nodes, embedder=embedder)


def persist_report(minio_repo, course_name: str, run_id: str, report: Dict) -> str:
    ## worker split kills memory report
    payload = json.dumps(report, ensure_ascii=False, default=str).encode("utf-8")
    for object_name in (f"{course_name}/_reports/{run_id}.json",
                        f"{course_name}/_reports/latest.json"):
        minio_repo.client.put_object(
            minio_repo.bucket_name, object_name,
            io.BytesIO(payload), length=len(payload),
            content_type="application/json"
        )
    return f"minio://{minio_repo.bucket_name}/{course_name}/_reports/{run_id}.json"
