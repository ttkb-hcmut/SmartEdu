"""Dual-index corpus loader: write benchmark paragraphs to Milvus AND Neo4j.

Satisfies the `verify_corpus_ready` gate in run_ablation.py, which requires every
canonical manifest URI to appear as a Milvus `id` (scoped by `community`) and as a
Neo4j `Passage.uri` (scoped by prefix). Nothing else writes both stores under a
shared identity.

Usage (from capstone/):
    uv run python test/eval/load_corpus.py --corpus test/eval/corpus/musique_cs \
        --course Bench_MuSiQue [--dry-run]
"""

import argparse
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

PASSAGE_ROLE = "Passage"
EMB_DIM = 768
RESULTS_DIR = Path("test/eval/results")


@contextmanager
def measure(timings: dict, name: str):
    started = perf_counter()
    try:
        yield
    finally:
        timings[name] = round((perf_counter() - started) * 1000, 3)


def boot_stores():
    ## mirrors run_ablation.boot_ta minus llm/tracker, lazy so --dry-run needs no DBs
    from core.repo.graph.graphdb import GraphDB
    from core.repo.milvus_db.mil import MilvusDB
    from core.model.embedding import Embedder
    from core.config import Neo, Mil_conf, Emb_conf

    return (
        GraphDB(config=Neo()),
        MilvusDB(config=Mil_conf()),
        Embedder(config=Emb_conf()),
    )


def load_manifest(corpus_dir: Path) -> dict:
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paragraphs = manifest.get("paragraphs", {})
    if not paragraphs:
        raise ValueError(f"no canonical paragraphs in {manifest_path}")
    return paragraphs


def check_scope(paragraphs: dict, course: str) -> None:
    ## URI prefix is the graph-side scope filter, mismatch = silent empty retrieval
    stray = [uri for uri in paragraphs if not uri.startswith(f"{course}/")]
    if stray:
        raise ValueError(
            f"{len(stray)} paragraph URIs do not start with '{course}/', "
            f"e.g. {stray[0]} -- wrong --course for this corpus"
        )


def build_milvus_nodes(paragraphs: dict) -> list[dict]:
    ## rrole must be non-empty or insert_data drops the row
    return [
        {
            "id": uri,
            "name": rec["title"],
            "content": rec["text"],
            "rrole": PASSAGE_ROLE,
        }
        for uri, rec in paragraphs.items()
    ]


def section_id_for(course: str, title: str) -> str:
    ## title is the section identity, hashed so punctuation/unicode cannot break the id
    slug = hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]
    return f"{course}/section/{slug}"


def build_graph_payload(paragraphs: dict, course: str, embedder) -> tuple[list[dict], list[dict]]:
    items = list(paragraphs.items())
    ## batched masked mean, get_embedding averages pad tokens in
    vectors = embedder.get_embeddings([rec["text"] for _, rec in items])

    ## one :Section per source article, not one synthetic bucket -- concept extraction
    ## runs per section, and a single flat section would collapse to one giant call
    sections, order = {}, {}
    for uri, rec in items:
        title = rec["title"]
        sid = section_id_for(course, title)
        if sid not in sections:
            sections[sid] = {
                "id": sid,
                "title": title,
                "level": 0,
                "p_num": [0, 0],
                "order": len(sections),
                "parent_id": None,
            }
            order[sid] = 0
    passages = []
    for (uri, rec), vector in zip(items, vectors):
        sid = section_id_for(course, rec["title"])
        local = order[sid]
        order[sid] = local + 1
        sections[sid]["p_num"] = [0, order[sid]]
        passages.append({
            "id": uri,
            "section_id": sid,
            "p_num": [local, local],
            "text": rec["text"],
            "emb": vector,
            "uri": uri,
        })
    return list(sections.values()), passages


def extract_concept_layer(graph_db, milvus_db, embedder, course: str, db_name,
                          workers: int, limit_sections: int) -> dict:
    ## the missing half: passages alone give anchor_concepts an empty concept set,
    ## so a textbook-only corpus has always produced 0 Entity nodes and 0 anchors
    import asyncio

    from core.config import Ingest_param
    from knowledge.ingest.anchor import anchor_concepts
    from knowledge.ingest.persist import persist_slide_kg
    from knowledge.pipeline.tasks import build_textbook_queue_items, extract_textbook_kg

    sections = graph_db.list_sections_for_extraction(f"{course}/", db_name)
    items = build_textbook_queue_items(sections, course)
    if limit_sections:
        items = items[:limit_sections]
    if not items:
        return {"sections": 0, "concepts": 0, "edges": 0, "anchors": 0}

    nodes, edges, clusters = asyncio.run(extract_textbook_kg(course, items, workers))
    persist_slide_kg(graph_db, milvus_db, embedder, db_name, course, nodes, edges, clusters)
    concept_nodes = [n for n in nodes if n.get("typeNode") == "Concept"]
    anchors = anchor_concepts(
        embedder, graph_db, concept_nodes, course, db_name, Ingest_param()
    )
    return {
        "sections": len(items),
        "concepts": len(concept_nodes),
        "edges": len(edges),
        "anchors": anchors,
    }


def verify(milvus_db, graph_db, expected: set[str], course: str) -> None:
    missing_milvus = expected - milvus_db.list_ids(course)
    missing_neo4j = expected - graph_db.list_passage_uris(f"{course}/")
    if missing_milvus or missing_neo4j:
        raise ValueError(
            f"corpus gate still failing: Milvus missing {len(missing_milvus)}, "
            f"Neo4j missing {len(missing_neo4j)}"
        )


def write_report(
    course: str,
    corpus: Path,
    paragraph_count: int,
    timings: dict,
    status: str,
    error: str = "",
) -> Path:
    total_ms = timings.get("total_ms", 0)
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "course": course,
        "corpus": str(corpus),
        "paragraphs": paragraph_count,
        "stores": ["milvus", "neo4j"],
        "status": status,
        "error": error,
        "timings": timings,
        "paragraphs_per_second": round(
            paragraph_count / (total_ms / 1000), 3
        ) if status == "COMPLETED" and total_ms else None,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"ingest_{course}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="test/eval/corpus/musique_cs")
    ap.add_argument("--course", default="Bench_MuSiQue")
    ap.add_argument("--db-name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extract-concepts", action="store_true",
                    help="run LLM concept extraction + anchoring after passages are written")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit-sections", type=int, default=0,
                    help="cap extracted sections (smoke runs); 0 = all")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    paragraphs = load_manifest(corpus)
    check_scope(paragraphs, args.course)
    print(f"manifest: {len(paragraphs)} canonical paragraphs, scope '{args.course}'")

    if args.dry_run:
        nodes = build_milvus_nodes(paragraphs)
        print(f"dry-run: would insert {len(nodes)} Milvus rows "
              f"(community='{args.course}') and {len(nodes)} :Passage nodes")
        return

    started = perf_counter()
    timings = {}
    try:
        with measure(timings, "store_boot_ms"):
            graph_db, milvus_db, embedder = boot_stores()

        ## community, NOT course -- list_ids/course_scope filter on community only
        with measure(timings, "milvus_write_ms"):
            milvus_db.insert_data(
                nodes=build_milvus_nodes(paragraphs),
                embedder=embedder,
                community=args.course,
            )
        print(f"milvus: inserted {len(paragraphs)} rows")

        with measure(timings, "neo4j_embedding_ms"):
            sections, passages = build_graph_payload(paragraphs, args.course, embedder)
        with measure(timings, "neo4j_write_ms"):
            graph_db.write_textbook_tree(
                sections=sections,
                passages=passages,
                book_uri=args.course,
                db_name=args.db_name,
                dim=EMB_DIM,
            )
        print(f"neo4j: wrote {len(passages)} :Passage nodes under {len(sections)} :Section")

        with measure(timings, "verification_ms"):
            verify(milvus_db, graph_db, set(paragraphs), args.course)
        print("corpus gate: PASS")

        if args.extract_concepts:
            with measure(timings, "concept_extraction_ms"):
                stats = extract_concept_layer(
                    graph_db, milvus_db, embedder, args.course, args.db_name,
                    args.workers, args.limit_sections,
                )
            timings.update({f"concept_{k}": v for k, v in stats.items()})
            print(
                f"concepts: {stats['concepts']} nodes, {stats['edges']} edges, "
                f"{stats['anchors']} anchors from {stats['sections']} sections"
            )
            if stats["concepts"] and not stats["anchors"]:
                raise ValueError(
                    f"{stats['concepts']} concepts extracted but 0 anchored to passages -- "
                    f"orphan concept layer, check anchor_score_min"
                )
    except Exception as exc:
        timings["total_ms"] = round((perf_counter() - started) * 1000, 3)
        report_path = write_report(
            args.course, corpus, len(paragraphs), timings,
            "FAILED", f"{type(exc).__name__}: {exc}",
        )
        print(f"failed ingestion report -> {report_path}")
        raise

    timings["total_ms"] = round((perf_counter() - started) * 1000, 3)
    report_path = write_report(
        args.course, corpus, len(paragraphs), timings, "COMPLETED"
    )
    print(f"ingestion report -> {report_path}")


if __name__ == "__main__":
    main()
