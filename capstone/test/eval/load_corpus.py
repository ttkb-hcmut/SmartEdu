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
import json
from pathlib import Path

PASSAGE_ROLE = "Passage"
EMB_DIM = 768


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


def build_graph_payload(paragraphs: dict, course: str, embedder) -> tuple[list[dict], list[dict]]:
    items = list(paragraphs.items())
    ## batched masked mean, get_embedding averages pad tokens in
    vectors = embedder.get_embeddings([rec["text"] for _, rec in items])

    section_id = f"{course}/section/benchmark"
    sections = [{
        "id": section_id,
        "title": f"{course} benchmark corpus",
        "level": 0,
        "p_num": [0, len(items)],
        "order": 0,
        "parent_id": None,
    }]
    passages = [
        {
            "id": uri,
            "section_id": section_id,
            "p_num": [idx, idx],
            "text": rec["text"],
            "emb": vector,
            "uri": uri,
        }
        for idx, ((uri, rec), vector) in enumerate(zip(items, vectors))
    ]
    return sections, passages


def verify(milvus_db, graph_db, expected: set[str], course: str) -> None:
    missing_milvus = expected - milvus_db.list_ids(course)
    missing_neo4j = expected - graph_db.list_passage_uris(f"{course}/")
    if missing_milvus or missing_neo4j:
        raise ValueError(
            f"corpus gate still failing: Milvus missing {len(missing_milvus)}, "
            f"Neo4j missing {len(missing_neo4j)}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="test/eval/corpus/musique_cs")
    ap.add_argument("--course", default="Bench_MuSiQue")
    ap.add_argument("--db-name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paragraphs = load_manifest(Path(args.corpus))
    check_scope(paragraphs, args.course)
    print(f"manifest: {len(paragraphs)} canonical paragraphs, scope '{args.course}'")

    if args.dry_run:
        nodes = build_milvus_nodes(paragraphs)
        print(f"dry-run: would insert {len(nodes)} Milvus rows "
              f"(community='{args.course}') and {len(nodes)} :Passage nodes")
        return

    graph_db, milvus_db, embedder = boot_stores()

    ## community, NOT course -- list_ids/course_scope filter on community only
    milvus_db.insert_data(
        nodes=build_milvus_nodes(paragraphs),
        embedder=embedder,
        community=args.course,
    )
    print(f"milvus: inserted {len(paragraphs)} rows")

    sections, passages = build_graph_payload(paragraphs, args.course, embedder)
    graph_db.write_textbook_tree(
        sections=sections,
        passages=passages,
        book_uri=args.course,
        db_name=args.db_name,
        dim=EMB_DIM,
    )
    print(f"neo4j: wrote {len(passages)} :Passage nodes under 1 :Section")

    verify(milvus_db, graph_db, set(paragraphs), args.course)
    print("corpus gate: PASS")


if __name__ == "__main__":
    main()
