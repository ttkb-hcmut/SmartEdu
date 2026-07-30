"""Ablation runner: drive the TA over a QA fixture per preset, then score traces.

Requires DBs + Ollama up. Judge metrics additionally require
`deepeval set-ollama --model=<model> --base-url=http://localhost:11434`.

Usage (from capstone/):
    uv run python test/eval/run_ablation.py --fixture test/eval/fixtures/musique_cs.json \
        --presets plain,rag,full --limit 10
    uv run python test/eval/run_ablation.py --fixture ... --score-only [--judge]
    uv run python test/eval/run_ablation.py --fixture ... --calibrate
"""

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

from TA.tracing.writer import _DEFAULT_LOG_DIR

BENCH_STUDENT = "bench_student"
TRACE_DIR = _DEFAULT_LOG_DIR
RESULTS_DIR = Path("test/eval/results")


def boot_ta():
    ## mirrors core/api/life_span.py minus fastapi/knowledge
    from core.llm.llm_engine import CoreLLMEngine
    from core.repo.graph.graphdb import GraphDB
    from core.repo.milvus_db.mil import MilvusDB
    from core.model.embedding import Embedder
    from core.repo.storage.minio_repo import MinioDB
    from core.repo.nosql.mongo_db import Mongo_DB
    from core.repo.sql.sql_db import SQL_DB
    from core.config import Neo, Mil_conf, Emb_conf, Minio_conf, MySQL_conf, NeoStudent, TA_conf
    from student.Student_Tracker import Student_Tracker
    from TA.ta_module import TAModule

    llm = CoreLLMEngine()
    graph_db = GraphDB(config=Neo())
    milvus_db = MilvusDB(config=Mil_conf())
    embedder = Embedder(config=Emb_conf())
    minio_repo = MinioDB(config=Minio_conf())
    tracker = Student_Tracker(graphdb=GraphDB(config=NeoStudent), sqldb=SQL_DB(config=MySQL_conf()), mongodb=Mongo_DB())
    ta = TAModule(llm=llm, graph_db=graph_db, milvus_db=milvus_db, embedder=embedder,
                  minio=minio_repo, student_tracker=tracker, config=TA_conf())
    return ta, tracker


def session_name(preset: str, track: str, run_id: str) -> str:
    return f"bench_{preset.lower()}_{track}_{run_id}"


def read_code_state():
    from core.schema.retrieval import RetrievalCodeState

    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return RetrievalCodeState(revision=revision, dirty=dirty)


def validate_run_sessions(sessions, expected_ids: set[str], run_id: str):
    chats = [chat for session in sessions for chat in session.chat]
    ids = [chat.question_id for chat in chats]
    if any(chat.run_id != run_id for chat in chats):
        raise ValueError(f"stale trace found outside run {run_id}")
    if any(chat.warmup for chat in chats):
        raise ValueError("warm-up trace mixed into scored cases")
    if any(chat.status != "SUCCESS" for chat in chats):
        failed = [chat.question_id for chat in chats if chat.status != "SUCCESS"]
        raise ValueError(f"invalid benchmark cases: {failed}")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate benchmark case trace")
    if set(ids) != expected_ids:
        raise ValueError(
            f"incomplete run: missing={sorted(expected_ids - set(ids))}, "
            f"extra={sorted(set(ids) - expected_ids)}"
        )
    return chats


def verify_corpus_ready(milvus_db, graph_db, manifest_path: Path, course: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = set(manifest.get("paragraphs", {}))
    if not expected:
        raise ValueError(f"no canonical paragraphs in {manifest_path}")
    milvus_ids = milvus_db.list_ids(course)
    neo4j_ids = graph_db.list_passage_uris(f"{course}/")
    missing_milvus = expected - milvus_ids
    missing_neo4j = expected - neo4j_ids
    if missing_milvus or missing_neo4j:
        raise ValueError(
            f"dual-index corpus gate failed: Milvus missing {len(missing_milvus)}, "
            f"Neo4j missing {len(missing_neo4j)}"
        )


async def run_case(
    ta,
    tracker,
    preset: str,
    item: dict,
    track: str,
    course: str,
    run_id: str,
    harness_id,
    code_state,
    warmup: bool = False,
):
    from core.config import Retrieve_param
    from core.schema.retrieval import RetrievalCase, RetrievalCaseKind, RetrievalRoute

    question_id = f"warmup-{item['id']}" if warmup else item["id"]
    prefix = "WARMUP" if warmup else preset
    sid = f"{session_name(prefix, track, run_id)}__{question_id}"
    rp = Retrieve_param.from_preset(
        preset,
        harness_id=harness_id,
        course_scope=course,
    )
    tracker.create_chat_session(BENCH_STUDENT, sid)
    t0 = time.time()
    try:
        result = await ta.run(
            user_input=item["question"],
            session_id=sid,
            language="eng",
            retrieve_param=rp,
            retrieval_case=RetrievalCase(
                run_id=run_id,
                question_id=question_id,
                kind=RetrievalCaseKind.WARMUP if warmup else RetrievalCaseKind.BENCHMARK,
                forced_route=RetrievalRoute.RETRIEVE,
            ),
            code_state=code_state,
        )
        label = "WARMUP" if warmup else preset
        if result["status"] == "SUCCESS":
            print(f"[{label}] {item['id']} ({time.time()-t0:.1f}s)")
        else:
            print(f"[{label}] {item['id']} FAILED: {result['errors']}")
    finally:
        tracker.drop_session(sid)


async def run_preset(
    ta,
    tracker,
    preset: str,
    fixture: list,
    track: str,
    course: str,
    run_id: str,
    harness_id,
    code_state,
):
    for i, q in enumerate(fixture):
        print(f"[{preset}] {i+1}/{len(fixture)}", end=" ")
        await run_case(
            ta,
            tracker,
            preset,
            q,
            track,
            course,
            run_id,
            harness_id,
            code_state,
        )


def score(fixture: list, presets: list, track: str, run_id: str, judge: bool) -> None:
    from TA.tracing.evaluator import load_session, evaluate_session, render_table, judge_chat

    rows = []
    expected_ids = {item["id"] for item in fixture}
    for preset in presets:
        paths = sorted(TRACE_DIR.glob(f"{session_name(preset, track, run_id)}__*.json"))
        if not paths:
            raise ValueError(f"no traces for {preset} in run {run_id}")
        sessions = [load_session(path) for path in paths]
        validate_run_sessions(sessions, expected_ids, run_id)
        for session in sessions:
            rows += evaluate_session(session, fixture)

    if judge:
        for row in rows:
            row.update(judge_chat(row, row.get("retrieval_context", [])))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"ablation_{track}_{stamp}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    table = render_table(rows)
    (RESULTS_DIR / f"ablation_{track}_{stamp}.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"\nper-question rows -> {out}")


def calibrate(fixture: list, corpus_dir: Path) -> None:
    """Judge floor test: null answers + random-chunk answers must score low."""
    from TA.tracing.evaluator import judge_chat

    chunks = [p.read_text(encoding="utf-8") for p in sorted(corpus_dir.rglob("*.txt"))[:200]]
    if not chunks:
        raise SystemExit(f"no corpus chunks under {corpus_dir}")

    results = {"null": [], "random": []}
    for q in fixture:
        base = {"query": q["question"], "gold_answer": q.get("gold_answer", "")}
        null_row = {**base, "answer": "I don't know."}
        rand_chunk = random.choice(chunks)
        rand_row = {**base, "answer": rand_chunk[:300]}
        results["null"].append(judge_chat(null_row, [], runs=3))
        results["random"].append(judge_chat(rand_row, [rand_chunk], runs=3))

    print("\n== Judge calibration (all metrics must sit at the floor) ==")
    for agent, rows in results.items():
        keys = sorted({k for r in rows for k in r if not k.endswith("_spread")})
        for k in keys:
            vals = [r[k] for r in rows if k in r]
            if vals:
                print(f"{agent:>7} {k}: median={statistics.median(vals):.2f} max={max(vals):.2f}")
    print("gate: if any median is high, the judge is NOT trusted — fix before reading ablation judge columns")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--presets", default="plain,rag,full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--course", default="", help="benchmark course scope, e.g. Bench_MuSiQue")
    ap.add_argument("--harness", default="agentic-v1", choices=("agentic-v1", "fanout-v1"))
    ap.add_argument("--run-id", default="")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--corpus", default="test/eval/corpus/musique_cs")
    args = ap.parse_args()

    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    if args.limit:
        fixture = fixture[:args.limit]
    track = fixture[0]["track"] if fixture else "unknown"
    presets = [p.strip().upper() for p in args.presets.split(",")]
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    if args.calibrate:
        calibrate(fixture, Path(args.corpus))
        return

    if not args.score_only:
        from core.schema.retrieval import RetrievalHarnessId

        ta, tracker = boot_ta()
        if not args.course:
            raise SystemExit("live benchmark requires --course for dual-index scoping")
        verify_corpus_ready(
            ta.tools_factory.milvus_db,
            ta.tools_factory.graph_db,
            Path(args.corpus) / "manifest.json",
            args.course,
        )
        code_state = read_code_state()
        harness_id = RetrievalHarnessId(args.harness)
        try:
            if fixture:
                await run_case(
                    ta,
                    tracker,
                    "FULL",
                    fixture[0],
                    track,
                    args.course,
                    run_id,
                    harness_id,
                    code_state,
                    warmup=True,
                )
            for preset in presets:
                await run_preset(
                    ta,
                    tracker,
                    preset,
                    fixture,
                    track,
                    args.course,
                    run_id,
                    harness_id,
                    code_state,
                )
        finally:
            tracker.delete_student(BENCH_STUDENT)
    elif not args.run_id:
        raise SystemExit("--score-only requires the exact --run-id")

    score(fixture, presets, track, run_id, judge=args.judge)


if __name__ == "__main__":
    asyncio.run(main())
