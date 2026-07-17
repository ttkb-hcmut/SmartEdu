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


def session_name(preset: str, track: str) -> str:
    return f"bench_{preset.lower()}_{track}"


async def run_preset(ta, tracker, preset: str, fixture: list, track: str, course: str):
    from core.config import Retrieve_param
    rp = Retrieve_param.from_preset(preset, benchmark_course=course) if course else Retrieve_param.from_preset(preset)
    for i, q in enumerate(fixture):
        ## fresh session per question -> no history leak between fixture items
        sid = f"{session_name(preset, track)}__{q['id']}"
        tracker.create_chat_session(BENCH_STUDENT, sid)
        t0 = time.time()
        try:
            await ta.run(user_input=q["question"], session_id=sid, language="eng", retrieve_param=rp)
            print(f"[{preset}] {i+1}/{len(fixture)} {q['id']} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"[{preset}] {i+1}/{len(fixture)} {q['id']} FAILED: {e}")
        finally:
            tracker.drop_session(sid)


def score(fixture: list, presets: list, track: str, judge: bool) -> None:
    from TA.tracing.evaluator import load_session, evaluate_session, render_table, judge_chat

    rows = []
    for preset in presets:
        paths = sorted(TRACE_DIR.glob(f"{session_name(preset, track)}__*.json"))
        if not paths:
            print(f"no traces for {preset}: {TRACE_DIR / (session_name(preset, track) + '__*.json')}")
            continue
        for path in paths:
            rows += evaluate_session(load_session(path), fixture)

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

    if args.calibrate:
        calibrate(fixture, Path(args.corpus))
        return

    if not args.score_only:
        ta, tracker = boot_ta()
        for preset in presets:
            await run_preset(ta, tracker, preset, fixture, track, args.course)

    score(fixture, presets, track, judge=args.judge)


if __name__ == "__main__":
    asyncio.run(main())
