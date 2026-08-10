"""Ablation runner: drive the TA over a QA fixture per preset, then score traces.

Requires DBs + Ollama up. Judge metrics additionally require
`deepeval set-ollama --model=<model> --base-url=http://localhost:11434`.

Usage (from capstone/):
    uv run python test/eval/run_ablation.py --fixture test/eval/fixtures/musique_cs.json \
        --presets rag,full --limit 10
    uv run python test/eval/run_ablation.py --fixture ... --score-only [--judge]
    uv run python test/eval/run_ablation.py --fixture ... --calibrate
"""

import argparse
import asyncio
import hashlib
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


def harness_policy_id(harness_id):
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId

    return (
        RetrievalPolicyId.BASELINE_V4
        if harness_id is RetrievalHarnessId.AGENTIC_V3
        else RetrievalPolicyId.BASELINE_V3
    )


def harness_run_ids(base_run_id: str, harnesses: list[str]) -> dict[str, str]:
    if len(harnesses) == 1:
        return {harnesses[0]: base_run_id}
    return {harness: f"{base_run_id}__{harness}" for harness in harnesses}


def repeat_fixture(fixture: list[dict], repeats: int) -> list[dict]:
    if repeats <= 1:
        return fixture
    return [
        {
            **item,
            "id": f"{item['id']}__repeat_{repeat_index}",
            "source_fixture_id": item.get("source_fixture_id", item["id"]),
        }
        for item in fixture
        for repeat_index in range(1, repeats + 1)
    ]


def write_run_manifest(
    results_dir: Path,
    *,
    run_id: str,
    fixture: list[dict],
    corpus_dir: Path,
    course: str,
    harness_runs: dict[str, str],
) -> Path:
    manifest_path = corpus_dir / "manifest.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ingestion_reports = sorted(results_dir.glob(f"ingest_{course}_*.json"))
    ingestion_report = ""
    expected_corpus = corpus_dir.resolve()
    for candidate in reversed(ingestion_reports):
        try:
            report = json.loads(candidate.read_text(encoding="utf-8"))
            report_corpus = Path(report.get("corpus", "")).resolve()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if report.get("status") == "COMPLETED" and report_corpus == expected_corpus:
            ingestion_report = str(candidate)
            break
    payload = {
        "run_id": run_id,
        "course": course,
        "fixture_cases": len(fixture),
        "fixture_ids": [item["id"] for item in fixture],
        "corpus_manifest": str(manifest_path),
        "corpus_manifest_sha256": digest,
        "ingestion_report": ingestion_report,
        "harness_runs": harness_runs,
    }
    path = results_dir / f"run_manifest_{run_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


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
        policy_id=harness_policy_id(harness_id),
        harness_id=harness_id,
        course_scope=course,
    )
    tracker.create_chat_session(BENCH_STUDENT, sid)
    t0 = time.perf_counter()
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
            print(f"[{label}] {item['id']} ({time.perf_counter()-t0:.1f}s)")
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


def score(
    fixture: list,
    presets: list,
    track: str,
    harness_runs: dict[str, str],
    base_run_id: str,
    judge: bool,
    run_manifest: Path | None = None,
) -> None:
    from TA.tracing.evaluator import (
        evaluate_session,
        judge_chat,
        load_session,
        render_paired_report,
        render_report,
    )

    rows = []
    expected_ids = {item["id"] for item in fixture}
    for harness, run_id in harness_runs.items():
        for preset in presets:
            paths = sorted(TRACE_DIR.glob(f"{session_name(preset, track, run_id)}__*.json"))
            if not paths:
                raise ValueError(f"no traces for {harness}/{preset} in run {run_id}")
            sessions = [load_session(path) for path in paths]
            validate_run_sessions(sessions, expected_ids, run_id)
            for session in sessions:
                harness_rows = evaluate_session(session, fixture)
                for row in harness_rows:
                    row["harness_id"] = row.get("harness_id") or harness
                rows += harness_rows

    if judge:
        for row in rows:
            row.update(judge_chat(row, row.get("retrieval_context", [])))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"ablation_{track}_{stamp}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sections = []
    if run_manifest:
        sections += [f"Run manifest: `{run_manifest}`", ""]
    for harness in harness_runs:
        group = [row for row in rows if row.get("harness_id") == harness]
        sections += [f"# Harness: {harness}", "", render_report(group), ""]
    if {"agentic-v2", "agentic-v3"} <= set(harness_runs):
        sections += [render_paired_report(rows), ""]
    report = "\n".join(sections)
    report_path = RESULTS_DIR / f"ablation_{track}_{base_run_id}_{stamp}.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nper-question rows -> {out}")
    print(f"full report -> {report_path}")


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
    ap.add_argument("--presets", default="rag,full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--course", default="", help="benchmark course scope, e.g. Bench_MuSiQue")
    ap.add_argument(
        "--harness",
        default="agentic-v3",
        choices=("agentic-v1", "agentic-v2", "agentic-v3", "fanout-v1"),
    )
    ap.add_argument(
        "--harnesses",
        default="",
        help="comma-separated harnesses for a paired run, e.g. agentic-v2,agentic-v3",
    )
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--ids", default="", help="comma-separated fixture IDs to run")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--corpus", default="test/eval/corpus/musique_cs")
    args = ap.parse_args()

    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    if args.ids:
        selected = {item.strip() for item in args.ids.split(",") if item.strip()}
        fixture = [item for item in fixture if item["id"] in selected]
    if args.limit:
        fixture = fixture[:args.limit]
    fixture = repeat_fixture(fixture, args.repeat)
    track = fixture[0]["track"] if fixture else "unknown"
    presets = [p.strip().upper() for p in args.presets.split(",")]
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    harness_names = (
        [item.strip() for item in args.harnesses.split(",") if item.strip()]
        if args.harnesses
        else [args.harness]
    )
    allowed_harnesses = {"agentic-v1", "agentic-v2", "agentic-v3", "fanout-v1"}
    unknown_harnesses = set(harness_names) - allowed_harnesses
    if unknown_harnesses:
        raise SystemExit(f"unknown harnesses: {sorted(unknown_harnesses)}")
    harness_runs = harness_run_ids(run_id, harness_names)

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
        try:
            for harness_name, harness_run_id in harness_runs.items():
                harness_id = RetrievalHarnessId(harness_name)
                if fixture:
                    await run_case(
                        ta,
                        tracker,
                        "FULL",
                        fixture[0],
                        track,
                        args.course,
                        harness_run_id,
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
                        harness_run_id,
                        harness_id,
                        code_state,
                    )
        finally:
            tracker.delete_student(BENCH_STUDENT)
    elif not args.run_id:
        raise SystemExit("--score-only requires the exact --run-id")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_manifest = write_run_manifest(
        RESULTS_DIR,
        run_id=run_id,
        fixture=fixture,
        corpus_dir=Path(args.corpus),
        course=args.course,
        harness_runs=harness_runs,
    )
    score(
        fixture,
        presets,
        track,
        harness_runs,
        run_id,
        judge=args.judge,
        run_manifest=run_manifest,
    )


if __name__ == "__main__":
    asyncio.run(main())
