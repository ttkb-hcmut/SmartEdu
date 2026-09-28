"""Ablation runner: drive the TA over a QA fixture per preset, then score traces.

Requires DBs + Ollama up. Judge metrics additionally require
`deepeval set-ollama --model=<model> --base-url=http://localhost:11434`.

Usage (from capstone/):
    uv run python -m test.eval.run_ablation --fixture test/eval/fixtures/musique_cs.json \
        --presets rag,full --limit 10
    uv run python -m test.eval.run_ablation --fixture ... --score-only [--judge]
    uv run python -m test.eval.run_ablation --fixture ... --calibrate
"""

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

from TA.tracing.writer import _DEFAULT_LOG_DIR
from test.eval.provider_budget import ProviderRequestLedger

BENCH_STUDENT = "bench_student"
TRACE_DIR = _DEFAULT_LOG_DIR
RESULTS_DIR = Path("test/eval/results")
_RUN_CONTRACT_KEYS = (
    "version",
    "run_id",
    "fixture_digest",
    "corpus_digest",
    "policy_digests",
    "harness_id",
    "harness_digest",
    "code_digest",
    "schedule",
)


def harness_policy_id(harness_id):
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId

    if harness_id is RetrievalHarnessId.AGENTIC_V4:
        return RetrievalPolicyId.BASELINE_V5
    if harness_id is RetrievalHarnessId.AGENTIC_V3:
        return RetrievalPolicyId.BASELINE_V4
    return RetrievalPolicyId.BASELINE_V3


def paired_schedule(fixture: list[dict], presets: list[str]) -> list[dict]:
    schedule = []
    for index, item in enumerate(fixture):
        order = presets if index % 2 == 0 else list(reversed(presets))
        for preset in order:
            schedule.append({
                "fixture_index": index,
                "question_id": item["id"],
                "preset": preset,
                "case_key": f"{item['id']}::{preset}",
            })
    return schedule


def classify_provider_error(error) -> dict[str, object]:
    message = " ".join(error) if isinstance(error, (list, tuple)) else str(error or "")
    lowered = message.casefold()
    if "provider request budget exhausted" in lowered:
        return {"category": "provider_request_budget", "pause_immediately": True}
    session_limit = "429" in lowered and "session" in lowered and "usage limit" in lowered
    if session_limit:
        return {"category": "session_usage_limit", "pause_immediately": True}
    if "timeout" in lowered or any(code in lowered for code in ("500", "502", "503", "504")):
        return {"category": "transient", "pause_immediately": False}
    if any(code in lowered for code in ("400", "401", "403", "404", "409", "422", "429")):
        return {"category": "permanent_4xx", "pause_immediately": False}
    return {"category": "execution", "pause_immediately": False}


def build_run_state(
    *,
    run_id: str,
    fixture: list[dict],
    presets: list[str],
    fixture_digest: str,
    corpus_digest: str,
    policy_digests: dict[str, str],
    harness_id: str,
    harness_digest: str,
    code_digest: str,
) -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "version": 3,
        "run_id": run_id,
        "fixture_digest": fixture_digest,
        "corpus_digest": corpus_digest,
        "policy_digests": policy_digests,
        "harness_id": harness_id,
        "harness_digest": harness_digest,
        "code_digest": code_digest,
        "schedule": paired_schedule(fixture, presets),
        "successful_cases": [],
        "failed_attempts": [],
        "provider_request_attempts": [],
        "pause_reason": "",
        "status": "READY",
        "created_at": now,
        "updated_at": now,
    }


def save_run_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_run_state(path: Path, expected: dict, *, resume: bool) -> dict:
    if not path.exists():
        if resume:
            raise ValueError(f"resume state not found: {path}")
        return expected
    if not resume:
        raise ValueError(f"run state already exists: {path}; use --resume")
    current = json.loads(path.read_text(encoding="utf-8"))
    drift = [key for key in _RUN_CONTRACT_KEYS if current.get(key) != expected.get(key)]
    if drift:
        raise ValueError(f"resume contract drift: {', '.join(drift)}")
    return current


def record_run_attempt(
    state: dict,
    case: dict,
    *,
    status: str,
    errors: list,
    provider_error: dict | None = None,
    latency_s: float | None = None,
    provider_requests: int = 0,
) -> None:
    case_key = case["case_key"]
    record_provider_requests(state, case, provider_requests)
    if status == "SUCCESS":
        if case_key not in state["successful_cases"]:
            state["successful_cases"].append(case_key)
        return
    state["failed_attempts"].append({
        "case_key": case_key,
        "question_id": case["question_id"],
        "preset": case["preset"],
        "errors": list(errors),
        "provider_error": provider_error or classify_provider_error(errors),
        "latency_s": latency_s,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def local_budget_date() -> str:
    return time.strftime("%Y-%m-%d")


def record_provider_requests(state: dict, case: dict, count: int) -> None:
    state.setdefault("provider_request_attempts", []).append({
        "case_key": case["case_key"],
        "question_id": case["question_id"],
        "preset": case["preset"],
        "count": max(0, int(count)),
        "date": local_budget_date(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def provider_requests_used(state: dict, date: str | None = None) -> int:
    target = date or local_budget_date()
    return sum(
        int(attempt.get("count", 0))
        for attempt in state.get("provider_request_attempts", [])
        if attempt.get("date") == target
    )


def profile_provider_key(profile_name: str) -> str | None:
    from core.llm.config import config_instance

    profile = config_instance.profiles.get(profile_name)
    if profile is None:
        raise ValueError(f"unknown model profile: {profile_name}")
    if profile.provider == "ollama":
        return None
    return f"{profile.provider}:{profile.model_name}"


def provider_request_reservation(context, profile_keys: dict[str, str | None]) -> dict[str, int]:
    policy = context.policy
    planner_attempts = (
        (context.harness.max_tool_calls + 1)
        * (policy.schema_repair_attempts + 1)
        * (policy.model_transport_retries + 1)
    )
    answer_attempts = policy.answer_transport_retries + 1
    requirements: dict[str, int] = {}
    for profile_name, attempts in (
        (policy.model_profile, planner_attempts),
        (policy.answer_model_profile, answer_attempts),
    ):
        provider_key = profile_keys.get(profile_name)
        if provider_key:
            requirements[provider_key] = requirements.get(provider_key, 0) + attempts
    return requirements


def combine_provider_reservations(*reservations: dict[str, int]) -> dict[str, int]:
    total: dict[str, int] = {}
    for reservation in reservations:
        for provider_key, attempts in reservation.items():
            total[provider_key] = total.get(provider_key, 0) + attempts
    return total


def provider_claim_gate(
    ledger: ProviderRequestLedger,
    profile_keys: dict[str, str | None],
    metadata: dict[str, object],
):
    def claim(profile_name: str) -> None:
        if profile_name not in profile_keys:
            raise RuntimeError(f"provider budget has no profile mapping: {profile_name}")
        provider_key = profile_keys[profile_name]
        if provider_key:
            ledger.claim(provider_key, {**metadata, "profile": profile_name})

    return claim


def pending_schedule(state: dict) -> list[dict]:
    successful = set(state["successful_cases"])
    return [case for case in state["schedule"] if case["case_key"] not in successful]


def summarize_run_state(state: dict) -> dict:
    expected_counts = {}
    for case in state["schedule"]:
        preset = case["preset"]
        expected_counts[preset] = expected_counts.get(preset, 0) + 1
    provider_errors = {}
    for attempt in state["failed_attempts"]:
        category = attempt.get("provider_error", {}).get("category", "execution")
        provider_errors[category] = provider_errors.get(category, 0) + 1
    failed_latencies = [
        attempt["latency_s"]
        for attempt in state["failed_attempts"]
        if attempt.get("latency_s") is not None
    ]
    return {
        "expected_counts": expected_counts,
        "provider_errors": provider_errors,
        "provider_requests": sum(
            int(attempt.get("count", 0))
            for attempt in state.get("provider_request_attempts", [])
        ),
        "provider_requests_today": provider_requests_used(state),
        "partial": state.get("status") != "COMPLETED",
        "median_failed_latency_s": statistics.median(failed_latencies) if failed_latencies else None,
    }


def archive_failed_trace(session_id: str, run_id: str) -> Path | None:
    source = TRACE_DIR / f"{session_id}.json"
    if not source.exists():
        return None
    attempts = TRACE_DIR / "attempts" / run_id
    attempts.mkdir(parents=True, exist_ok=True)
    destination = attempts / f"{session_id}__{time.time_ns()}.json"
    source.replace(destination)
    return destination


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


def json_digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_code_digest(root: Path) -> str:
    digest = hashlib.sha256()
    paths = []
    for folder in ("TA", "core", "knowledge", "student", "test/eval"):
        paths.extend((root / folder).rglob("*.py"))
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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


def validate_run_sessions(
    sessions,
    expected_ids: set[str],
    run_id: str,
    *,
    allow_partial: bool = False,
):
    chats = [chat for session in sessions for chat in session.chat]
    ids = [chat.question_id for chat in chats]
    if any(chat.run_id != run_id for chat in chats):
        raise ValueError(f"stale trace found outside run {run_id}")
    if any(chat.warmup for chat in chats):
        raise ValueError("warm-up trace mixed into scored cases")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate benchmark case trace")
    observed = set(ids)
    if observed - expected_ids or (not allow_partial and observed != expected_ids):
        raise ValueError(
            f"incomplete run: missing={sorted(expected_ids - observed)}, "
            f"extra={sorted(observed - expected_ids)}"
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
    provider_gate=None,
):
    from core.config import Retrieve_param
    from core.schema.retrieval import RetrievalCase, RetrievalCaseKind, RetrievalRoute
    from TA.helper.model_call import count_provider_requests, provider_request_gate

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
    outcome = None
    with count_provider_requests() as request_counter:
        with provider_request_gate(provider_gate) if provider_gate else nullcontext():
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
                outcome = {
                    "status": result.get("status", "FAIL"),
                    "errors": list(result.get("errors", [])),
                    "session_id": sid,
                    "latency_s": time.perf_counter() - t0,
                }
            except Exception as exc:
                outcome = {
                    "status": "FAIL",
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "session_id": sid,
                    "latency_s": time.perf_counter() - t0,
                }
            finally:
                tracker.drop_session(sid)
    outcome["provider_requests"] = request_counter["count"]
    label = "WARMUP" if warmup else preset
    if outcome["status"] == "SUCCESS":
        print(f"[{label}] {item['id']} ({outcome['latency_s']:.1f}s)", flush=True)
    else:
        print(f"[{label}] {item['id']} FAILED: {outcome['errors']}", flush=True)
    return outcome


async def run_resumable_schedule(
    *,
    ta,
    tracker,
    fixture: list[dict],
    track: str,
    course: str,
    run_id: str,
    harness_id,
    code_state,
    state: dict,
    state_path: Path,
    max_provider_requests: int = 0,
    provider_ledger: ProviderRequestLedger | None = None,
    provider_reservation: dict[str, int] | None = None,
    provider_case_reservations: dict[str, dict[str, int]] | None = None,
    provider_profile_keys: dict[str, str | None] | None = None,
) -> bool:
    if provider_ledger is None:
        return await _run_resumable_schedule(
            ta=ta,
            tracker=tracker,
            fixture=fixture,
            track=track,
            course=course,
            run_id=run_id,
            harness_id=harness_id,
            code_state=code_state,
            state=state,
            state_path=state_path,
            max_provider_requests=max_provider_requests,
        )
    with provider_ledger.hold_run_lock():
        return await _run_resumable_schedule(
            ta=ta,
            tracker=tracker,
            fixture=fixture,
            track=track,
            course=course,
            run_id=run_id,
            harness_id=harness_id,
            code_state=code_state,
            state=state,
            state_path=state_path,
            max_provider_requests=max_provider_requests,
            provider_ledger=provider_ledger,
            provider_reservation=provider_reservation,
            provider_case_reservations=provider_case_reservations,
            provider_profile_keys=provider_profile_keys,
        )


async def _run_resumable_schedule(
    *,
    ta,
    tracker,
    fixture: list[dict],
    track: str,
    course: str,
    run_id: str,
    harness_id,
    code_state,
    state: dict,
    state_path: Path,
    max_provider_requests: int = 0,
    provider_ledger: ProviderRequestLedger | None = None,
    provider_reservation: dict[str, int] | None = None,
    provider_case_reservations: dict[str, dict[str, int]] | None = None,
    provider_profile_keys: dict[str, str | None] | None = None,
) -> bool:
    by_id = {item["id"]: item for item in fixture}
    state.update(status="RUNNING", pause_reason="")
    save_run_state(state_path, state)
    transient_failures = 0
    pending = pending_schedule(state)
    current_question = None
    for index, case in enumerate(pending, start=1):
        if case["question_id"] != current_question:
            successful = set(state["successful_cases"])
            sibling_succeeded = any(
                scheduled["question_id"] == case["question_id"]
                and scheduled["case_key"] in successful
                for scheduled in state["schedule"]
            )
            reservation = (
                (provider_case_reservations or {}).get(case["preset"], provider_reservation or {})
                if sibling_succeeded
                else provider_reservation or {}
            )
            provider_budget_exhausted = (
                provider_ledger is not None
                and not provider_ledger.can_reserve(reservation)
            )
            legacy_budget_exhausted = (
                provider_ledger is None
                and max_provider_requests > 0
                and provider_requests_used(state) >= max_provider_requests
                and not sibling_succeeded
            )
            if provider_budget_exhausted or legacy_budget_exhausted:
                state.update(status="PAUSED", pause_reason="provider_request_budget")
                save_run_state(state_path, state)
                return False
            current_question = case["question_id"]
        print(
            f"[{harness_id}] {index}/{len(pending)} {case['question_id']} {case['preset']}",
            flush=True,
        )

        result = await run_case(
            ta=ta,
            tracker=tracker,
            preset=case["preset"],
            item=by_id[case["question_id"]],
            track=track,
            course=course,
            run_id=run_id,
            harness_id=harness_id,
            code_state=code_state,
            provider_gate=(
                provider_claim_gate(
                    provider_ledger,
                    provider_profile_keys or {},
                    {
                        "run_id": run_id,
                        "case_key": case["case_key"],
                        "question_id": case["question_id"],
                        "preset": case["preset"],
                    },
                )
                if provider_ledger
                else None
            ),
        )
        provider_error = None
        if result["status"] != "SUCCESS":
            provider_error = classify_provider_error(result["errors"])
            archive_failed_trace(result["session_id"], run_id)
        record_run_attempt(
            state,
            case,
            status=result["status"],
            errors=result["errors"],
            provider_error=provider_error,
            latency_s=result.get("latency_s"),
            provider_requests=result.get("provider_requests", 0),
        )
        save_run_state(state_path, state)

        category = provider_error["category"] if provider_error else ""
        transient_failures = transient_failures + 1 if category == "transient" else 0
        if provider_error and provider_error["pause_immediately"]:
            state.update(status="PAUSED", pause_reason=category)
            save_run_state(state_path, state)
            return False
        if transient_failures >= 3:
            state.update(status="PAUSED", pause_reason="three_consecutive_transient_failures")
            save_run_state(state_path, state)
            return False

    all_succeeded = len(state["successful_cases"]) == len(state["schedule"])
    state.update(
        status="COMPLETED" if all_succeeded else "PARTIAL",
        pause_reason="" if all_succeeded else "incomplete_cases",
    )
    save_run_state(state_path, state)
    return True


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
    partial: bool = False,
    run_states: dict[str, dict] | None = None,
) -> None:
    from TA.tracing.evaluator import (
        evaluate_session,
        judge_chat,
        load_session,
        render_paired_report,
        render_report,
    )

    rows = []
    run_states = run_states or {}
    report_meta = {}
    expected_ids = {item["id"] for item in fixture}
    for harness, run_id in harness_runs.items():
        state_summary = summarize_run_state(run_states[harness]) if harness in run_states else {
            "expected_counts": {preset: len(fixture) for preset in presets},
            "provider_errors": {},
            "provider_requests": 0,
            "provider_requests_today": 0,
            "partial": partial,
            "median_failed_latency_s": None,
        }
        report_meta[harness] = state_summary
        for preset in presets:
            paths = sorted(TRACE_DIR.glob(f"{session_name(preset, track, run_id)}__*.json"))
            if not paths:
                if state_summary["partial"]:
                    continue
                raise ValueError(f"no traces for {harness}/{preset} in run {run_id}")
            sessions = [load_session(path) for path in paths]
            validate_run_sessions(
                sessions,
                expected_ids,
                run_id,
                allow_partial=state_summary["partial"],
            )
            for session in sessions:
                harness_rows = evaluate_session(session, fixture)
                for row in harness_rows:
                    row["harness_id"] = row.get("harness_id") or harness
                rows += harness_rows

    if judge:
        for row in rows:
            if row.get("status") != "SUCCESS" or row.get("retrieval_validity") != "valid":
                continue
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
        meta = report_meta[harness]
        sections += [
            f"# Harness: {harness}",
            "",
            (
                f"Provider requests (retries included): {meta['provider_requests']} total; "
                f"{meta['provider_requests_today']} today."
            ),
            "",
            render_report(
                group,
                partial=meta["partial"],
                expected_counts=meta["expected_counts"],
                provider_errors=meta["provider_errors"],
                median_failed_latency_s=meta["median_failed_latency_s"],
            ),
            "",
        ]
    if {"agentic-v3", "agentic-v4"} <= set(harness_runs):
        sections += [
            render_paired_report(rows, baseline="agentic-v3", treatment="agentic-v4"),
            "",
        ]
    elif {"agentic-v2", "agentic-v3"} <= set(harness_runs):
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
        default="agentic-v4",
        choices=("agentic-v1", "agentic-v2", "agentic-v3", "agentic-v4", "fanout-v1"),
    )
    ap.add_argument(
        "--harnesses",
        default="",
        help="comma-separated harnesses for a paired run, e.g. agentic-v2,agentic-v3",
    )
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--ids", default="", help="comma-separated fixture IDs to run")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--max-provider-requests",
        type=int,
        default=500,
        help="hard daily external-model request budget (default: 500); pauses before an unreservable pair",
    )
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
    allowed_harnesses = {"agentic-v1", "agentic-v2", "agentic-v3", "agentic-v4", "fanout-v1"}
    unknown_harnesses = set(harness_names) - allowed_harnesses
    if unknown_harnesses:
        raise SystemExit(f"unknown harnesses: {sorted(unknown_harnesses)}")
    if args.max_provider_requests and set(harness_names) != {"agentic-v4"}:
        raise SystemExit("--max-provider-requests currently supports agentic-v4 only")
    harness_runs = harness_run_ids(run_id, harness_names)

    if args.calibrate:
        calibrate(fixture, Path(args.corpus))
        return
    if args.resume and not args.run_id:
        raise SystemExit("--resume requires the exact --run-id")

    all_complete = True
    run_states = {}
    provider_ledger = (
        ProviderRequestLedger(
            RESULTS_DIR / "provider_request_ledger",
            max_requests=args.max_provider_requests,
        )
        if args.max_provider_requests
        else None
    )
    if not args.score_only:
        from core.config import Retrieve_param
        from core.schema.retrieval import RetrievalHarnessId
        from TA.retrieval.policy import resolve_retrieval_context

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
        source_digest = source_code_digest(Path(__file__).resolve().parents[2])
        fixture_digest = json_digest(fixture)
        corpus_digest = hashlib.sha256(
            (Path(args.corpus) / "manifest.json").read_bytes()
        ).hexdigest()
        try:
            for harness_name, harness_run_id in harness_runs.items():
                harness_id = RetrievalHarnessId(harness_name)
                contexts = {
                    preset: resolve_retrieval_context(Retrieve_param.from_preset(
                        preset,
                        policy_id=harness_policy_id(harness_id),
                        harness_id=harness_id,
                        course_scope=args.course,
                    ))
                    for preset in presets
                }
                harness_digests = {context.harness.digest for context in contexts.values()}
                if len(harness_digests) != 1:
                    raise RuntimeError(f"harness digest differs across arms: {harness_name}")
                profile_names = {
                    profile_name
                    for context in contexts.values()
                    for profile_name in (
                        context.policy.model_profile,
                        context.policy.answer_model_profile,
                    )
                }
                provider_profile_keys = {
                    profile_name: profile_provider_key(profile_name)
                    for profile_name in profile_names
                }
                case_reservations = {
                    preset: provider_request_reservation(context, provider_profile_keys)
                    for preset, context in contexts.items()
                }
                pair_reservation = combine_provider_reservations(*case_reservations.values())
                expected_state = build_run_state(
                    run_id=harness_run_id,
                    fixture=fixture,
                    presets=presets,
                    fixture_digest=fixture_digest,
                    corpus_digest=corpus_digest,
                    policy_digests={preset: context.policy.digest for preset, context in contexts.items()},
                    harness_id=harness_name,
                    harness_digest=next(iter(harness_digests)),
                    code_digest=source_digest,
                )
                state_path = RESULTS_DIR / f"run_state_{harness_run_id}.json"
                state = load_run_state(state_path, expected_state, resume=args.resume)
                run_states[harness_name] = state
                if fixture and not args.resume:
                    warmup_preset = "FULL" if "FULL" in contexts else presets[0]
                    warmup_reservation = provider_request_reservation(
                        contexts[warmup_preset], provider_profile_keys
                    )
                    if provider_ledger and not provider_ledger.can_reserve(
                        combine_provider_reservations(warmup_reservation, pair_reservation)
                    ):
                        state.update(status="PAUSED", pause_reason="provider_request_budget")
                        save_run_state(state_path, state)
                        all_complete = False
                        break
                    warmup_result = await run_case(
                        ta,
                        tracker,
                        warmup_preset,
                        fixture[0],
                        track,
                        args.course,
                        harness_run_id,
                        harness_id,
                        code_state,
                        warmup=True,
                        provider_gate=(
                            provider_claim_gate(
                                provider_ledger,
                                provider_profile_keys,
                                {
                                    "run_id": harness_run_id,
                                    "case_key": "__warmup__",
                                    "question_id": fixture[0]["id"],
                                    "preset": "WARMUP",
                                },
                            )
                            if provider_ledger
                            else None
                        ),
                    )
                    record_provider_requests(
                        state,
                        {
                            "case_key": "__warmup__",
                            "question_id": fixture[0]["id"],
                            "preset": "WARMUP",
                        },
                        warmup_result.get("provider_requests", 0),
                    )
                    if warmup_result["status"] != "SUCCESS":
                        provider_error = classify_provider_error(warmup_result["errors"])
                        archive_failed_trace(warmup_result["session_id"], harness_run_id)
                        state["failed_attempts"].append({
                            "case_key": "__warmup__",
                            "question_id": fixture[0]["id"],
                            "preset": "WARMUP",
                            "errors": warmup_result["errors"],
                            "provider_error": provider_error,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        })
                        save_run_state(state_path, state)
                        if provider_error["pause_immediately"]:
                            state.update(status="PAUSED", pause_reason=provider_error["category"])
                            save_run_state(state_path, state)
                            all_complete = False
                            break
                complete = await run_resumable_schedule(
                    ta=ta,
                    tracker=tracker,
                    fixture=fixture,
                    track=track,
                    course=args.course,
                    run_id=harness_run_id,
                    harness_id=harness_id,
                    code_state=code_state,
                    state=state,
                    state_path=state_path,
                    max_provider_requests=args.max_provider_requests,
                    provider_ledger=provider_ledger,
                    provider_reservation=pair_reservation,
                    provider_case_reservations=case_reservations,
                    provider_profile_keys=provider_profile_keys,
                )
                if not complete:
                    all_complete = False
                    print(
                        f"run paused: {state_path} ({state.get('pause_reason', 'provider failure')})",
                        flush=True,
                    )
                    break
                if state.get("status") != "COMPLETED":
                    all_complete = False
        finally:
            tracker.delete_student(BENCH_STUDENT)
    elif not args.run_id:
        raise SystemExit("--score-only requires the exact --run-id")
    else:
        for harness_name, harness_run_id in harness_runs.items():
            state_path = RESULTS_DIR / f"run_state_{harness_run_id}.json"
            if state_path.exists():
                run_states[harness_name] = json.loads(state_path.read_text(encoding="utf-8"))
                all_complete = all_complete and run_states[harness_name].get("status") == "COMPLETED"

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
        partial=not all_complete,
        run_states=run_states,
    )


if __name__ == "__main__":
    asyncio.run(main())
