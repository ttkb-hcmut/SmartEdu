"""Create an evidence directory and immutable run manifest for manual inspection."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run_id(label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in label)
    return f"{stamp}_{safe or 'inspection'}"


def create_run(run_id: str) -> Path:
    out = RESULTS / run_id
    if out.exists():
        raise FileExistsError(f"inspection run already exists: {out}")

    (out / "screenshots").mkdir(parents=True)
    (out / "ingest").mkdir()
    (out / "qa").mkdir()

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": "single-host-musique-characterization",
        "repository": str(ROOT),
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "artifacts": {
            "ingestion_course": "Inspect_MuSiQue_<run_id>",
            "benchmark_course": "Bench_MuSiQue",
            "synthetic_student_prefix": "inspect_<run_id>",
        },
        "thresholds": "characterization-only; human review required",
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (out / "human_observations.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(
            ["scenario_id", "action", "expected", "actual", "classification", "screenshot", "notes"]
        )
    (out / "api_events.jsonl").touch()
    (out / "resource_samples.csv").write_text(
        "timestamp_utc,source,cpu_percent,memory_mb,gpu_percent,vram_mb,notes\n",
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--label", default="inspection")
    args = parser.parse_args()
    out = create_run(args.run_id or _run_id(args.label))
    print(out)


if __name__ == "__main__":
    main()
