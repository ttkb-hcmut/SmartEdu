# MuSiQue inspection recorder

This directory contains system-level inspection helpers. They are not the product runtime and are not unit-test fixtures.

Start a run from `capstone/`:

```powershell
uv run python test/inspection/record_run.py --label musique-single-host
```

The command creates an ignored result directory under `test/inspection/results/` and prints its absolute path. Keep that run ID when naming screenshots, browser exports, database snapshots, traces, and benchmark outputs.

## Evidence layout

```text
results/<run_id>/
├── run_manifest.json       # environment and source identity
├── human_observations.csv  # operator's expected/actual record
├── api_events.jsonl        # request and SSE timestamps
├── resource_samples.csv   # CPU/RAM/GPU/container samples
├── screenshots/            # numbered UI and database evidence
├── ingest/                 # reports, Prefect exports, store snapshots
└── qa/                     # traces, benchmark rows, tables, charts
```

Do not put secrets, bearer tokens, signed object URLs, passwords, or raw stack traces in the evidence directory. Redact before sharing screenshots or archives.

The first run is characterization: record measurements and observations without inventing performance thresholds. Hard integrity failures remain failures immediately.
