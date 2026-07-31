# TA — Teaching Assistant module

The agentic layer. A chat turn enters at `ta_module.py`, gets routed to one of five
intents, and every step assembles its agent on the fly from a registry keyed by node
name. Nothing here touches a database directly — agents work through tools that are
pre-bound to the student's session.

## Layout

| Path | What lives there |
|---|---|
| `ta_module.py` | Module entry point, DI wiring, the retrieve path benchmarks run against |
| `workflow/` | The graphs: `smart_edu.py` (router), `retrieve.py`, `roadmap.py`, `teach.py` |
| `agent/` | Per-step agent assembly — `injector.py`, `middleware.py`, `base.py` |
| `retrieval/` | Typed retrieval policy (`policy.py`) and its enforcing middleware |
| `tools/` | Tool adapters: `neo/`, `student/`, `minio/`, `retrieval.py`, `factory.py` |
| `helper/` | Prompts, schemas, few-shots, tree ordering/rendering |
| `tracing/` | Trace capture, writer, and the scoring `evaluator.py` |
| `api/route.py` | FastAPI surface |

Shared contracts live outside the module in `core/schema/retrieval.py` — the typed
enums and frozen dataclasses (`RetrievalPreset`, `RetrievalPolicyId`,
`RetrievalHarnessId`, `RetrievalRunContext`, `RetrievalValidation`) that both the
workflow and the benchmark runner speak.

## Workflow

`smart_edu.py` runs a fast router — one constrained token, no tools, temperature zero —
that classifies a turn into five intents, each with its own finish node:

- **retrieve** — factual answers; quick graph lookup first, deeper structural pass only on a real gap.
- **roadmap** — ranks concepts by out-degree centrality, finds the shortest prerequisite path, runs a critic-actor review, then waits for student confirmation before committing.
- **teach** — the lesson loop, a 6-node subgraph: `Teach_Understand → Teach_Lookup → [Teach_Lecture | Teach_RAG → Teach_Lecture] → Teach_Evaluate → Next_Topic`. PDF lookup first, RAG only as fallback. Evaluate either advances position or holds the concept for review.
- **confirm** — applies a pending proposal the student agreed to.
- **unknown** — anything that fits nothing else.

All five are wired end-to-end.

## Retrieval policy

Retrieval behavior is a **frozen contract in code**, not a prompt someone can drift.
`retrieval/policy.py` registers the prompt, per-preset toolset, minimum call counts, and
limits, then hashes the whole payload into `_POLICY_DIGEST`. Changing any of it means
minting a new policy ID — and a new baseline family. Old numbers stay comparable because
they name the policy that produced them.

Two axes:

- **Preset** (`plain` / `rag` / `full`) — the ablation arms, expressed as different toolsets under the *same* policy.
- **Harness** (`agentic-v1` / `fanout-v1`) — *how* retrieval executes. `agentic-v1` is the default tool loop; `fanout-v1` is a deterministic fan-out + RRF control, kept frozen. There is no silent fallback between them, and every benchmark row names the harness explicitly.

`RetrievalRunContext` is injected by the system, never model-controllable. Validity is
mechanical, not self-reported: a tool error marks the row invalid, and a minimum-call
violation marks it policy-invalid.

Rationale and rejected alternatives are recorded in `docs/adr/0008`.

## Testing

**Both the working directory and `PYTHONPATH` matter.** Run from `capstone/`, and set
`PYTHONPATH` to that same directory — there is no `conftest.py` and no pytest config, so
without it `core`, `student`, and `knowledge` fail to import and pytest collects nothing.

```bash
cd capstone
PYTHONPATH="$PWD" uv run pytest test/test_script/TA/eval/ -q
```

On Windows PowerShell:

```powershell
cd capstone
$env:PYTHONPATH = $PWD; uv run pytest test/test_script/TA/eval/ -q
```

### Current state

| Scope | Result |
|---|---|
| `test/test_script/TA/eval/` | **67 passed** |
| Full suite, 4 broken modules excluded | **103 passed, 2 failed** |

The full-suite run needs four modules excluded, all broken for reasons unrelated to the
agent work:

```bash
PYTHONPATH="$PWD" uv run pytest test/ -q \
  --ignore=test/test_script/TA/test_learning_tree.py \
  --ignore=test/test_script/TA/test_minio.py \
  --ignore=test/test_script/TA/test_mongo.py \
  --ignore=test/test_script/TA/test_reset.py
```

- `test_reset.py` — file is corrupted; literal `\n` escapes in the source, raises `SyntaxError` at collection.
- `test_learning_tree.py` — executes a Mongo delete at import time, so it needs a live database just to collect.
- `test_minio.py` / `test_mongo.py` — basename collision with the same-named files under `test/db/`; pytest cannot import both without unique names.

The 2 remaining failures (`test_concurrency`, `test_context_smoke`) are async tests with
no asyncio plugin configured — pytest reports *"async def functions are not natively
supported"*. They are not assertion failures and not regressions; installing and enabling
`pytest-asyncio` is what fixes them.

### Benchmark

The ablation runner is built and unit-tested, but **has never produced a scored run** —
`test/eval/results/` is empty.

Load the corpus into both stores first, or the pre-flight gate rejects the run:

```bash
uv run python test/eval/load_corpus.py --corpus test/eval/corpus/musique_cs \
  --course Bench_MuSiQue
```

Then:

```bash
uv run python test/eval/run_ablation.py \
  --fixture test/eval/fixtures/musique_cs.json \
  --course Bench_MuSiQue \
  --harness agentic-v1 \
  --presets plain,rag,full \
  --limit 30
```

Flags: `--corpus`, `--run-id`, `--score-only`, `--judge`, `--calibrate`.

This needs live Milvus, Neo4j, and Ollama. Scoring is set arithmetic between trace
`chunks[].uri` and the fixture's `gold_chunk_ids`, so identity has to match exactly in
both stores — `load_corpus.py` is what guarantees that. Three ways a run scores zero
while looking healthy:

- Milvus rows inserted with `course=` instead of `community=` — only `community` is filtered on.
- `:Passage` nodes written without `emb` — they pass the gate, but the scoped vector query requires `emb IS NOT NULL`, so textbook retrieval returns nothing.
- URIs recomputed from the corpus `.txt` files instead of read from `manifest.json`.

Corpus contract: `test/eval/README.md`.

Note the committed fixture holds **5 questions** (2 gold chunks each), not 30. The
documented 30×3 baseline needs `build_musique.py --limit 30` re-run first — that
regenerates fixture and corpus together, so the loader must then be re-run.

Fixture and corpus builds run offline and work today:

```bash
uv run python test/eval/build_musique.py --limit 30
uv run python test/eval/build_musique.py --canonicalize-existing
```
