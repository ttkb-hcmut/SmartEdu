# QA Benchmark — retrieval ablation

This benchmark measures retrieval behavior through the production TA retrieval
path. It compares two retrieval arms:

| Arm | Retrieval behavior | Purpose |
|---|---|---|
| `RAG` | registered semantic retrieval | single-component comparison |
| `FULL` | registered semantic + textbook retrieval | combined ceiling |

The default harness is `agentic-v3`: it searches every arm-allowed source in
parallel, appends all candidates to an evidence ledger, lets the 120B
aggregator make up to four focused one-call retrieval rounds, then gives the
complete ledger and synthesis to a separate 120B no-tool answerer. RRF orders
evidence inside one round but never deletes evidence from prior rounds.
`agentic-v2` and `fanout-v1` are frozen controls, never silent fallbacks.

This is a retrieval benchmark, not a complete teaching-quality evaluation.
MuSiQue contains connected 2–4-hop questions, so it can measure corpus
coverage, retrieval ranking, trace validity, and latency. It cannot establish
that roadmap sequencing, Socratic teaching, or course pedagogy are good.

The harness does not ingest data. `load_corpus.py` is the explicit benchmark
loader that writes the same canonical paragraphs to both retrieval stores.

## Tracks

| Track | Fixture | Corpus | Source |
|---|---|---|---|
| 1 | `fixtures/musique_cs.json` | `corpus/musique_cs/` | MuSiQue (CC BY 4.0), STEM-filtered multi-hop population |
| 2 | `fixtures/ncert_bio.json` | `corpus/ncert_bio/` | NCERT Biology chapters (user-supplied text), LLM-drafted Q + human verify |
| 3 | `fixtures/own_corpus.json` | (product corpus) | MOOCCubeX recipe over our ingested passages — deferred (Iter 5) |

MOOCCubeX itself is a methodological template only: Chinese data, GPL-3.0,
no verified English subset. Nothing from it is imported.

## Fixture record

```json
{
  "id": "2hop__123_456",
  "track": "musique",
  "question": "...",
  "gold_answer": "...",
  "gold_chunk_ids": ["Bench_MuSiQue/paragraph/<sha256>", "..."],
  "hops": 2,
  "type": "open"
}
```

## Gold-chunk ID contract (for the ingestion session)

Context precision/recall is **set arithmetic** between trace `chunks[].uri`
and `gold_chunk_ids`. That only works if ingestion preserves the URI scheme:

- Corpus layout: `corpus/<track>/<question_id>/<para_idx>.txt` preserves source occurrences.
- MuSiQue paragraph identity is `sha256(normalize(title) + NUL + normalize(text))`.
  Identical paragraphs across questions therefore share one URI:
  `Bench_MuSiQue/paragraph/<sha256>`.
- Each `.txt` starts with a `# <title>` line; body is the paragraph text.
  `manifest.json` maps every occurrence to its canonical URI and preserves the
  occurrence list under the canonical paragraph record.
- Ingest each track as its own course so `Retrieve_param.course_scope`
  scopes retrieval to it (URI prefix filter on graph side, `community` field
  on Milvus side).
- Distractor paragraphs are included on purpose — they are part of the
  benchmark (retriever must rank gold above same-question distractors).

## Verify a build

Run these commands from `capstone/`. Use module form for live commands so
`capstone/` remains the Python import root.

```powershell
cd D:\Project\AI\Capstone\capstone

# reproducibly sample 30 eligible rows across the official splits
uv run python -m test.eval.build_musique `
  --limit 30 `
  --splits train,validation `
  --seed 42 `
  --name musique_cs

# verify fixture size
uv run python -c "import json; d=json.load(open('test/eval/fixtures/musique_cs.json')); print(len(d), 'questions')"

# verify the dual-index payload without contacting databases
uv run python -m test.eval.load_corpus `
  --corpus test/eval/corpus/musique_cs `
  --course Bench_MuSiQue `
  --dry-run
```

`--canonicalize-existing` is only for migrating an older occurrence-based
fixture to canonical paragraph IDs. Do not run it after a normal fresh build.

## Load the corpus

The retrieval harness does not ingest. `load_corpus.py` is the dual-index loader that
writes canonical paragraph URIs to both stores, which is what `verify_corpus_ready`
gates every live run on:

```powershell
uv run python -m test.eval.load_corpus `
  --corpus test/eval/corpus/musique_cs `
  --course Bench_MuSiQue
```

URIs are read from `manifest.json`, never recomputed from the `.txt` files — the files
hold raw text while the hash is over NFKC-normalized text, so re-hashing yields
digests that pass the gate and then score zero against the fixture.

`--course` must match the `--course` passed to `run_ablation.py`: it becomes the Milvus
`community` value and the Neo4j `uri` prefix. The loader is idempotent (Milvus upserts
on primary key, Cypher `MERGE`s), so re-running is safe.

## Live benchmark sequence

### 1. Start the data and model dependencies

The direct retrieval benchmark needs Neo4j, Milvus, MongoDB, SQL student
storage, MinIO, and Ollama. Prefect is not part of this direct benchmark path.
Start the repository services using the project setup instructions, then
confirm the configured Ollama model is available.

### 2. Load the benchmark course

```powershell
uv run python -m test.eval.load_corpus `
  --corpus test/eval/corpus/musique_cs `
  --course Bench_MuSiQue
```

The command must end with `corpus gate: PASS`. A failure here invalidates every
later score; do not run the ablation against a partial index. The loader writes
`test/eval/results/ingest_<course>_<timestamp>.json` with cold store boot,
Milvus write, Neo4j embedding/write, verification, total time, and throughput.
Failed live loads also write a report before re-raising the error.

### 3. Reproduce the known worst case

Run the same case five times on both harnesses and both arms. This measures
chain success frequency and latency instead of treating one model sample as a
result:

```powershell
$runId = "musique-worst-case-YYYYMMDD"

uv run python -m test.eval.run_ablation `
  --fixture test/eval/fixtures/musique_cs.json `
  --presets rag,full `
  --ids 2hop__42027_42004 `
  --repeat 5 `
  --course Bench_MuSiQue `
  --harnesses agentic-v2,agentic-v3 `
  --run-id $runId
```

### 4. Run the three-question mechanics smoke

Create an evidence directory first:

```powershell
uv run python test/inspection/record_run.py --label musique-benchmark-smoke
```

Then run a small live comparison:

```powershell
$runId = "musique-smoke-YYYYMMDD"

uv run python -m test.eval.run_ablation `
  --fixture test/eval/fixtures/musique_cs.json `
  --presets rag,full `
  --limit 3 `
  --course Bench_MuSiQue `
  --harnesses agentic-v2,agentic-v3 `
  --run-id $runId
```

Stop if the corpus gate fails or traces are missing. A failed case remains a
scored and reported outcome; investigate it rather than deleting it.
The smoke run must demonstrate:

- `RAG` and `FULL` use the expected typed policy and course scope;
- no model-generated tool argument contains scope or `top_k`;
- each question has one valid trace per arm;
- the warm-up trace is excluded from scoring;
- the output contains final/candidate retrieval scores, selection loss,
  workflow/turn/TTFT latency, node-time diagnostics, and token data;
- every executed trace step contains effective model, context, retry, tool,
  harness, and top-k configuration.

Inspect the generated table under `test/eval/results/` and matching traces
under `test/TA/`. Copy the table, JSONL rows, and representative traces into
the inspection run's `qa/` directory. Record any human judgment separately in
`human_observations.csv`.

### 5. Run the paired 30-question comparison

Run this only after the smoke is clean and the code revision is stable:

```powershell
$runId = "musique-baseline-YYYYMMDD"

uv run python -m test.eval.run_ablation `
  --fixture test/eval/fixtures/musique_cs.json `
  --presets rag,full `
  --limit 30 `
  --course Bench_MuSiQue `
  --harnesses agentic-v2,agentic-v3 `
  --run-id $runId
```

The report includes paired per-question V2→V3 deltas and deterministic
bootstrap 95% intervals. The runner records policy ID/digest, harness ID,
effective per-node model
configuration, monotonic workflow/turn latency, first-token latency, code
revision, dirty state, question identity, trace status, and errors. An
official result should have a clean code state. If `dirty=true`, label the run
exploratory rather than treating it as the final baseline.

### 6. Build the maximum eligible population

This does not run the expensive benchmark. It deterministically samples up to
10,000 eligible STEM rows; if fewer qualify, it keeps all and records the
actual population in the corpus manifest:

```powershell
uv run python -m test.eval.build_musique `
  --limit 10000 `
  --splits train,validation `
  --seed 42 `
  --name musique_stem_population
```

Do not execute the full population until the paired 30-case report and its
projected runtime/cost have been reviewed.

### 7. Re-score without contacting live services

If traces already exist, render the deterministic table again:

```powershell
uv run python -m test.eval.run_ablation `
  --fixture test/eval/fixtures/musique_cs.json `
  --presets rag,full `
  --score-only `
  --run-id $runId
```

Trace schema 1.3 is required for per-node configuration and true wall latency.
Older traces still score, but the new configuration and latency-distribution
sections display `-`; re-run live rather than inferring missing measurements.

Do not enable `--judge` for the first baseline. Judge-free retrieval metrics
are the integrity baseline; model-based judging is a later calibrated step.

## What to record

For every run, preserve:

- the generated `run_manifest_<run-id>.json`;
- the exact run ID and Git revision;
- corpus question/paragraph counts;
- the Markdown summary and per-question JSONL rows;
- one representative trace for each arm;
- failed requests, missing traces, or scope mismatches;
- screenshots or terminal captures showing the run and result table.

Never store bearer tokens, passwords, signed object URLs, or raw secrets in the
evidence directory.

## Interpretation rules

- Empty retrieval hits are valid; repository/tool failures are invalid.
- Failed executions remain in the report. Only missing, duplicate, stale, or
  warm-up-contaminated traces invalidate run completeness.
- `source_hit_rate` measures retrieval calls that found gold evidence;
  `candidate_recall` measures evidence found before selection;
  `selection_recall_loss` measures evidence discarded before final context.
- `support_f1`, `support_exact_match`, and `complete_chain` are paragraph-level
  evidence scores. `first_gold_round` and `complete_chain_round` show how many
  retrieval rounds were needed. V3 selection loss should be structurally zero
  because its answer context is the complete append-only ledger.
- Answer exact match and token F1 are normalized MuSiQue QA scores against the
  fixture answer. `aggregator_calls`, aggregator latency, and answer latency
  separate deterministic retrieval, agentic expansion, and answer generation.
- Use workflow latency for end-to-end agent comparison. Summed node latency is
  diagnostic only because concurrent fanout branches overlap.
- Compare arms only when fixture, corpus, course scope, model, temperature,
  policy digest, and code revision are identical.
- Keep each harness in its own table; use the paired-delta section for V2→V3
  comparison. Run `fanout-v1` only as a separately labeled deterministic control.
- Do not claim pedagogical quality from MuSiQue results alone.
