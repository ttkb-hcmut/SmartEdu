# QA Benchmark Fixtures — retrieval ablation

Offline fixtures + corpus files for the PLAIN → RAG → FULL retrieval ablation.
Built by the `build_*.py` scripts here; consumed by `run_ablation.py` and
`TA/tracing/evaluator.py`.

## Tracks

| Track | Fixture | Corpus | Source |
|---|---|---|---|
| 1 | `fixtures/musique_cs.json` | `corpus/musique_cs/` | MuSiQue (CC BY-SA 4.0), CS/ML/math-filtered multi-hop slice |
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
  "gold_chunk_ids": ["Bench_MuSiQue/2hop__123_456/3", "..."],
  "hops": 2,
  "type": "open"
}
```

## Gold-chunk ID contract (for the ingestion session)

Context precision/recall is **set arithmetic** between trace `chunks[].uri`
and `gold_chunk_ids`. That only works if ingestion preserves the URI scheme:

- Corpus layout: `corpus/<track>/<question_id>/<para_idx>.txt`
- Required passage URI after ingest: `Bench_MuSiQue/<question_id>/<para_idx>`
  (resp. `Bench_NCERT/<chapter>/<para_idx>`) — i.e. course name
  `Bench_MuSiQue` / `Bench_NCERT`, then the relative path without extension.
- Each `.txt` starts with a `# <title>` line; body is the paragraph text.
  `manifest.json` in each corpus dir maps file → expected URI.
- Ingest each track as its own course so `Retrieve_param.benchmark_course`
  scopes retrieval to it (URI prefix filter on graph side, `community` field
  on Milvus side).
- Distractor paragraphs are included on purpose — they are part of the
  benchmark (retriever must rank gold above same-question distractors).

## Verify a build

```
uv run python test/eval/build_musique.py --limit 30
uv run python -c "import json; d=json.load(open('test/eval/fixtures/musique_cs.json')); print(len(d), 'questions')"
```
