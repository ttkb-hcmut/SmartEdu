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

```
uv run python test/eval/build_musique.py --limit 30
uv run python test/eval/build_musique.py --canonicalize-existing
uv run python -c "import json; d=json.load(open('test/eval/fixtures/musique_cs.json')); print(len(d), 'questions')"
```

## Load the corpus

The retrieval harness does not ingest. `load_corpus.py` is the dual-index loader that
writes canonical paragraph URIs to both stores, which is what `verify_corpus_ready`
gates every live run on:

```
uv run python test/eval/load_corpus.py --corpus test/eval/corpus/musique_cs \
    --course Bench_MuSiQue [--dry-run]
```

URIs are read from `manifest.json`, never recomputed from the `.txt` files — the files
hold raw text while the hash is over NFKC-normalized text, so re-hashing yields
digests that pass the gate and then score zero against the fixture.

`--course` must match the `--course` passed to `run_ablation.py`: it becomes the Milvus
`community` value and the Neo4j `uri` prefix. The loader is idempotent (Milvus upserts
on primary key, Cypher `MERGE`s), so re-running is safe.
