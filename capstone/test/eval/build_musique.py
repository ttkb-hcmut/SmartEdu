"""Build track-1 fixture: CS/ML/math-filtered MuSiQue slice.

Pulls rows from the HF datasets-server REST API (no `datasets` dep),
keyword-filters to STEM questions, writes fixture JSON + corpus dir
following the gold-chunk URI contract in README.md.

Usage (from capstone/):
    uv run python test/eval/build_musique.py --limit 30
"""

import argparse
import json
import re
from pathlib import Path

import requests

API = "https://datasets-server.huggingface.co/rows"
DATASET = "dgslibisey/MuSiQue"
COURSE = "Bench_MuSiQue"

STEM_TERMS = [
    "algorithm", "computer", "software", "programming", "matrix", "vector",
    "neural", "machine learning", "artificial intelligence", "database",
    "network", "mathemat", "calculus", "algebra", "probability", "statistic",
    "physics", "theorem", "derivative", "compiler", "binary", "encryption",
    "processor", "operating system", "internet", "data",
]
_STEM_RE = re.compile("|".join(re.escape(t) for t in STEM_TERMS), re.I)


def is_stem(row: dict) -> bool:
    text = row["question"] + " " + " ".join(p["title"] for p in row["paragraphs"])
    return bool(_STEM_RE.search(text))


def fetch_rows(split: str, offset: int, length: int = 100) -> list:
    r = requests.get(API, params={
        "dataset": DATASET, "config": "default",
        "split": split, "offset": offset, "length": length,
    }, timeout=60)
    r.raise_for_status()
    return [x["row"] for x in r.json()["rows"]]


def hops_of(qid: str) -> int:
    m = re.match(r"(\d)hop", qid)
    return int(m.group(1)) if m else 2


def build(limit: int, split: str, out_dir: Path) -> None:
    fixtures_dir = out_dir / "fixtures"
    corpus_dir = out_dir / "corpus" / "musique_cs"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    fixture, manifest = [], {}
    offset = 0
    while len(fixture) < limit:
        rows = fetch_rows(split, offset)
        if not rows:
            break
        offset += len(rows)
        for row in rows:
            if len(fixture) >= limit:
                break
            if not row.get("answerable", True) or not is_stem(row):
                continue
            qid = row["id"]
            gold_ids = []
            qdir = corpus_dir / qid
            qdir.mkdir(exist_ok=True)
            for p in row["paragraphs"]:
                idx = p["idx"]
                uri = f"{COURSE}/{qid}/{idx}"
                path = qdir / f"{idx}.txt"
                path.write_text(f"# {p['title']}\n{p['paragraph_text']}\n", encoding="utf-8")
                manifest[str(path.relative_to(corpus_dir))] = uri
                if p["is_supporting"]:
                    gold_ids.append(uri)
            fixture.append({
                "id": qid,
                "track": "musique",
                "question": row["question"],
                "gold_answer": row["answer"],
                "gold_chunk_ids": gold_ids,
                "hops": hops_of(qid),
                "type": "open",
            })
        print(f"scanned {offset} rows -> {len(fixture)} kept")

    (fixtures_dir / "musique_cs.json").write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    n_para = len(manifest)
    print(f"wrote {len(fixture)} questions, {n_para} corpus paragraphs -> {corpus_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default="test/eval")
    args = ap.parse_args()
    build(args.limit, args.split, Path(args.out))
