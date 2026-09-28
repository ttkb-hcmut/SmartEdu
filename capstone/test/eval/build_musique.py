"""Build track-1 fixture: CS/ML/math-filtered MuSiQue slice.

Pulls rows from the HF datasets-server REST API (no `datasets` dep),
keyword-filters to STEM questions, writes fixture JSON + corpus dir
following the gold-chunk URI contract in README.md.

Usage (from capstone/):
    uv run python test/eval/build_musique.py --limit 30
"""

import argparse
import hashlib
import json
import random
import re
import time
import unicodedata
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
_FETCH_ATTEMPTS = 6
_MAX_RETRY_DELAY = 60.0
_PAGE_DELAY = 2.0


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def canonical_uri(title: str, text: str, course: str = COURSE) -> str:
    identity = f"{_normalize(title)}\0{_normalize(text)}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{course}/paragraph/{digest}"


def _new_manifest(course: str = COURSE) -> dict:
    return {
        "schema_version": "2.0",
        "identity": "sha256(normalize(title) + NUL + normalize(text))",
        "scope": course,
        "occurrences": {},
        "paragraphs": {},
    }


def _record_paragraph(
    manifest: dict,
    relative_path: str,
    question_id: str,
    paragraph_idx: int | str,
    title: str,
    text: str,
    course: str = COURSE,
) -> str:
    uri = canonical_uri(title, text, course)
    occurrence = {
        "question_id": question_id,
        "paragraph_idx": int(paragraph_idx),
        "path": relative_path,
    }
    manifest["occurrences"][relative_path] = uri
    paragraph = manifest["paragraphs"].setdefault(
        uri,
        {"title": title, "text": text, "occurrences": []},
    )
    paragraph["occurrences"].append(occurrence)
    return uri


def canonicalize_existing(
    corpus_dir: Path, fixture_path: Path, course: str = COURSE
) -> dict:
    manifest = _new_manifest(course)
    old_to_new = {}
    for path in sorted(corpus_dir.rglob("*.txt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = lines[0].removeprefix("# ").strip() if lines else ""
        text = "\n".join(lines[1:]).strip()
        relative = path.relative_to(corpus_dir).as_posix()
        question_id, paragraph_idx = path.parent.name, path.stem
        uri = _record_paragraph(
            manifest,
            relative,
            question_id,
            paragraph_idx,
            title,
            text,
            course,
        )
        old_to_new[canonical_uri(title, text)] = uri
        old_to_new[f"{COURSE}/{question_id}/{paragraph_idx}"] = uri

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    for item in fixture:
        item["gold_chunk_ids"] = [old_to_new.get(uri, uri) for uri in item["gold_chunk_ids"]]
    fixture_path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def is_stem(row: dict) -> bool:
    text = row["question"] + " " + " ".join(p["title"] for p in row["paragraphs"])
    return bool(_STEM_RE.search(text))


def reservoir_sample(rows, limit: int, seed: int) -> list:
    rng = random.Random(seed)
    sample = []
    for index, row in enumerate(rows):
        if index < limit:
            sample.append(row)
            continue
        selected = rng.randint(0, index)
        if selected < limit:
            sample[selected] = row
    return sample


def select_stem_population(rows, limit: int, seed: int) -> list:
    eligible = (
        row
        for row in rows
        if row.get("answerable", True) and is_stem(row)
    )
    return reservoir_sample(eligible, limit=limit, seed=seed)


def fetch_rows(split: str, offset: int, length: int = 100) -> list:
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": split,
        "offset": offset,
        "length": length,
    }
    for attempt in range(_FETCH_ATTEMPTS):
        response = requests.get(API, params=params, timeout=60)
        retryable = response.status_code == 429 or 500 <= response.status_code < 600
        if not retryable:
            response.raise_for_status()
            return [x["row"] for x in response.json()["rows"]]
        if attempt == _FETCH_ATTEMPTS - 1:
            response.raise_for_status()
        try:
            delay = float(response.headers.get("Retry-After", ""))
        except (TypeError, ValueError):
            delay = 2.0**attempt
        time.sleep(min(max(delay, 0.0), _MAX_RETRY_DELAY))
    raise RuntimeError("unreachable fetch retry state")


def hops_of(qid: str) -> int:
    m = re.match(r"(\d)hop", qid)
    return int(m.group(1)) if m else 2


def iter_dataset_rows(splits: list[str]):
    for split in splits:
        offset = 0
        while True:
            rows = fetch_rows(split, offset)
            if not rows:
                break
            offset += len(rows)
            print(f"scanned {split}:{offset}")
            yield from rows
            time.sleep(_PAGE_DELAY)


def build(
    limit: int,
    splits: list[str],
    out_dir: Path,
    seed: int = 42,
    name: str = "musique_cs",
    course: str = COURSE,
) -> None:
    fixtures_dir = out_dir / "fixtures"
    corpus_dir = out_dir / "corpus" / name
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    fixture, manifest = [], _new_manifest(course)
    selected = select_stem_population(iter_dataset_rows(splits), limit=limit, seed=seed)
    for row in selected:
        qid = row["id"]
        gold_ids = []
        qdir = corpus_dir / qid
        qdir.mkdir(exist_ok=True)
        for p in row["paragraphs"]:
            idx = p["idx"]
            path = qdir / f"{idx}.txt"
            path.write_text(f"# {p['title']}\n{p['paragraph_text']}\n", encoding="utf-8")
            uri = _record_paragraph(
                manifest,
                path.relative_to(corpus_dir).as_posix(),
                qid,
                idx,
                p["title"],
                p["paragraph_text"],
                course,
            )
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

    manifest["sampling"] = {
        "population": "answerable STEM-filtered MuSiQue",
        "splits": splits,
        "seed": seed,
        "target": limit,
        "actual": len(fixture),
        "scope": course,
    }

    (fixtures_dir / f"{name}.json").write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    n_para = len(manifest["paragraphs"])
    n_occurrences = len(manifest["occurrences"])
    print(
        f"wrote {len(fixture)} questions, {n_para} canonical paragraphs, "
        f"{n_occurrences} occurrences -> {corpus_dir}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--split", default="", help="legacy single-split override")
    ap.add_argument("--splits", default="train,validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="musique_cs")
    ap.add_argument("--course", default=COURSE)
    ap.add_argument("--out", default="test/eval")
    ap.add_argument("--canonicalize-existing", action="store_true")
    args = ap.parse_args()
    if args.canonicalize_existing:
        root = Path(args.out)
        canonicalize_existing(
            root / "corpus" / args.name,
            root / "fixtures" / f"{args.name}.json",
            course=args.course,
        )
    else:
        splits = [args.split] if args.split else [item.strip() for item in args.splits.split(",") if item.strip()]
        build(
            args.limit,
            splits,
            Path(args.out),
            seed=args.seed,
            name=args.name,
            course=args.course,
        )
