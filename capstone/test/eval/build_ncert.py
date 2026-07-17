"""Build track-2 fixture skeleton: NCERT Biology chapters.

Input: a dir of chapter .txt/.md files (user-extracted from NCERT PDFs).
Output: corpus passages under the URI contract + a DRAFT fixture whose
question/gold_answer fields are empty — filled by LLM drafting then a
human verify pass before the file is renamed ncert_bio.json.

Usage (from capstone/):
    uv run python test/eval/build_ncert.py --src data/ncert_bio/
"""

import argparse
import json
import re
from pathlib import Path

COURSE = "Bench_NCERT"
MIN_CHARS = 300   ## short blocks merge forward, sub-question fragments alone are unanswerable


def split_passages(text: str) -> list:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    passages, buf = [], ""
    for b in blocks:
        buf = f"{buf}\n\n{b}".strip() if buf else b
        if len(buf) >= MIN_CHARS:
            passages.append(buf)
            buf = ""
    if buf:
        if passages:
            passages[-1] += "\n\n" + buf
        else:
            passages.append(buf)
    return passages


def build(src: Path, out_dir: Path) -> None:
    fixtures_dir = out_dir / "fixtures"
    corpus_dir = out_dir / "corpus" / "ncert_bio"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    draft, manifest = [], {}
    chapters = sorted(list(src.glob("*.txt")) + list(src.glob("*.md")))
    if not chapters:
        raise SystemExit(f"no chapter .txt/.md files in {src}")

    for ch_file in chapters:
        chapter = ch_file.stem
        ch_dir = corpus_dir / chapter
        ch_dir.mkdir(exist_ok=True)
        for i, passage in enumerate(split_passages(ch_file.read_text(encoding="utf-8"))):
            uri = f"{COURSE}/{chapter}/{i}"
            path = ch_dir / f"{i}.txt"
            path.write_text(f"# {chapter} p{i}\n{passage}\n", encoding="utf-8")
            manifest[str(path.relative_to(corpus_dir))] = uri
            draft.append({
                "id": f"ncert_{chapter}_{i}",
                "track": "ncert",
                "question": "",
                "gold_answer": "",
                "gold_chunk_ids": [uri],
                "hops": 1,
                "type": "open",
                "_passage_preview": passage[:200],
            })

    (fixtures_dir / "ncert_bio.draft.json").write_text(
        json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(draft)} draft entries from {len(chapters)} chapters -> {corpus_dir}")
    print("next: draft questions per passage, human-verify, drop _passage_preview, save as ncert_bio.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="test/eval")
    args = ap.parse_args()
    build(Path(args.src), Path(args.out))
