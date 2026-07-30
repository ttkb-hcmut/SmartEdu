import json

from test.eval.build_musique import canonical_uri, canonicalize_existing


def test_canonical_uri_uses_normalized_content_not_occurrence():
    first = canonical_uri("  Hewlett   Packard ", "Merged\nwith Compaq in 2002.")
    second = canonical_uri("Hewlett Packard", "Merged with Compaq in 2002.")

    assert first == second


def test_canonical_manifest_preserves_duplicate_occurrences(tmp_path):
    corpus = tmp_path / "corpus"
    for qid, idx in (("q1", "2"), ("q2", "7")):
        path = corpus / qid / f"{idx}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Same title\nSame paragraph.\n", encoding="utf-8")
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            [
                {"id": "q1", "gold_chunk_ids": ["Bench_MuSiQue/q1/2"]},
                {"id": "q2", "gold_chunk_ids": ["Bench_MuSiQue/q2/7"]},
            ]
        ),
        encoding="utf-8",
    )

    manifest = canonicalize_existing(corpus, fixture_path)

    assert len(manifest["paragraphs"]) == 1
    paragraph = next(iter(manifest["paragraphs"].values()))
    assert len(paragraph["occurrences"]) == 2
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture[0]["gold_chunk_ids"] == fixture[1]["gold_chunk_ids"]
