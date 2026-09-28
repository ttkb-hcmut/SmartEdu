import json
from pathlib import Path

from test.eval import build_musique
from test.eval.build_musique import (
    canonical_uri,
    canonicalize_existing,
    reservoir_sample,
    select_stem_population,
)
from test.eval import load_corpus


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


def test_canonicalize_existing_can_change_scope_without_changing_identity(tmp_path):
    corpus = tmp_path / "corpus"
    path = corpus / "q1" / "2.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Same title\nSame paragraph.\n", encoding="utf-8")
    old_uri = build_musique.canonical_uri("Same title", "Same paragraph.")
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps([{"id": "q1", "gold_chunk_ids": [old_uri]}]),
        encoding="utf-8",
    )

    manifest = build_musique.canonicalize_existing(
        corpus, fixture_path, course="Bench_MuSiQue_500"
    )

    new_uri = next(iter(manifest["paragraphs"]))
    assert new_uri.startswith("Bench_MuSiQue_500/")
    assert new_uri.removeprefix("Bench_MuSiQue_500/") == old_uri.removeprefix("Bench_MuSiQue/")
    assert json.loads(fixture_path.read_text(encoding="utf-8"))[0]["gold_chunk_ids"] == [new_uri]


def test_ingestion_report_records_failure_timing(tmp_path, monkeypatch):
    monkeypatch.setattr(load_corpus, "RESULTS_DIR", tmp_path)

    path = load_corpus.write_report(
        "Bench", Path("corpus"), 10, {"store_boot_ms": 5.0, "total_ms": 8.0},
        "FAILED", "ConnectionError: unavailable",
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["status"] == "FAILED"
    assert report["timings"]["store_boot_ms"] == 5.0
    assert report["paragraphs_per_second"] is None


def test_stem_population_reservoir_is_reproducible_and_uses_all_when_small():
    rows = [{"id": str(index)} for index in range(100)]

    first = reservoir_sample(rows, limit=10, seed=42)
    second = reservoir_sample(rows, limit=10, seed=42)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 10
    assert reservoir_sample(rows[:3], limit=10, seed=42) == rows[:3]


def test_population_selection_filters_unanswerable_and_non_stem_rows():
    rows = [
        {"id": "keep", "answerable": True, "question": "Which computer algorithm?", "paragraphs": []},
        {"id": "wrong-domain", "answerable": True, "question": "Who painted this?", "paragraphs": []},
        {"id": "unanswerable", "answerable": False, "question": "Which database?", "paragraphs": []},
    ]

    selected = select_stem_population(rows, limit=10, seed=42)

    assert [row["id"] for row in selected] == ["keep"]


def test_fetch_rows_retries_rate_limit_with_bounded_backoff(monkeypatch):
    class Response:
        def __init__(self, status_code, payload=None, headers=None):
            self.status_code = status_code
            self._payload = payload or {"rows": []}
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError(f"unexpected terminal status {self.status_code}")

        def json(self):
            return self._payload

    responses = iter([
        Response(429),
        Response(503),
        Response(200, {"rows": [{"row": {"id": "q1"}}]}),
    ])
    sleeps = []
    monkeypatch.setattr(build_musique.requests, "get", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(build_musique.time, "sleep", sleeps.append)

    assert build_musique.fetch_rows("train", 0) == [{"id": "q1"}]
    assert sleeps == [1.0, 2.0]


def test_fetch_rows_honors_retry_after_header(monkeypatch):
    class Response:
        status_code = 429
        headers = {"Retry-After": "7"}

        def raise_for_status(self):
            raise AssertionError("retry should continue")

    class Success:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"rows": []}

    responses = iter([Response(), Success()])
    sleeps = []
    monkeypatch.setattr(build_musique.requests, "get", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(build_musique.time, "sleep", sleeps.append)

    assert build_musique.fetch_rows("validation", 0) == []
    assert sleeps == [7.0]


def test_iter_dataset_rows_paces_pages(monkeypatch):
    pages = iter([[{"id": "q1"}], []])
    sleeps = []
    monkeypatch.setattr(build_musique, "fetch_rows", lambda *_args: next(pages))
    monkeypatch.setattr(build_musique.time, "sleep", sleeps.append)

    assert list(build_musique.iter_dataset_rows(["train"])) == [{"id": "q1"}]
    assert sleeps == [build_musique._PAGE_DELAY]
