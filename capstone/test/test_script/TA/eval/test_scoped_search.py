from core.repo.graph.cypher.kg_const.bind import (
    CYPHER_anchor_search,
    CYPHER_anchor_search_scoped,
    CYPHER_passage_search_vec,
    CYPHER_passage_search_vec_scoped,
)
from core.repo.graph.graphdb import GraphDB
from core.repo.milvus_db.mil import MilvusDB


def _graph():
    graph = object.__new__(GraphDB)
    graph.db_name = "neo4j"
    calls = []

    def run_query(db_name, query, params=None):
        calls.append((db_name, query, params))
        return []

    graph.run_query = run_query
    return graph, calls


def test_scoped_passage_search_uses_exact_query_without_probe():
    graph, calls = _graph()

    graph.passage_search([0.1], top_k=5, uri_prefix="Bench/")

    assert calls[0][1] == CYPHER_passage_search_vec_scoped
    assert calls[0][2] == {"k": 5, "emb": [0.1], "prefix": "Bench/"}
    assert "probe" not in calls[0][2]


def test_unscoped_passage_search_keeps_ann_query():
    graph, calls = _graph()

    graph.passage_search([0.1], top_k=5)

    assert calls[0][1] == CYPHER_passage_search_vec
    assert calls[0][2]["probe"] == 5


def test_scoped_anchor_search_uses_exact_query():
    graph, calls = _graph()

    graph.anchor_search([0.1], top_k=5, uri_prefix="Bench/")

    assert calls[0][1] == CYPHER_anchor_search_scoped
    assert calls[0][2] == {"k": 5, "emb": [0.1], "prefix": "Bench/"}


def test_unscoped_anchor_search_keeps_ann_query():
    graph, calls = _graph()

    graph.anchor_search([0.1], top_k=5)

    assert calls[0][1] == CYPHER_anchor_search


def test_passage_repository_errors_are_not_empty_hits(monkeypatch):
    class FakeClientError(Exception):
        pass

    graph = object.__new__(GraphDB)
    graph.db_name = "neo4j"
    graph.run_query = lambda *args, **kwargs: (_ for _ in ()).throw(FakeClientError("down"))
    monkeypatch.setattr(graphdb_module, "ClientError", FakeClientError)

    with pytest.raises(FakeClientError):
        graph.passage_search([0.1], query_text="q", uri_prefix="Bench/")


def test_milvus_repository_owns_course_expression():
    db = object.__new__(MilvusDB)
    calls = []
    db.search_vec = lambda vector, top_k=5, expr=None: calls.append((vector, top_k, expr)) or []
    embedder = SimpleEmbedder()

    db.search("q", embedder, top_k=3, course_scope='Bench "A"')

    assert calls == [([0.3], 3, 'community == "Bench \\"A\\""')]


def test_milvus_insert_stamps_explicit_course_scope():
    db = object.__new__(MilvusDB)
    db.collection = type(
        "Collection",
        (),
        {"insert": lambda self, rows: setattr(self, "rows", rows), "flush": lambda self: None},
    )()

    db.insert_data(
        [{"id": "p1", "name": "T", "content": "text", "rrole": "Passage"}],
        SimpleEmbedder(),
        community="Bench",
    )

    assert db.collection.rows[0]["community"] == "Bench"


class SimpleEmbedder:
    def get_embedding(self, text):
        return [0.3]
import pytest

import core.repo.graph.graphdb as graphdb_module
