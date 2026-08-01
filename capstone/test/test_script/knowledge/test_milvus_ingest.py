import json

from core.config import Ingest_param
from core.repo.milvus_db import mil
from core.repo.milvus_db.mil import MilvusDB
from core.repo.graph.cypher.kg_const.video import CYPHER_write_segs
from core.schema.graph.type import NodeType
from knowledge.ingest import vid


class _Collection:
    def __init__(self):
        self.rows = []
        self.flushed = False

    def insert(self, rows):
        self.rows.extend(rows)

    def flush(self):
        self.flushed = True


class _Embedder:
    def get_embedding(self, _text):
        return [0.1, 0.2]


def test_reset_drops_and_recreates_collection(monkeypatch):
    db = object.__new__(MilvusDB)
    db.collection_name = "nodes"
    db.collection = object()
    replacement = object()
    dropped = []

    monkeypatch.setattr(mil.utility, "has_collection", lambda name: name == "nodes")
    monkeypatch.setattr(mil.utility, "drop_collection", dropped.append)
    monkeypatch.setattr(db, "_init_collection", lambda: replacement)

    db.reset()

    assert dropped == ["nodes"]
    assert db.collection is replacement


def test_insert_projects_concept_type_and_course():
    db = object.__new__(MilvusDB)
    db.collection = _Collection()

    db.insert_data(
        nodes=[
            {
                "id": "concept-1",
                "name": "Gradient Descent",
                "content": "Optimization by iterative updates.",
                "typeNode": NodeType.CONCEPT,
            },
            {
                "id": "definition-1",
                "name": "Learning rate",
                "content": "Step size for each update.",
                "typeNode": NodeType.RHETORICAL,
                "rrole": "Definition",
            },
            {
                "id": "topic-1",
                "name": "Optimization",
                "content": "Course topic.",
                "typeNode": NodeType.TOPIC,
            },
        ],
        embedder=_Embedder(),
        community="cluster-a",
        course="Machine Learning",
    )

    rows = db.collection.rows
    assert [row["id"] for row in rows] == ["concept-1", "definition-1"]
    assert rows[0]["rrole"] == ""
    assert rows[0]["typeNode"] == "Concept"
    assert rows[0]["course"] == "Machine Learning"
    assert rows[0]["community"] == "cluster-a"
    assert db.collection.flushed is True


def test_video_ann_filters_concepts_to_current_course(monkeypatch):
    course = 'Machine "Learning"'
    calls = []

    class _Milvus:
        def search_vec(self, vector, top_k, expr):
            calls.append({"vector": vector, "top_k": top_k, "expr": expr})
            return [{"id": "concept-1", "name": "Gradient Descent", "score": 0.8}]

    monkeypatch.setattr(
        vid,
        "group_passages",
        lambda items, embed, cfg: [{
            "id": "video-1_s0",
            "p_num": (0.0, 5.0),
            "text": "Gradient descent lowers loss.",
            "emb": [0.1, 0.2],
        }],
    )

    vid.build_vid_segs(
        _Embedder(),
        _Milvus(),
        "video-1",
        course,
        [{"text": "Gradient descent lowers loss.", "t_lo": 0.0, "t_hi": 5.0}],
        Ingest_param(),
    )

    assert calls == [{
        "vector": [0.1, 0.2],
        "top_k": Ingest_param().segment_top_k,
        "expr": f'typeNode == "Concept" and course == {json.dumps(course)}',
    }]


def test_video_anchor_profile_keeps_candidate_identity_aligned(monkeypatch):
    class _Milvus:
        def search_vec(self, _vector, top_k, expr):
            return [
                {"id": "concept-low", "name": "Low", "score": 0.70},
                {"id": "concept-high", "name": "High", "score": 0.90},
                {"id": "concept-mid", "name": "Mid", "score": 0.80},
            ]

    monkeypatch.setattr(
        vid,
        "group_passages",
        lambda items, embed, cfg: [{
            "id": "video-1_s0",
            "p_num": (0.0, 5.0),
            "text": "Optimization lecture.",
            "emb": [0.1, 0.2],
        }],
    )

    segments, links, _ = vid.build_vid_segs(
        _Embedder(),
        _Milvus(),
        "video-1",
        "Machine Learning",
        [{"text": "Optimization lecture.", "t_lo": 0.0, "t_hi": 5.0}],
        Ingest_param(),
    )

    segment = segments[0]
    assert segment["candidate_ids"] == ["concept-high", "concept-mid", "concept-low"]
    assert segment["candidate_names"] == ["High", "Mid", "Low"]
    assert segment["candidate_scores"] == [0.90, 0.80, 0.70]
    assert segment["best_score"] == 0.90
    assert segment["anchor_model"]
    assert segment["anchor_version"] == vid.ANCHOR_VERSION
    assert [link["entity_name"] for link in links] == ["High", "Mid", "Low"]


def test_video_cypher_replaces_score_only_anchor_profile():
    for field in (
        "candidate_ids",
        "candidate_names",
        "candidate_scores",
        "anchor_model",
        "anchor_version",
    ):
        assert f"n.{field}=s.{field}" in CYPHER_write_segs
    assert "REMOVE n.anchor_scores" in CYPHER_write_segs
