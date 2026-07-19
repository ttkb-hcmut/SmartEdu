from TA.tools.neo.course_tree import _assemble_tree

CONCEPTS = [
    {"name": "Vector", "topic": "Basics", "type": "Concept", "score": 9.0, "description": "d"},
    {"name": "Matrix", "topic": "Basics", "type": "Concept", "score": 7.0, "description": "d"},
    {"name": "SVD", "topic": "Decomp", "type": "Method", "score": 8.0, "description": "d"},
    {"name": "Loose", "topic": None, "type": "Concept", "score": 5.0, "description": "d"},
]
PREREQS = [("Vector", "Matrix")]


def test_tree_shape_and_order():
    tree = _assemble_tree("LA", CONCEPTS, PREREQS, max_per_topic=5, max_orphans=5)
    assert tree["course"] == "LA"
    topics = {t["name"]: t for t in tree["topics"]}
    assert [c["name"] for c in topics["Basics"]["concepts"]] == ["Vector", "Matrix"]
    assert topics["Basics"]["concepts"][1]["requires"] == ["Vector"]
    assert [c["name"] for c in tree["orphan_concepts"]] == ["Loose"]


def test_max_per_topic_caps_selection():
    many = [{"name": f"C{i}", "topic": "T", "type": "Concept", "score": float(i), "description": ""}
            for i in range(10)]
    tree = _assemble_tree("X", many, [], max_per_topic=3, max_orphans=5)
    kept = tree["topics"][0]["concepts"]
    assert len(kept) == 3 and kept[0]["name"] == "C9"
