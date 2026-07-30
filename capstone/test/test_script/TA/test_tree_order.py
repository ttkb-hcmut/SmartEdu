from TA.helper.tree_order import topo_order


def test_prerequisite_comes_left():
    names = ["B", "A"]
    edges = [("A", "B")]
    assert topo_order(names, edges, {"A": 1.0, "B": 9.0}) == ["A", "B"]


def test_tiebreak_by_score_desc():
    names = ["low", "high", "mid"]
    assert topo_order(names, [], {"low": 1.0, "high": 3.0, "mid": 2.0}) == ["high", "mid", "low"]


def test_cycle_does_not_crash_and_keeps_all_nodes():
    names = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "A")]
    out = topo_order(names, edges, {"A": 2.0, "B": 1.0, "C": 3.0})
    assert sorted(out) == ["A", "B", "C"]
    assert out[0] == "C"


def test_edges_outside_group_ignored():
    out = topo_order(["A", "B"], [("X", "A"), ("A", "B")], {"A": 1.0, "B": 1.0})
    assert out == ["A", "B"]
