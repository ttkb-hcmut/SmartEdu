from TA.helper.tree_render import render_frontier

TREE = {"course": "ML", "topics": [
    {"name": "Basics", "concepts": [
        {"name": "Vector", "requires": [], "description": "v"},
        {"name": "Matrix", "requires": ["Vector"], "description": "m"}]},
    {"name": "Supervised", "concepts": [
        {"name": "LinReg", "requires": [], "description": "l"},
        {"name": "LogReg", "requires": ["LinReg"], "description": "lr"}]},
    {"name": "Deep", "concepts": [{"name": "CNN", "requires": [], "description": "c"}]},
], "orphan_concepts": [{"name": "Ethics", "requires": [], "description": "e"}]}

LEARNING = {"Vector": {"mastery": 5}, "Matrix": {"mastery": 4},
            "LinReg": {"mastery": 4}, "LogReg": {"mastery": 1}}


def test_current_branch_expanded_and_marked():
    out = render_frontier(TREE, LEARNING, current_pos="LogReg", char_budget=2000)
    assert "← current" in out and "LogReg" in out
    assert "requires: LinReg" in out
    assert "✓ Basics" in out
    assert "CNN" not in out
    assert "○ Deep" in out


def test_budget_truncates_distant_first():
    out = render_frontier(TREE, LEARNING, current_pos="LogReg", char_budget=200)
    assert "LogReg" in out
    assert "Ethics" not in out
    assert len(out) <= 200


def test_deterministic():
    a = render_frontier(TREE, LEARNING, "LogReg", 500)
    b = render_frontier(TREE, LEARNING, "LogReg", 500)
    assert a == b
