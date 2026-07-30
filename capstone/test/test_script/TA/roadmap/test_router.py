import sys
import types
import pytest


def _stub():
    for m in [
        "langgraph.graph", "langchain_core.runnables", "core.schema.wf_state",
        "TA.helper.schema", "TA.helper.prompt", "TA.helper.few_shot",
        "TA.helper.utils", "TA.helper.context", "TA.tracing.tracer",
    ]:
        if m not in sys.modules:
            sys.modules[m] = types.ModuleType(m)

    lg = sys.modules["langgraph.graph"]
    lg.END = "END"
    lg.StateGraph = type("SG", (), {
        "__init__": lambda s, *a, **k: None,
        "add_node": lambda s, *a, **k: None,
        "set_entry_point": lambda s, *a, **k: None,
        "add_conditional_edges": lambda s, *a, **k: None,
        "add_edge": lambda s, *a, **k: None,
        "compile": lambda s: None,
    })
    ws = sys.modules["core.schema.wf_state"]
    ws.AgentState = dict


def _router():
    _stub()
    import importlib
    import TA.workflow.roadmap as rm
    importlib.reload(rm)
    return rm._explore_router, rm.END


def st(steps, goal):
    return {"worker_results": {"Roadmap": {"steps": steps, "goal": goal}}}


def test_steps_present_routes_to_evaluator():
    router, _ = _router()
    assert router(st([{"name": "Python"}], "learn python")) == "Roadmap_Evaluator"


def test_empty_steps_empty_goal_routes_to_end():
    router, end = _router()
    assert router(st([], "")) == end


def test_empty_steps_whitespace_goal_routes_to_end():
    router, end = _router()
    assert router(st([], "   ")) == end


def test_goal_present_no_steps_routes_to_evaluator():
    router, _ = _router()
    assert router(st([], "learn ML")) == "Roadmap_Evaluator"


def test_missing_roadmap_key_routes_to_end():
    router, end = _router()
    assert router({"worker_results": {}}) == end
