import sys
import types
import pytest


@pytest.fixture(autouse=True)
def clean_sys_modules():
    original = dict(sys.modules)
    yield
    for k in list(sys.modules):
        if k not in original:
            del sys.modules[k]
        elif sys.modules[k] is not original[k]:
            sys.modules[k] = original[k]


def _install_stub(name: str, attrs: dict):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


def _router():
    _StateGraph = type("StateGraph", (), {
        "__init__": lambda s, *a, **k: None,
        "add_node": lambda s, *a, **k: None,
        "set_entry_point": lambda s, *a, **k: None,
        "add_conditional_edges": lambda s, *a, **k: None,
        "add_edge": lambda s, *a, **k: None,
        "compile": lambda s: None,
    })
    _util_names = [
        "filter_mastery", "safe_parse_structured", "extract_llm_raw_text",
        "extract_agent_result", "extract_kg_context",
    ]
    _install_stub("langgraph.graph", {"END": "END", "StateGraph": _StateGraph})
    _install_stub("langchain_core.runnables", {"RunnableConfig": dict})
    _install_stub("core.schema.wf_state", {"AgentState": dict, "AgentOutput": dict, "ConceptNode": dict})
    _install_stub("TA.helper.schema", {
        "RoadmapExplore": object, "RoadmapCritique": object, "RoadmapFinal": object,
    })
    _install_stub("TA.helper.prompt", {"ROADMAP_PROMPT": {}})
    _install_stub("TA.helper.few_shot", {"get_language_instruction": lambda *a, **k: ""})
    _install_stub("TA.helper.utils", {n: (lambda *a, **k: None) for n in _util_names})
    _install_stub("TA.helper.context", {"extract_ta_context": lambda *a, **k: ""})
    _install_stub("TA.tracing.tracer", {
        "AgentTracer": type("AgentTracer", (), {"logging": staticmethod(lambda *a, **k: None)}),
    })

    import importlib
    sys.modules.pop("TA.workflow.roadmap", None)
    rm = importlib.import_module("TA.workflow.roadmap")
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
