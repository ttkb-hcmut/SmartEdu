"""Regression guard for the roadmap explore→evaluate retry loop.

Before this loop existed, an `is_feasible=False` critique changed nothing: the graph
always ran Evaluator → TA_Advice → END and shipped the infeasible path as final. These
tests pin the contract that an infeasible (or unknown) verdict reroutes to Roadmap_Explore
until MAX_ROADMAP_ATTEMPTS, then falls through to TA_Advice.

Fully stubs roadmap.py's imports so the router logic is exercised without langgraph /
langchain / the KG stack installed.
"""
import sys
import types
import importlib

import pytest


def _install_stub(name: str, attrs: dict):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


def _load_roadmap():
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

    # Unconditional overwrite — a half-present real module (e.g. langchain_core) must
    # not leak through, which is exactly what the older test got wrong.
    _install_stub("langgraph.graph", {"END": "__end__", "StateGraph": _StateGraph})
    _install_stub("langchain_core.runnables", {"RunnableConfig": dict})
    _install_stub("core.schema.wf_state", {"AgentState": dict, "AgentOutput": dict, "ConceptNode": dict})
    _install_stub("TA.edu.helper.schema", {
        "RoadmapExplore": object, "RoadmapCritique": object, "RoadmapFinal": object,
    })
    _install_stub("TA.edu.helper.prompt", {"ROADMAP_PROMPT": {}})
    _install_stub("TA.edu.helper.few_shot", {"get_language_instruction": lambda *a, **k: ""})
    _install_stub("TA.edu.helper.utils", {n: (lambda *a, **k: None) for n in _util_names})
    _install_stub("TA.edu.helper.context", {"extract_ta_context": lambda *a, **k: ""})
    _install_stub("TA.tracing.tracer", {
        "AgentTracer": type("AgentTracer", (), {"logging": staticmethod(lambda *a, **k: None)}),
    })

    sys.modules.pop("TA.edu.workflow.roadmap", None)
    return importlib.import_module("TA.edu.workflow.roadmap")


def _state(is_feasible, attempts):
    critique = {} if is_feasible == "missing" else {"is_feasible": is_feasible}
    return {
        "worker_results": {"Roadmap": {"critique": critique}},
        "roadmap_attempts": attempts,
    }


def test_infeasible_under_cap_reroutes_to_explore():
    rm = _load_roadmap()
    assert rm._evaluator_router(_state(False, 0)) == "Roadmap_Explore"
    assert rm._evaluator_router(_state(False, 1)) == "Roadmap_Explore"


def test_infeasible_at_cap_falls_through_to_advice():
    rm = _load_roadmap()
    assert rm.MAX_ROADMAP_ATTEMPTS == 2
    assert rm._evaluator_router(_state(False, 2)) == "TA_Advice"


def test_feasible_goes_to_advice():
    rm = _load_roadmap()
    assert rm._evaluator_router(_state(True, 0)) == "TA_Advice"


def test_unknown_verdict_is_retried_not_shipped():
    # safe_parse fallback can drop is_feasible → None. None must NOT be treated as feasible.
    rm = _load_roadmap()
    assert rm._evaluator_router(_state(None, 0)) == "Roadmap_Explore"
    assert rm._evaluator_router(_state("missing", 0)) == "Roadmap_Explore"
    # ...but the cap still bounds it.
    assert rm._evaluator_router(_state(None, 2)) == "TA_Advice"
