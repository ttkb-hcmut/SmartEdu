"""Teacher Judge — trace consumer for the retrieval ablation.

Deterministic gold-chunk + trajectory metrics are pure functions here;
judge metrics (DeepEval triad + answer correctness) are lazy-imported and
opt-in — the app must boot with deepeval uninstalled.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from TA.tracing.schema import ChatTrace, TraceSession

logger = logging.getLogger(__name__)

_COMPONENT_PREFIX = "Comp_"
_FUSION_NODE = "Fusion"
_ACCEPTED_SCHEMAS = {"1.0", "1.1"}


# ── deterministic metrics ─────────────────────────────────────────────

def context_scores(retrieved: List[str], gold: List[str]) -> Dict[str, Optional[float]]:
    ## no gold -> referenceless question, metric undefined not zero
    if not gold:
        return {"context_precision": None, "context_recall": None}
    r, g = set(retrieved), set(gold)
    hit = len(r & g)
    return {
        "context_precision": hit / len(r) if r else 0.0,
        "context_recall": hit / len(g),
    }


def trajectory_scores(chat: ChatTrace, gold: List[str]) -> Dict[str, Any]:
    comp_steps = [s for s in chat.agent if s.node.startswith(_COMPONENT_PREFIX)]
    gold_set = set(gold)

    useful = sum(1 for s in comp_steps if gold_set & {c.get("uri") for c in s.chunks})
    gathered = {c.get("uri") for s in comp_steps for c in s.chunks}

    tokens: Dict[str, int] = {}
    for s in chat.agent:
        for k, v in s.tokens.items():
            tokens[k] = tokens.get(k, 0) + v

    return {
        ## no component calls -> precision vacuous, never inflate PLAIN
        "traj_precision": useful / len(comp_steps) if comp_steps else None,
        "traj_recall": len(gathered & gold_set) / len(gold_set) if gold_set else None,
        "latency_ms": sum(s.latency_ms for s in chat.agent),
        "tokens": tokens,
    }


def retrieved_uris(chat: ChatTrace) -> List[str]:
    for step in chat.agent:
        if step.node == _FUSION_NODE:
            return [c.get("uri") for c in step.chunks]
    return []


def retrieved_texts(chat: ChatTrace) -> List[str]:
    for step in chat.agent:
        if step.node == _FUSION_NODE:
            return [c.get("text", "") for c in step.chunks]
    return []


def evaluate_chat(chat: ChatTrace, fixture_item: Dict) -> Dict[str, Any]:
    gold = fixture_item.get("gold_chunk_ids", [])
    row: Dict[str, Any] = {
        "fixture_id": fixture_item["id"],
        "track": fixture_item.get("track", ""),
        "preset": chat.preset,
        "query": chat.query,
        "answer": chat.final_output,
        "gold_answer": fixture_item.get("gold_answer", ""),
    }
    row.update(context_scores(retrieved_uris(chat), gold))
    row.update(trajectory_scores(chat, gold))
    row["retrieval_context"] = retrieved_texts(chat)
    return row


def evaluate_session(session: TraceSession, fixture: List[Dict]) -> List[Dict[str, Any]]:
    if session.schema_version not in _ACCEPTED_SCHEMAS:
        logger.warning(f"[evaluator] unknown schema {session.schema_version}, skipping session")
        return []
    by_question = {f["question"]: f for f in fixture}
    rows = []
    for chat in session.chat:
        item = by_question.get(chat.query)
        if item:
            rows.append(evaluate_chat(chat, item))
    return rows


# ── aggregation + rendering ───────────────────────────────────────────

_TABLE_METRICS = ["context_precision", "context_recall", "traj_precision", "traj_recall", "latency_ms"]


def _agg(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def render_table(rows: List[Dict[str, Any]]) -> str:
    presets = sorted({r["preset"] for r in rows})
    lines = ["| preset | n | " + " | ".join(_TABLE_METRICS) + " |",
             "|---" * (len(_TABLE_METRICS) + 2) + "|"]
    for p in presets:
        group = [r for r in rows if r["preset"] == p]
        cells = []
        for m in _TABLE_METRICS:
            v = _agg([r.get(m) for r in group])
            cells.append("-" if v is None else (f"{v:.0f}" if m == "latency_ms" else f"{v:.2f}"))
        lines.append(f"| {p} | {len(group)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_session(path: str | Path) -> TraceSession:
    return TraceSession.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


# ── judge metrics (opt-in, lazy deepeval) ─────────────────────────────

def judge_chat(row: Dict[str, Any], retrieval_context: List[str], runs: int = 3) -> Dict[str, Any]:
    """Median-of-runs DeepEval triad + answer correctness. Relative evidence only."""
    try:
        from deepeval.metrics import (
            FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric, GEval,
        )
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError:
        logger.warning("[evaluator] deepeval not installed, judge metrics skipped")
        return {}

    case = LLMTestCase(
        input=row["query"],
        actual_output=row["answer"],
        expected_output=row.get("gold_answer") or None,
        retrieval_context=retrieval_context or None,
    )
    metrics: Dict[str, Any] = {
        "faithfulness": lambda: FaithfulnessMetric(),
        "answer_relevancy": lambda: AnswerRelevancyMetric(),
        "contextual_relevancy": lambda: ContextualRelevancyMetric(),
    }
    if row.get("gold_answer"):
        metrics["answer_correctness"] = lambda: GEval(
            name="AnswerCorrectness",
            criteria="Does the actual output convey the same answer as the expected output?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
        )

    out: Dict[str, Any] = {}
    for name, make in metrics.items():
        if name in ("faithfulness", "contextual_relevancy") and not retrieval_context:
            continue
        scores = []
        for _ in range(runs):
            try:
                m = make()
                m.measure(case)
                scores.append(m.score)
            except Exception as e:
                logger.warning(f"[evaluator] judge metric {name} failed: {e}")
        if scores:
            out[f"judge_{name}"] = statistics.median(scores)
            out[f"judge_{name}_spread"] = max(scores) - min(scores)
    return out
