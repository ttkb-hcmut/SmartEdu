"""Teacher Judge — trace consumer for the retrieval ablation.

Deterministic gold-chunk + trajectory metrics are pure functions here;
judge metrics (DeepEval triad + answer correctness) are lazy-imported and
opt-in — the app must boot with deepeval uninstalled.
"""

from __future__ import annotations

import json
import logging
import random
import re
import statistics
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from TA.tracing.schema import ChatTrace, TraceSession

logger = logging.getLogger(__name__)

_COMPONENT_PREFIX = "Comp_"
_FINAL_RETRIEVAL_NODES = {"Fusion", "Agentic_Retrieve"}
_ACCEPTED_SCHEMAS = {"1.0", "1.1", "1.2", "1.3"}


# ── deterministic metrics ─────────────────────────────────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)


def _normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(char for char in value if char not in string.punctuation)
    return " ".join(_ARTICLES.sub(" ", value).split())


def _answer_f1(prediction: str, gold: str) -> float:
    predicted_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_scores(answer: str, gold_answers: str | List[str]) -> Dict[str, Optional[float]]:
    """Normalized EM and token F1 against any accepted answer."""
    golds = [gold_answers] if isinstance(gold_answers, str) else gold_answers
    golds = [gold for gold in golds if gold]
    if not golds:
        return {"answer_exact_match": None, "answer_f1": None}
    normalized = _normalize_answer(answer)
    return {
        "answer_exact_match": max(float(normalized == _normalize_answer(gold)) for gold in golds),
        "answer_f1": max(_answer_f1(answer, gold) for gold in golds),
    }


def context_scores(retrieved: List[str], gold: List[str]) -> Dict[str, Optional[float]]:
    ## no gold -> referenceless question, metric undefined not zero
    if not gold:
        return {
            "context_precision": None,
            "context_recall": None,
            "support_f1": None,
            "support_exact_match": None,
            "complete_chain": None,
        }
    r, g = set(retrieved), set(gold)
    hit = len(r & g)
    precision = hit / len(r) if r else 0.0
    recall = hit / len(g)
    support_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "context_precision": precision,
        "context_recall": recall,
        "support_f1": support_f1,
        "support_exact_match": float(r == g),
        "complete_chain": float(g <= r),
    }


def round_scores(chat: ChatTrace, gold: List[str]) -> Dict[str, Any]:
    rounds = [step for step in chat.agent if step.node == "Retrieval_Round"]
    gold_set = set(gold)
    gathered: set[str] = set()
    growth = []
    first_gold_round = None
    complete_chain_round = None
    for fallback_index, step in enumerate(rounds):
        round_index = step.tool_result.get("round", fallback_index)
        before = len(gathered)
        gathered.update(str(chunk.get("uri")) for chunk in step.chunks if chunk.get("uri"))
        growth.append(len(gathered) - before)
        if first_gold_round is None and gathered & gold_set:
            first_gold_round = round_index
        if gold_set and complete_chain_round is None and gold_set <= gathered:
            complete_chain_round = round_index
    return {
        "first_gold_round": first_gold_round,
        "complete_chain_round": complete_chain_round,
        "ledger_growth": growth,
        "retrieval_rounds": len(rounds),
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

    source_hit_rate = useful / len(comp_steps) if comp_steps else None
    candidate_recall = len(gathered & gold_set) / len(gold_set) if gold_set else None
    node_latency_sum = sum(s.latency_ms for s in chat.agent)
    return {
        "source_hit_rate": source_hit_rate,
        "candidate_recall": candidate_recall,
        "traj_precision": source_hit_rate,
        "traj_recall": candidate_recall,
        "node_latency_sum_ms": node_latency_sum,
        "latency_ms": chat.workflow_latency_ms or node_latency_sum,
        "workflow_latency_ms": chat.workflow_latency_ms,
        "turn_latency_ms": chat.turn_latency_ms,
        "time_to_first_token_ms": chat.time_to_first_token_ms,
        "tokens": tokens,
    }


def retrieved_uris(chat: ChatTrace) -> List[str]:
    for step in reversed(chat.agent):
        if step.node in _FINAL_RETRIEVAL_NODES:
            return [c.get("uri") for c in step.chunks]
    return []


def retrieved_texts(chat: ChatTrace) -> List[str]:
    for step in reversed(chat.agent):
        if step.node in _FINAL_RETRIEVAL_NODES:
            return [c.get("text", "") for c in step.chunks]
    return []


def answer_context_uris(chat: ChatTrace) -> Optional[List[str]]:
    for step in reversed(chat.agent):
        if step.node in _FINAL_RETRIEVAL_NODES:
            uris = step.tool_result.get("answer_context_uris")
            return [str(uri) for uri in uris] if uris is not None else None
    return None


def retrieval_outcome(chat: ChatTrace) -> Dict[str, Any]:
    for step in reversed(chat.agent):
        if step.node in _FINAL_RETRIEVAL_NODES:
            return {
                "retrieval_validity": step.tool_result.get("validity", "unknown"),
                "retrieval_attempts": step.tool_result.get("attempts"),
                "seed_calls": step.tool_result.get("seed_calls"),
                "aggregator_calls": step.tool_result.get("aggregator_calls"),
                "blocked_tool_calls": step.tool_result.get("blocked_tool_calls", 0),
                "aggregator_latency_ms": step.tool_result.get("aggregator_latency_ms"),
                "retrieval_rounds": step.tool_result.get("rounds"),
                "invalid_source_requests": step.tool_result.get("invalid_source_requests", 0),
                "repeated_queries": step.tool_result.get("repeated_queries", 0),
                "context_limit_stops": step.tool_result.get("context_limit_stops", 0),
                "planner_stop_reason": step.tool_result.get("stop_reason", ""),
                "planner_schema_repairs": step.tool_result.get("schema_repairs", 0),
                "planner_transport_retries": step.tool_result.get("transport_retries", 0),
                "planner_calls": step.tool_result.get("planner_calls"),
                "retrieval_errors": step.tool_result.get("errors", []),
            }
    return {
        "retrieval_validity": "not-recorded",
        "retrieval_attempts": None,
        "seed_calls": None,
        "aggregator_calls": None,
        "blocked_tool_calls": 0,
        "aggregator_latency_ms": None,
        "retrieval_rounds": None,
        "invalid_source_requests": 0,
        "repeated_queries": 0,
        "context_limit_stops": 0,
        "planner_stop_reason": "",
        "planner_schema_repairs": 0,
        "planner_transport_retries": 0,
        "planner_calls": None,
        "retrieval_errors": [],
    }


def evaluate_chat(chat: ChatTrace, fixture_item: Dict) -> Dict[str, Any]:
    gold = fixture_item.get("gold_chunk_ids", [])
    final_uris = retrieved_uris(chat)
    compiled_uris = answer_context_uris(chat)
    component_uris = {
        chunk.get("uri")
        for step in chat.agent
        if step.node.startswith(_COMPONENT_PREFIX)
        for chunk in step.chunks
    }
    final_gold = set(final_uris) & set(gold)
    candidate_gold = component_uris & set(gold)
    row: Dict[str, Any] = {
        "fixture_id": fixture_item["id"],
        "track": fixture_item.get("track", ""),
        "preset": chat.preset,
        "query": chat.query,
        "answer": chat.final_output,
        "gold_answer": fixture_item.get("gold_answer", ""),
        "hops": fixture_item.get("hops"),
        "status": chat.status,
        "errors": chat.errors,
        "policy_id": chat.policy_id,
        "policy_digest": chat.policy_digest,
        "harness_id": chat.harness_id,
        "run_id": chat.run_id,
        "question_id": chat.question_id,
        "model": chat.model,
        "temperature": chat.temperature,
        "code_revision": chat.code_revision,
        "dirty": chat.dirty,
        "node_configs": chat.node_configs,
        "node_execution": [
            {
                "node": step.node,
                "latency_ms": step.latency_ms,
                "config": step.execution_config,
                "error": step.tool_result.get("error", ""),
            }
            for step in chat.agent
        ],
    }
    row.update(context_scores(final_uris, gold))
    row.update(answer_scores(row["answer"], fixture_item.get("gold_answer", "")))
    row.update(trajectory_scores(chat, gold))
    row.update(round_scores(chat, gold))
    row.update(retrieval_outcome(chat))
    row["answer_latency_ms"] = next(
        (step.latency_ms for step in reversed(chat.agent) if step.node == "TA_Retrieve_Finish"),
        None,
    )
    candidate_recall = row.get("candidate_recall")
    row["selection_recall_loss"] = (
        max(candidate_recall - row["context_recall"], 0.0)
        if candidate_recall is not None and row["context_recall"] is not None
        else None
    )
    row["gold_lost_in_selection"] = len(candidate_gold - final_gold)
    row["retrieval_context"] = retrieved_texts(chat)
    row["answer_context_uris"] = compiled_uris
    row["answer_context_recall"] = (
        len(set(compiled_uris) & set(gold)) / len(set(gold))
        if compiled_uris is not None and gold
        else None
    )
    return row


def evaluate_session(session: TraceSession, fixture: List[Dict]) -> List[Dict[str, Any]]:
    if session.schema_version not in _ACCEPTED_SCHEMAS:
        logger.warning(f"[evaluator] unknown schema {session.schema_version}, skipping session")
        return []
    by_id = {f["id"]: f for f in fixture}
    by_question = {f["question"]: f for f in fixture}
    rows = []
    for chat in session.chat:
        if chat.warmup:
            continue
        item = by_id.get(chat.question_id) if chat.question_id else by_question.get(chat.query)
        if item:
            rows.append(evaluate_chat(chat, item))
    return rows


# ── aggregation + rendering ───────────────────────────────────────────

_TABLE_METRICS = [
    "answer_exact_match",
    "answer_f1",
    "context_precision",
    "context_recall",
    "answer_context_recall",
    "support_f1",
    "support_exact_match",
    "complete_chain",
    "source_hit_rate",
    "candidate_recall",
    "selection_recall_loss",
    "aggregator_calls",
    "blocked_tool_calls",
    "retrieval_rounds",
    "aggregator_latency_ms",
    "answer_latency_ms",
    "latency_ms",
]


def _agg(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def _quality_eligible(row: Dict[str, Any]) -> bool:
    return row.get("status") == "SUCCESS" and row.get("retrieval_validity") == "valid"


def render_table(
    rows: List[Dict[str, Any]],
    expected_counts: Optional[Dict[str, int]] = None,
) -> str:
    expected_counts = expected_counts or {}
    presets = sorted({r["preset"] for r in rows} | set(expected_counts))
    lines = [
        "| preset | valid n | invalid n | availability | " + " | ".join(_TABLE_METRICS) + " |",
        "|---" * (len(_TABLE_METRICS) + 4) + "|",
    ]
    for p in presets:
        group = [r for r in rows if r["preset"] == p]
        quality = [row for row in group if _quality_eligible(row)]
        expected = expected_counts.get(p, len(group))
        successful = sum(row.get("status") == "SUCCESS" for row in group)
        availability = successful / expected if expected else 0.0
        cells = []
        for m in _TABLE_METRICS:
            v = _agg([r.get(m) for r in quality])
            cells.append(
                "-" if v is None else (
                    f"{v:.0f}" if m.endswith("_ms") else f"{v:.2f}"
                )
            )
        lines.append(
            f"| {p} | {len(quality)} | {len(group) - len(quality)} | {availability:.2f} | "
            + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def _percentile(values: List[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


_PAIR_METRICS = (
    "answer_exact_match",
    "answer_f1",
    "context_recall",
    "complete_chain",
    "workflow_latency_ms",
    "answer_latency_ms",
)


def paired_arm_summary(
    rows: List[Dict[str, Any]],
    *,
    baseline: str = "RAG",
    treatment: str = "FULL",
    resamples: int = 2_000,
    seed: int = 42,
) -> Dict[str, Any]:
    grouped: Dict[tuple[str, str, str], Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        key = (
            str(row.get("harness_id", "")),
            str(row.get("run_id", "")),
            str(row.get("fixture_id", "")),
        )
        grouped.setdefault(key, {}).setdefault(row.get("preset", ""), []).append(row)

    pairs = []
    for key, arms in grouped.items():
        left = arms.get(baseline, [])
        right = arms.get(treatment, [])
        if len(left) == len(right) == 1 and _quality_eligible(left[0]) and _quality_eligible(right[0]):
            pairs.append((key[2], left[0], right[0]))
    pairs.sort(key=lambda item: item[0])

    arm_values = {baseline: {}, treatment: {}}
    for arm, index in ((baseline, 1), (treatment, 2)):
        for metric in _PAIR_METRICS:
            values = [pair[index].get(metric) for pair in pairs]
            arm_values[arm][metric] = _agg(values)
        workflow = [pair[index].get("workflow_latency_ms") for pair in pairs]
        arm_values[arm]["workflow_p50_ms"] = _percentile([value for value in workflow if value is not None], 0.5)
        arm_values[arm]["workflow_p95_ms"] = _percentile([value for value in workflow if value is not None], 0.95)

    rng = random.Random(seed)
    deltas = {}
    for metric in _PAIR_METRICS:
        values = [
            right.get(metric) - left.get(metric)
            for _, left, right in pairs
            if left.get(metric) is not None and right.get(metric) is not None
        ]
        if not values:
            continue
        bootstrap = [
            statistics.mean(rng.choice(values) for _ in values)
            for _ in range(resamples)
        ]
        deltas[metric] = {
            "n": len(values),
            "mean_delta": statistics.mean(values),
            "ci95_low": _percentile(bootstrap, 0.025),
            "ci95_high": _percentile(bootstrap, 0.975),
        }
    return {
        "baseline": baseline,
        "treatment": treatment,
        "question_ids": [question_id for question_id, _, _ in pairs],
        "arms": arm_values,
        "deltas": deltas,
    }


def render_paired_arm_report(
    rows: List[Dict[str, Any]],
    *,
    expected_counts: Optional[Dict[str, int]] = None,
) -> str:
    expected_counts = expected_counts or {}
    summary = paired_arm_summary(rows)
    pair_count = len(summary["question_ids"])
    planned = min(expected_counts.get("RAG", 0), expected_counts.get("FULL", 0))
    lines = [
        "## Paired RAG → FULL headline",
        "",
        f"paired questions: {pair_count}" + (f" / {planned} planned" if planned else ""),
        "",
        "| arm | paired n | answer EM | answer F1 | context recall | complete chain | mean workflow ms | workflow p50 ms | workflow p95 ms | mean answer ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("RAG", "FULL"):
        values = summary["arms"][arm]
        cells = []
        for metric in (
            "answer_exact_match", "answer_f1", "context_recall", "complete_chain",
            "workflow_latency_ms", "workflow_p50_ms", "workflow_p95_ms", "answer_latency_ms",
        ):
            value = values.get(metric)
            cells.append("-" if value is None else (f"{value:.0f}" if metric.endswith("_ms") else f"{value:.2f}"))
        lines.append(f"| {arm} | {pair_count} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| metric | n | FULL − RAG | bootstrap 95% interval |",
        "|---|---:|---:|---:|",
    ]
    for metric, delta in summary["deltas"].items():
        lines.append(
            f"| {metric} | {delta['n']} | {delta['mean_delta']:.3f} | "
            f"[{delta['ci95_low']:.3f}, {delta['ci95_high']:.3f}] |"
        )
    if not summary["deltas"]:
        lines.append("| no paired valid rows | 0 | - | - |")
    return "\n".join(lines)


def paired_harness_summary(
    rows: List[Dict[str, Any]],
    *,
    baseline: str = "agentic-v2",
    treatment: str = "agentic-v3",
    resamples: int = 2_000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    metrics = ("answer_f1", "support_f1", "complete_chain", "latency_ms")
    indexed = {
        (row.get("harness_id"), row.get("preset"), row.get("fixture_id")): row
        for row in rows
        if _quality_eligible(row)
    }
    presets = sorted({row.get("preset") for row in rows if row.get("preset")})
    rng = random.Random(seed)
    summaries = []
    for preset in presets:
        ids = sorted({
            fixture_id
            for harness, arm, fixture_id in indexed
            if arm == preset
            and (baseline, arm, fixture_id) in indexed
            and (treatment, arm, fixture_id) in indexed
        })
        for metric in metrics:
            deltas = [
                indexed[(treatment, preset, fixture_id)].get(metric)
                - indexed[(baseline, preset, fixture_id)].get(metric)
                for fixture_id in ids
                if indexed[(treatment, preset, fixture_id)].get(metric) is not None
                and indexed[(baseline, preset, fixture_id)].get(metric) is not None
            ]
            if not deltas:
                continue
            bootstrap = [
                statistics.mean(rng.choice(deltas) for _ in deltas)
                for _ in range(resamples)
            ]
            summaries.append({
                "preset": preset,
                "metric": metric,
                "n": len(deltas),
                "mean_delta": statistics.mean(deltas),
                "median_delta": statistics.median(deltas),
                "ci95_low": _percentile(bootstrap, 0.025),
                "ci95_high": _percentile(bootstrap, 0.975),
                "baseline": baseline,
                "treatment": treatment,
                "bootstrap_seed": seed,
                "bootstrap_resamples": resamples,
            })
    return summaries


def render_paired_report(
    rows: List[Dict[str, Any]],
    *,
    baseline: str = "agentic-v2",
    treatment: str = "agentic-v3",
) -> str:
    summaries = paired_harness_summary(rows, baseline=baseline, treatment=treatment)
    lines = [
        f"## Paired {baseline} → {treatment} deltas",
        "",
        "Positive quality deltas favor V3; positive latency deltas mean V3 is slower.",
        "",
        "| preset | metric | n | mean delta | median delta | bootstrap 95% interval |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['preset']} | {item['metric']} | {item['n']} | "
            f"{item['mean_delta']:.3f} | {item['median_delta']:.3f} | "
            f"[{item['ci95_low']:.3f}, {item['ci95_high']:.3f}] |"
        )
    return "\n".join(lines)


def _config_drift(rows: List[Dict[str, Any]]) -> List[str]:
    seen: Dict[tuple[str, str], set[str]] = {}
    for row in rows:
        for node, config in row.get("node_configs", {}).items():
            seen.setdefault((row["preset"], node), set()).add(
                json.dumps(config, sort_keys=True, default=str)
            )
    return [f"{preset}/{node}" for (preset, node), values in seen.items() if len(values) > 1]


def render_report(
    rows: List[Dict[str, Any]],
    *,
    partial: bool = False,
    expected_counts: Optional[Dict[str, int]] = None,
    provider_errors: Optional[Dict[str, int]] = None,
    median_failed_latency_s: Optional[float] = None,
) -> str:
    expected_counts = expected_counts or {}
    presets = sorted({row["preset"] for row in rows} | set(expected_counts))
    quality_rows = [row for row in rows if _quality_eligible(row)]
    lines = ["# Retrieval benchmark report", ""]
    if partial:
        lines += ["> **PARTIAL EXECUTION:** quality statistics include successful, mechanically valid cases only.", ""]
    if {"RAG", "FULL"} <= set(presets):
        lines += [render_paired_arm_report(rows, expected_counts=expected_counts), ""]
    lines += ["## Quality", "", render_table(rows, expected_counts), ""]

    lines += [
        "## Answer quality by hop count",
        "",
        "| preset | hops | n | exact match | token F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for preset in presets:
        groups: Dict[int | None, List[Dict[str, Any]]] = {}
        for row in quality_rows:
            if row["preset"] == preset:
                groups.setdefault(row.get("hops"), []).append(row)
        for hops, group in sorted(groups.items(), key=lambda item: item[0] is None):
            em = _agg([row.get("answer_exact_match") for row in group])
            f1 = _agg([row.get("answer_f1") for row in group])
            lines.append(
                f"| {preset} | {hops if hops is not None else '-'} | {len(group)} | "
                f"{'-' if em is None else f'{em:.2f}'} | {'-' if f1 is None else f'{f1:.2f}'} |"
            )

    lines += [
        "",
        "## Answer quality conditional on retrieval chain",
        "",
        "| preset | chain outcome | n | exact match | token F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for preset in presets:
        for complete, label in ((1.0, "complete"), (0.0, "incomplete")):
            group = [
                row for row in quality_rows
                if row["preset"] == preset and row.get("complete_chain") == complete
            ]
            if not group:
                continue
            em = _agg([row.get("answer_exact_match") for row in group])
            f1 = _agg([row.get("answer_f1") for row in group])
            lines.append(
                f"| {preset} | {label} | {len(group)} | "
                f"{'-' if em is None else f'{em:.2f}'} | {'-' if f1 is None else f'{f1:.2f}'} |"
            )

    lines += [
        "## Run outcomes",
        "",
        "| preset | recorded | execution completed | execution errors | retrieval invalid | no-attempt contract |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for preset in presets:
        group = [row for row in rows if row["preset"] == preset]
        successful = sum(row.get("status") == "SUCCESS" for row in group)
        invalid = sum(row.get("retrieval_validity") == "invalid" for row in group)
        no_attempt = sum(row.get("retrieval_validity") == "policy-invalid" for row in group)
        lines.append(
            f"| {preset} | {len(group)} | {successful} | {len(group) - successful} | "
            f"{invalid} | {no_attempt} |"
        )

    provider_errors = provider_errors or {}
    lines += [
        "",
        "### Provider errors",
        "",
        "| category | attempts |",
        "|---|---:|",
    ]
    if provider_errors:
        for category, count in sorted(provider_errors.items()):
            lines.append(f"| {category} | {count} |")
    else:
        lines.append("| none | 0 |")

    if median_failed_latency_s is not None:
        lines += ["", f"Median failed-attempt latency: {median_failed_latency_s:.1f}s"]

    lines += [
        "",
        "## Planner diagnostics",
        "",
        "| preset | stop reasons | schema repairs | transport retries | repeated queries | mean rounds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for preset in presets:
        group = [row for row in quality_rows if row["preset"] == preset]
        reasons = Counter(row.get("planner_stop_reason") or "not-recorded" for row in group)
        reason_text = ", ".join(f"{name}:{count}" for name, count in sorted(reasons.items())) or "-"
        rounds = _agg([row.get("retrieval_rounds") for row in group])
        lines.append(
            f"| {preset} | {reason_text} | "
            f"{sum(row.get('planner_schema_repairs', 0) for row in group)} | "
            f"{sum(row.get('planner_transport_retries', 0) for row in group)} | "
            f"{sum(row.get('repeated_queries', 0) for row in group)} | "
            f"{'-' if rounds is None else f'{rounds:.2f}'} |"
        )

    lines += [
        "",
        "## Latency distribution",
        "",
        "Workflow latency is measured with a monotonic clock. Node-time sum is diagnostic only and may exceed wall-time when branches overlap.",
        "",
        "| preset | workflow p50 ms | workflow p95 ms | workflow max ms | turn p50 ms | TTFT p50 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for preset in presets:
        group = [row for row in rows if row["preset"] == preset]
        workflow = [row["workflow_latency_ms"] for row in group if row.get("workflow_latency_ms")]
        turn = [row["turn_latency_ms"] for row in group if row.get("turn_latency_ms")]
        ttft = [row["time_to_first_token_ms"] for row in group if row.get("time_to_first_token_ms") is not None]
        vals = (
            _percentile(workflow, 0.5),
            _percentile(workflow, 0.95),
            max(workflow) if workflow else None,
            _percentile(turn, 0.5),
            _percentile(ttft, 0.5),
        )
        cells = ["-" if value is None else f"{value:.0f}" for value in vals]
        lines.append(f"| {preset} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Effective node configuration",
        "",
        "| preset | node | kind | model | ctx | temperature | retries | harness | top-k |",
        "|---|---|---|---|---:|---:|---:|---|---:|",
    ]
    for preset in presets:
        group = [row for row in rows if row["preset"] == preset]
        executed = {
            step["node"]
            for row in group
            for step in row.get("node_execution", [])
        }
        configs = group[0].get("node_configs", {}) if group else {}
        visible = executed or set(configs)
        for node in sorted(visible):
            config = next(
                (
                    step.get("config", {})
                    for row in group
                    for step in row.get("node_execution", [])
                    if step["node"] == node and step.get("config")
                ),
                configs.get(node, {}),
            )
            lines.append(
                f"| {preset} | {node} | {config.get('kind', '-')} | "
                f"{config.get('model') or '-'} | {config.get('num_ctx') or '-'} | "
                f"{config.get('temperature') if config.get('temperature') is not None else '-'} | "
                f"{config.get('max_retries') if config.get('max_retries') is not None else '-'} | "
                f"{config.get('harness_id', '-')} | {config.get('top_k') or '-'} |"
            )

    drift = _config_drift(rows)
    lines += [
        "",
        f"Configuration drift within the same preset/node: {', '.join(drift) if drift else 'none detected'}.",
        "",
        "## Worst observed cases",
        "",
        "| preset | question | status | final recall | candidate recall | selection loss | workflow ms |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    worst = sorted(
        rows,
        key=lambda row: (
            row.get("context_recall") if row.get("context_recall") is not None else 1.0,
            -(row.get("selection_recall_loss") or 0.0),
            -(row.get("workflow_latency_ms") or 0.0),
        ),
    )[:10]
    for row in worst:
        values = [row.get("context_recall"), row.get("candidate_recall"), row.get("selection_recall_loss")]
        scores = ["-" if value is None else f"{value:.2f}" for value in values]
        lines.append(
            f"| {row['preset']} | {row['fixture_id']} | {row.get('status', '')} | "
            f"{' | '.join(scores)} | {row.get('workflow_latency_ms') or 0:.0f} |"
        )

    lines += [
        "",
        "## Metric interpretation",
        "",
        "- Final context precision/recall compare the returned evidence with canonical gold paragraph IDs.",
        "- Answer exact match and token F1 use normalized QA scoring against accepted gold answers.",
        "- Candidate recall measures gold found by retrieval sources before final selection.",
        "- Source hit rate is the fraction of retrieval calls that found at least one gold paragraph; it is not answer correctness.",
        "- Aggregator calls and latency separate agentic expansion from deterministic seed retrieval; answer latency is the no-tool benchmark answer call.",
        "- Selection recall loss is candidate recall minus final recall and isolates evidence dropped after retrieval.",
        "- Quality averages include only successful executions with mechanically valid retrieval.",
        "- Failed and invalid executions remain availability diagnostics and never contribute quality values.",
    ]
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
