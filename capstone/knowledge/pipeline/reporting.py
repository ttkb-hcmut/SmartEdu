"""Pure terminal report policy for course ingestion."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from core.schema.ingest import REPORT_COMPLETED, REPORT_FAILED, REPORT_PARTIAL


@dataclass(frozen=True)
class ReportOutcome:
    source: str
    value: Any = None
    file_name: Optional[str] = None
    error: Optional[BaseException] = None


@dataclass(frozen=True)
class ReportDecision:
    report: Dict
    error: Optional[BaseException]


def source_outcomes(source: str, file_names: Iterable[str], results: Iterable[Any]) -> tuple[ReportOutcome, ...]:
    return tuple(
        ReportOutcome(source, file_name=file_name, error=result)
        if isinstance(result, BaseException)
        else ReportOutcome(source, value=result, file_name=file_name)
        for file_name, result in zip(file_names, results)
    )


def reduce_report(seed: Dict, outcomes: Iterable[ReportOutcome], *, fatal: BaseException = None,
                  duration_s: float, finished_at: str) -> ReportDecision:
    report = deepcopy(seed)
    for outcome in outcomes:
        if outcome.error is not None:
            report["errors"].append({"file": outcome.file_name, "error": str(outcome.error)})
        elif outcome.value is not None:
            if outcome.source == "anchors":
                report["anchors"] = outcome.value
            else:
                report[outcome.source].append(outcome.value)

    if fatal is not None:
        report["errors"].append({"stage": "course-flow", "error": str(fatal)})

    has_results = bool(
        report["textbooks"] or report["slides"] or report["videos"] or report["anchors"]
    )
    report["status"] = (
        REPORT_PARTIAL if report["errors"] and has_results
        else REPORT_FAILED if report["errors"]
        else REPORT_COMPLETED
    )
    report["duration_s"] = duration_s
    report["finished_at"] = finished_at

    error = fatal
    if error is None and report["errors"]:
        error = RuntimeError(f"{len(report['errors'])} ingestion source(s) failed")
    return ReportDecision(report, error)
