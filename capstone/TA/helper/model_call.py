import asyncio
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

from core.schema.retrieval import RetrievalOutputMode


_provider_request_counter: ContextVar[dict[str, int] | None] = ContextVar(
    "provider_request_counter", default=None
)
_provider_request_gate: ContextVar[Callable[[str], None] | None] = ContextVar(
    "provider_request_gate", default=None
)


@contextmanager
def count_provider_requests():
    counter = {"count": 0}
    token = _provider_request_counter.set(counter)
    try:
        yield counter
    finally:
        _provider_request_counter.reset(token)


@contextmanager
def provider_request_gate(gate: Callable[[str], None]):
    token = _provider_request_gate.set(gate)
    try:
        yield
    finally:
        _provider_request_gate.reset(token)


def _record_provider_request(request_key: str) -> None:
    gate = _provider_request_gate.get()
    if gate is not None:
        gate(request_key)
    counter = _provider_request_counter.get()
    if counter is not None:
        counter["count"] += 1


def status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        match = re.search(r"status code:\s*(\d{3})", str(exc), re.I)
        status = int(match.group(1)) if match else None
    return int(status) if status is not None else None


def is_transport_error(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or status_code(exc) is not None


def is_provider_budget_error(exc: Exception) -> bool:
    return bool(getattr(exc, "provider_budget_exhausted", False))


def is_transient(exc: Exception) -> bool:
    status = status_code(exc)
    err_str = str(exc).lower()
    if "provider_overloaded" in err_str or "rate limit" in err_str or "overloaded" in err_str:
        return True
    return isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or (
        status is not None and status >= 500
    )


async def bounded_ainvoke(
    runner,
    payload,
    *,
    config,
    timeout_s: int,
    retries: int,
    request_key: str = "",
):
    used = 0
    while True:
        try:
            _record_provider_request(request_key)
            invocation = runner.ainvoke(payload, config=config)
            result = await asyncio.wait_for(invocation, timeout=timeout_s) if timeout_s else await invocation
            return result, used
        except Exception as exc:
            if used >= retries or not is_transient(exc):
                try:
                    exc.transport_retries_used = used
                except Exception:
                    pass
                raise
            await asyncio.sleep(2**used)
            used += 1


async def ta_ainvoke(runner, payload, config):
    ## live TA only; benchmark paths keep their own retry policy
    result, _ = await bounded_ainvoke(runner, payload, config=config, timeout_s=0, retries=2)
    return result


_ACTION_ALIASES = {
    "search": "retrieve",
    "lookup": "retrieve",
    "retrieval": "retrieve",
    "query": "retrieve",
    "done": "stop",
    "finish": "stop",
}

_STOP_REASON_ALIASES = {
    "no_basis": "insufficient_evidence",
    "no_evidence": "insufficient_evidence",
    "insufficient": "insufficient_evidence",
    "complete": "chain_complete",
    "chain_completed": "chain_complete",
    "budget": "budget_exhausted",
    "max_calls": "budget_exhausted",
}

_CLAIM_TEXT_ALIASES = ("statement", "text", "fact")


def _normalize_claims(payload: dict, key: str) -> tuple[str, ...]:
    entries = payload.get(key)
    if not isinstance(entries, list):
        return ()
    notes: tuple[str, ...] = ()
    rebuilt = []
    for entry in entries:
        if isinstance(entry, dict) and "claim" not in entry:
            entry = dict(entry)
            for alias in _CLAIM_TEXT_ALIASES:
                if alias in entry:
                    ## key rename only, claim text untouched -- no evidence invented
                    entry["claim"] = entry.pop(alias)
                    notes += (f"{key}[].{alias} -> claim",)
                    break
        rebuilt.append(entry)
    if notes:
        payload[key] = rebuilt
    return notes


def normalize_provider_payload(payload: dict) -> tuple[dict, tuple[str, ...]]:
    payload = dict(payload)
    notes: tuple[str, ...] = ()
    if len(payload) == 1:
        (key, value), = payload.items()
        wrapped_action = _ACTION_ALIASES.get(str(key).casefold(), str(key).casefold())
        if isinstance(value, dict) and wrapped_action in {"retrieve", "stop"}:
            ## envelope flattened, no evidence fabricated -- fields come straight from value
            payload = dict(value)
            payload["action"] = wrapped_action
            notes += (f"unwrapped nested envelope: {key} -> action={wrapped_action}",)
    if "action" not in payload and "type" in payload:
        payload["action"] = payload.pop("type")
        notes += ("type -> action",)
    if "action" not in payload and payload.get("query") and (payload.get("sources") or payload.get("source")):
        ## no evidence fabricated, only the action label is inferred from an unambiguous shape
        payload["action"] = "retrieve"
        notes += ("inferred action=retrieve from query+sources shape",)
    action = str(payload.get("action", "")).casefold()
    alias = _ACTION_ALIASES.get(action)
    if alias:
        payload["action"] = alias
        notes += (f"action alias: {action} -> {alias}",)
        action = alias
    if "sources" not in payload and "source" in payload:
        payload["sources"] = [payload.pop("source")]
        notes += ("source -> sources",)
    if action == "stop":
        cleared = [key for key in ("query", "sources", "basis_uris") if payload.get(key)]
        for key in cleared:
            payload.pop(key)
        if cleared:
            notes += (f"cleared stray fields on stop: {', '.join(cleared)}",)
        reason = str(payload.get("stop_reason", "")).casefold()
        reason_alias = _STOP_REASON_ALIASES.get(reason)
        if reason_alias:
            payload["stop_reason"] = reason_alias
            notes += (f"stop_reason alias: {reason} -> {reason_alias}",)
    notes += _normalize_claims(payload, "claims")
    notes += _normalize_claims(payload, "supported_claims")
    return payload, notes


def _text_content(content) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
            else:
                text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts) or None
    return None


def text_content(content) -> str:
    return _text_content(content) or ""


async def typed_call(
    model,
    schema,
    messages,
    *,
    config,
    output_mode,
    timeout_s: int,
    retries: int,
    request_key: str = "",
):
    if output_mode == RetrievalOutputMode.RAW_JSON:
        value, used = await bounded_ainvoke(
            model, messages, config=config, timeout_s=timeout_s, retries=retries,
            request_key=request_key,
        )
        content = getattr(value, "content", value)
    else:
        ## include_raw=True: with_structured_output's own validator uses field names
        ## looser than the pydantic schema (e.g. "type" not "action"); we normalize
        ## and validate the raw completion ourselves instead of trusting it
        runner = model.with_structured_output(schema, method="json_schema", include_raw=True)
        value, used = await bounded_ainvoke(
            runner, messages, config=config, timeout_s=timeout_s, retries=retries,
            request_key=request_key,
        )
        raw = value.get("raw") if isinstance(value, dict) else None
        content = getattr(raw, "content", raw)

    content = _text_content(content)
    if content is None:
        raise ValueError("planner returned non-text content")
    start = content.find("{")
    if start < 0:
        raise ValueError("planner returned no JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(content[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"planner returned invalid JSON: {exc}") from exc
    payload, notes = normalize_provider_payload(payload)
    return schema.model_validate(payload), notes, used
