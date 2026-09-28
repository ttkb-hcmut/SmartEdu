import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from core.schema.wf_state import AgentState
from core.repo.storage.minio_repo import resolve_pdf_reference
from TA.retrieval.schema import FinalChain

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceContextPolicy:
    excerpt_chars: int
    max_chars: int
    max_uncited: Optional[int] = None


@dataclass(frozen=True)
class CompiledEvidenceContext:
    text: str
    uris: tuple[str, ...]
    source_chars: int
    emitted_chars: int
    truncated: bool


def _ordered_ledger(
    ledger: Iterable[dict[str, Any]],
    chain: FinalChain,
    max_uncited: Optional[int] = None,
) -> list[dict[str, Any]]:
    by_uri: dict[str, dict[str, Any]] = {}
    ledger_order: list[str] = []
    for item in ledger:
        uri = str(item.get("uri", ""))
        if uri and uri not in by_uri:
            by_uri[uri] = item
            ledger_order.append(uri)
    cited = [
        uri
        for claim in (chain.claims if chain else [])
        for uri in claim.evidence_uris
        if uri in by_uri
    ]
    # ponytail: bound uncited distractors to prevent context poisoning in the answerer
    cited_unique = list(dict.fromkeys(cited))
    if max_uncited is not None:
        uncited = [uri for uri in ledger_order if uri not in cited_unique][:max_uncited]
        ordered_keys = [*cited_unique, *uncited]
    else:
        ordered_keys = list(dict.fromkeys([*cited_unique, *ledger_order]))
    return [by_uri[uri] for uri in ordered_keys]


def compile_evidence_context(
    ledger: Iterable[dict[str, Any]],
    chain: FinalChain,
    policy: EvidenceContextPolicy,
) -> CompiledEvidenceContext:
    if policy.excerpt_chars < 0 or policy.max_chars < 0:
        raise ValueError("evidence context limits must be non-negative")
    ordered = _ordered_ledger(ledger, chain, policy.max_uncited)
    source_chars = sum(len(str(item.get("text", ""))) for item in ordered)
    lines: list[str] = []
    uris: list[str] = []
    remaining = policy.max_chars
    truncated = False
    for index, item in enumerate(ordered):
        uri = str(item["uri"])
        excerpt = str(item.get("text", ""))[:policy.excerpt_chars]
        prefix = f"- [{uri}] "
        separator = 1 if lines else 0
        available = remaining - separator
        line = prefix + excerpt
        if len(line) <= available:
            lines.append(line)
            uris.append(uri)
            remaining -= len(line) + separator
            continue
        truncated = True
        if available > len(prefix):
            lines.append(prefix + excerpt[:available - len(prefix)])
            uris.append(uri)
        if index < len(ordered) - 1:
            truncated = True
        break
    text = "\n".join(lines)
    return CompiledEvidenceContext(
        text=text,
        uris=tuple(uris),
        source_chars=source_chars,
        emitted_chars=len(text),
        truncated=truncated,
    )


async def build_ui_citations(rag_result: dict, graph_db=None) -> list[dict]:
    cited = rag_result.get("cited_evidence") or []
    if not cited:
        return []
    semantic_ids = [item["uri"] for item in cited if item.get("source") == "semantic"]
    sources = {}
    if semantic_ids and graph_db is not None:
        try:
            sources = {
                row["id"]: row
                for row in await asyncio.to_thread(graph_db.resolve_entity_sources, semantic_ids)
            }
        except Exception:
            logger.exception("Semantic citation resolution failed")
    citations = []
    for item in cited:
        uri = str(item["uri"])
        source = sources.get(uri, {}) if item.get("source") == "semantic" else item
        document, page = resolve_pdf_reference(source)
        citations.append({"uri": uri, "document": document, "page": page})
    return citations


def extract_ta_context(state: AgentState, max_msgs: int = 2) -> str:
    messages = state.get("messages", [])
    ta_msgs = []
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            ta_msgs.append(msg.content)
        elif isinstance(msg, dict) and msg.get("role") == "assistant":
            ta_msgs.append(msg["content"])
        if len(ta_msgs) >= max_msgs:
            break
    return "\n---\n".join(reversed(ta_msgs))
