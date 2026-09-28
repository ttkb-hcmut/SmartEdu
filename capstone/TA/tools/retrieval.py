import asyncio
import logging
import re
import unicodedata
from time import perf_counter
from typing import Any, Literal, Optional, Type

from langchain.tools import BaseTool, ToolRuntime
from pydantic import BaseModel, Field

from core.schema.retrieval import RetrievalHarnessId, RetrievalRunContext, RetrievalToolId
from TA.retrieval.policy import get_tool_spec

logger = logging.getLogger(__name__)

ALLOWED_ENTITY_LABELS = {"PERSON", "ORG", "EVENT", "WORK_OF_ART", "PRODUCT", "GPE", "LOC"}
_ARTICLE_PREFIX_RE = re.compile(r"^(the|a|an)\s+", re.I)


def _clean_entity_text(raw: str) -> str:
    cleaned = unicodedata.normalize("NFKC", raw).strip()
    cleaned = _ARTICLE_PREFIX_RE.sub("", cleaned).strip()
    return cleaned.strip("\"'.,;:()[]{}")


def _is_valid_entity(name: str) -> bool:
    return len(name) >= 3 and not name.isdigit() and any(c.isalnum() for c in name)


_SPACY_NLP = None


def _get_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])
        except Exception as exc:
            logger.debug("spaCy unavailable for retrieval entity extraction: %s", exc)
            _SPACY_NLP = False
    return _SPACY_NLP if _SPACY_NLP is not False else None


class RetrievalQueryInput(BaseModel):
    query: str = Field(description="Text query to search for.")


class RetrieveMoreInput(RetrievalQueryInput):
    sources: list[str] = Field(
        description=(
            "Retrieval sources to search in parallel. Use only semantic or textbook; "
            "empty means conclude from the ledger."
        ),
    )


def _chunk(row: dict, source: str) -> dict:
    chunk = {
        "id": row.get("id"),
        "uri": row.get("id") or row.get("uri"),
        "text": row.get("text", ""),
        "score": float(row.get("score", 0.0) or 0.0),
        "source": source,
    }
    if source != RetrievalToolId.SEMANTIC.value and row.get("uri"):
        chunk["document_uri"] = row["uri"]
    for key in ("p_lo", "p_hi"):
        if row.get(key) is not None:
            chunk[key] = row[key]
    return chunk


def _artifact(tool: str, source: str, query: str, chunks: list, start: float, error: str = "") -> dict:
    return {
        "tool": tool,
        "source": source,
        "args": {"query": query},
        "chunks": chunks,
        "latency_ms": (perf_counter() - start) * 1000,
        "error": error,
    }


def _source_k(context: RetrievalRunContext) -> int:
    if context.harness.id in {RetrievalHarnessId.AGENTIC_V3, RetrievalHarnessId.AGENTIC_V4}:
        return context.harness.per_source_k
    return context.harness.top_k


def rank_round(pools: dict[str, list[dict]], rrf_k: int) -> list[dict]:
    ranked: dict[str, dict] = {}
    order: list[str] = []
    for source, chunks in pools.items():
        for rank, chunk in enumerate(chunks, start=1):
            uri = str(chunk["uri"])
            if uri not in ranked:
                ranked[uri] = {
                    **chunk,
                    "rrf_score": 0.0,
                    "source_ranks": {},
                    "source_scores": {},
                }
                order.append(uri)
            item = ranked[uri]
            item["rrf_score"] += 1.0 / (rrf_k + rank)
            item["source_ranks"][source] = rank
            item["source_scores"][source] = float(chunk.get("score", 0.0) or 0.0)
    position = {uri: index for index, uri in enumerate(order)}
    return sorted(
        ranked.values(),
        key=lambda item: (-item["rrf_score"], position[str(item["uri"])]),
    )


def append_ledger(
    ledger: list[dict],
    ranked: list[dict],
    *,
    round_index: int,
    query: str,
) -> tuple[list[dict], int, int]:
    result = [{**item, "occurrences": list(item.get("occurrences", []))} for item in ledger]
    by_uri = {str(item["uri"]): item for item in result}
    new_count = 0
    duplicate_count = 0
    for chunk in ranked:
        uri = str(chunk["uri"])
        occurrences = [
            {
                "round": round_index,
                "query": query,
                "source": source,
                "rank": rank,
                "score": chunk.get("source_scores", {}).get(source, chunk.get("score", 0.0)),
                "rrf_score": chunk.get("rrf_score", 0.0),
            }
            for source, rank in chunk.get("source_ranks", {}).items()
        ]
        if uri in by_uri:
            duplicate_count += 1
            by_uri[uri]["occurrences"].extend(occurrences)
            continue
        item = {**chunk, "first_seen_round": round_index, "occurrences": occurrences}
        result.append(item)
        by_uri[uri] = item
        new_count += 1
    return result, new_count, duplicate_count


class SemanticSearch(BaseTool):
    _spec = get_tool_spec(RetrievalToolId.SEMANTIC)
    name: str = _spec.name
    description: str = _spec.description
    args_schema: Type[BaseModel] = RetrievalQueryInput
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    milvus_db: Any = Field(exclude=True)
    embedder: Any = Field(exclude=True)

    def _run(self, query: str, runtime: ToolRuntime[RetrievalRunContext]):
        start = perf_counter()
        try:
            context = runtime.context
            rows = self.milvus_db.search(
                query=query,
                embedder=self.embedder,
                top_k=_source_k(context),
                course_scope=context.scope.course,
            )
            chunks = [_chunk(row, RetrievalToolId.SEMANTIC.value) for row in rows]
            content = "\n".join(f"- [{item['score']:.4f}] {item['text']}" for item in chunks)
            if not content:
                content = f"INFO: No semantic matches found for '{query}'."
            return content, _artifact(self.name, "semantic", query, chunks, start)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return f"ERROR: {error}", _artifact(self.name, "semantic", query, [], start, error)

    async def _arun(self, query: str, runtime: ToolRuntime[RetrievalRunContext]):
        return await asyncio.to_thread(self._run, query, runtime)


class TextbookSearch(BaseTool):
    _spec = get_tool_spec(RetrievalToolId.TEXTBOOK)
    name: str = _spec.name
    description: str = _spec.description
    args_schema: Type[BaseModel] = RetrievalQueryInput
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    graph_db: Any = Field(exclude=True)
    embedder: Any = Field(exclude=True)

    def _run(
        self,
        query: str,
        runtime: ToolRuntime[RetrievalRunContext],
        seed_uris: Optional[list[str]] = None,
    ):
        start = perf_counter()
        try:
            context = runtime.context
            scope = context.scope.course.rstrip("/")
            k = _source_k(context)
            rows = self.graph_db.passage_search(
                self.embedder.get_embedding(query),
                query_text=query,
                top_k=k,
                uri_prefix=f"{scope}/" if scope else None,
            )
            chunks = [_chunk(row, RetrievalToolId.TEXTBOOK.value) for row in rows]

            # Graph traversal from seed passages (e.g. prior ledger/basis URIs)
            graph_chunks = []
            if seed_uris and hasattr(self.graph_db, "graph_expand_passages"):
                try:
                    graph_rows = self.graph_db.graph_expand_passages(
                        seed_uris,
                        top_k=k,
                        max_degree=50,
                    )
                    graph_chunks = [_chunk(row, "graph_traversal") for row in graph_rows]
                except Exception as g_exc:
                    logger.warning("graph_expand_passages failed: %s", g_exc)

            # Query-driven entity lookup
            entity_chunks = []
            if hasattr(self.graph_db, "entity_passage_search"):
                try:
                    nlp = _get_spacy()
                    if nlp:
                        doc = nlp(query)
                        ent_names = [
                            _clean_entity_text(e.text)
                            for e in doc.ents
                            if e.label_ in ALLOWED_ENTITY_LABELS and _is_valid_entity(_clean_entity_text(e.text))
                        ]
                        if ent_names:
                            ent_rows = self.graph_db.entity_passage_search(
                                ent_names,
                                top_k=k,
                                max_degree=50,
                            )
                            entity_chunks = [_chunk(row, "graph_entity") for row in ent_rows]
                except Exception as e_exc:
                    logger.warning("entity_passage_search failed: %s", e_exc)

            if graph_chunks or entity_chunks:
                pools = {"textbook": chunks}
                if graph_chunks:
                    pools["graph_traversal"] = graph_chunks
                if entity_chunks:
                    pools["graph_entity"] = entity_chunks
                ranked = rank_round(pools, context.harness.rrf_k)
                chunks = [{**c, "source": RetrievalToolId.TEXTBOOK.value} for c in ranked[:k]]

            content = "\n".join(f"- [{item['score']:.4f}] {item['text']}" for item in chunks)
            if not content:
                content = f"INFO: No textbook matches found for '{query}'."
            return content, _artifact(self.name, "textbook", query, chunks, start)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return f"ERROR: {error}", _artifact(self.name, "textbook", query, [], start, error)

    async def _arun(
        self,
        query: str,
        runtime: ToolRuntime[RetrievalRunContext],
        seed_uris: Optional[list[str]] = None,
    ):
        return await asyncio.to_thread(self._run, query, runtime, seed_uris)


class RetrieveMore(BaseTool):
    name: str = "retrieve_more"
    description: str = (
        "Search selected retrieval sources in parallel for one focused follow-up query. "
        "Use discovered entities and missing links from prior evidence."
    )
    args_schema: Type[BaseModel] = RetrieveMoreInput
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    semantic: Any = Field(exclude=True)
    textbook: Any = Field(exclude=True)

    async def _arun(
        self,
        query: str,
        sources: list[str],
        runtime: ToolRuntime[RetrievalRunContext],
        seed_uris: Optional[list[str]] = None,
    ):
        started = perf_counter()
        allowed = {tool.value for tool in runtime.context.policy.allowed_tools}
        requested = list(dict.fromkeys(sources))
        executed = [source for source in requested if source in allowed]
        invalid = [source for source in requested if source not in allowed]
        # ponytail: when seed_uris exist (entity hop) and textbook graph traversal is allowed,
        # ensure textbook is executed even if planner prompt example defaulted to semantic only.
        if seed_uris and "textbook" in allowed and "textbook" not in executed:
            executed.append("textbook")
        if not executed:
            content = f"No requested source is allowed. Allowed sources: {', '.join(sorted(allowed))}."
            return content, {
                "tool": self.name,
                "source": "parallel",
                "args": {"query": query, "sources": requested},
                "requested_sources": requested,
                "executed_sources": [],
                "invalid_sources": invalid,
                "source_artifacts": [],
                "chunks": [],
                "latency_ms": (perf_counter() - started) * 1000,
                "error": "",
            }

        adapters = {"semantic": self.semantic, "textbook": self.textbook}

        def _call_adapter(s):
            adapter = adapters[s]
            if s == "textbook" and seed_uris:
                try:
                    return adapter._arun(query, runtime=runtime, seed_uris=seed_uris)
                except TypeError:
                    return adapter._arun(query, runtime=runtime)
            return adapter._arun(query, runtime=runtime)

        results = await asyncio.gather(
            *(_call_adapter(source) for source in executed)
        )
        source_artifacts = [artifact for _, artifact in results]
        pools = {artifact["source"]: artifact.get("chunks", []) for artifact in source_artifacts}
        ranked = rank_round(pools, runtime.context.harness.rrf_k)
        errors = [artifact["error"] for artifact in source_artifacts if artifact.get("error")]
        excerpt_chars = runtime.context.harness.evidence_excerpt_chars or 1_000
        content = "\n".join(
            f"- [{item['uri']}] {item.get('text', '')[:excerpt_chars]}" for item in ranked
        ) or "No matching evidence found."
        return content, {
            "tool": self.name,
            "source": "parallel",
            "args": {"query": query, "sources": requested},
            "requested_sources": requested,
            "executed_sources": executed,
            "invalid_sources": invalid,
            "source_artifacts": source_artifacts,
            "chunks": ranked,
            "latency_ms": (perf_counter() - started) * 1000,
            "error": "; ".join(errors),
        }

    def _run(self, query: str, sources: list[str], runtime: ToolRuntime[RetrievalRunContext]):
        return asyncio.run(self._arun(query, sources, runtime))
