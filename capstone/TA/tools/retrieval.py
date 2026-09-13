import asyncio
from time import perf_counter
from typing import Any, Literal, Type

from langchain.tools import BaseTool, ToolRuntime
from pydantic import BaseModel, Field

from core.schema.retrieval import RetrievalHarnessId, RetrievalRunContext, RetrievalToolId
from TA.retrieval.policy import get_tool_spec


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
    return {
        "id": row.get("id"),
        "uri": row.get("uri") or row.get("id"),
        "text": row.get("text", ""),
        "score": float(row.get("score", 0.0) or 0.0),
        "source": source,
    }


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

    def _run(self, query: str, runtime: ToolRuntime[RetrievalRunContext]):
        start = perf_counter()
        try:
            context = runtime.context
            scope = context.scope.course.rstrip("/")
            rows = self.graph_db.passage_search(
                self.embedder.get_embedding(query),
                query_text=query,
                top_k=_source_k(context),
                uri_prefix=f"{scope}/" if scope else None,
            )
            chunks = [_chunk(row, RetrievalToolId.TEXTBOOK.value) for row in rows]
            content = "\n".join(f"- [{item['score']:.4f}] {item['text']}" for item in chunks)
            if not content:
                content = f"INFO: No textbook matches found for '{query}'."
            return content, _artifact(self.name, "textbook", query, chunks, start)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return f"ERROR: {error}", _artifact(self.name, "textbook", query, [], start, error)

    async def _arun(self, query: str, runtime: ToolRuntime[RetrievalRunContext]):
        return await asyncio.to_thread(self._run, query, runtime)


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
    ):
        started = perf_counter()
        allowed = {tool.value for tool in runtime.context.policy.allowed_tools}
        requested = list(dict.fromkeys(sources))
        executed = [source for source in requested if source in allowed]
        invalid = [source for source in requested if source not in allowed]
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
        results = await asyncio.gather(
            *(adapters[source]._arun(query, runtime=runtime) for source in executed)
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
