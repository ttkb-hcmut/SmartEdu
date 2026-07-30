import asyncio
from time import perf_counter
from typing import Any, Literal, Type

from langchain.tools import BaseTool, ToolRuntime
from pydantic import BaseModel, Field

from core.schema.retrieval import RetrievalRunContext, RetrievalToolId
from TA.retrieval.policy import get_tool_spec


class RetrievalQueryInput(BaseModel):
    query: str = Field(description="Text query to search for.")


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
                top_k=context.harness.top_k,
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
                top_k=context.harness.top_k,
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
