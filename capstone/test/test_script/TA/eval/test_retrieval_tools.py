from types import SimpleNamespace

from core.config import Retrieve_param
from TA.retrieval.policy import resolve_retrieval_context


class _Milvus:
    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.rows


class _Graph:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.calls = []

    def passage_search(self, emb, **kwargs):
        self.calls.append((emb, kwargs))
        return self.rows


class _Embedder:
    def get_embedding(self, text):
        return [0.1, 0.2]


def _runtime(preset="FULL", course="Bench_MuSiQue", *, policy_id=None, harness_id=None):
    kwargs = {"course_scope": course}
    if policy_id is not None:
        kwargs["policy_id"] = policy_id
    if harness_id is not None:
        kwargs["harness_id"] = harness_id
    context = resolve_retrieval_context(
        Retrieve_param.from_preset(preset, **kwargs)
    )
    return SimpleNamespace(context=context)


def test_retrieval_tool_schemas_expose_query_only():
    from TA.tools.retrieval import SemanticSearch, TextbookSearch

    semantic = SemanticSearch(milvus_db=_Milvus(), embedder=_Embedder())
    textbook = TextbookSearch(graph_db=_Graph(), embedder=_Embedder())

    assert set(semantic.get_input_schema().model_fields) == {"query"}
    assert set(textbook.get_input_schema().model_fields) == {"query"}


def test_semantic_search_injects_scope_and_returns_artifact():
    from TA.tools.retrieval import SemanticSearch

    db = _Milvus(rows=[{"id": "p_1", "text": "evidence", "score": 0.9}])
    tool = SemanticSearch(milvus_db=db, embedder=_Embedder())

    content, artifact = tool._run("question", runtime=_runtime("RAG"))

    assert db.calls == [{
        "query": "question",
        "embedder": tool.embedder,
        "top_k": 5,
        "course_scope": "Bench_MuSiQue",
    }]
    assert "evidence" in content
    assert artifact["error"] == ""
    assert artifact["args"] == {"query": "question"}
    assert artifact["chunks"] == [{
        "id": "p_1",
        "uri": "p_1",
        "text": "evidence",
        "score": 0.9,
        "source": "semantic",
    }]


def test_textbook_search_uses_exact_course_prefix():
    from TA.tools.retrieval import TextbookSearch

    graph = _Graph(rows=[{
        "id": "p_2",
        "uri": "Bench_MuSiQue/p_2",
        "text": "book evidence",
        "score": 0.8,
    }])
    tool = TextbookSearch(graph_db=graph, embedder=_Embedder())

    _, artifact = tool._run("question", runtime=_runtime())

    assert graph.calls == [([0.1, 0.2], {
        "query_text": "question",
        "top_k": 5,
        "uri_prefix": "Bench_MuSiQue/",
    })]
    assert artifact["chunks"][0]["source"] == "textbook"


def test_tool_failure_is_returned_as_error_artifact():
    from TA.tools.retrieval import SemanticSearch

    tool = SemanticSearch(
        milvus_db=_Milvus(error=RuntimeError("database unavailable")),
        embedder=_Embedder(),
    )

    content, artifact = tool._run("question", runtime=_runtime("RAG"))

    assert content.startswith("ERROR:")
    assert artifact["chunks"] == []
    assert artifact["error"] == "RuntimeError: database unavailable"


def test_v4_adapters_request_eight_candidates_per_source():
    from core.schema.retrieval import RetrievalHarnessId, RetrievalPolicyId
    from TA.tools.retrieval import SemanticSearch, TextbookSearch

    runtime = _runtime(
        policy_id=RetrievalPolicyId.BASELINE_V5,
        harness_id=RetrievalHarnessId.AGENTIC_V4,
    )
    milvus = _Milvus()
    graph = _Graph()

    SemanticSearch(milvus_db=milvus, embedder=_Embedder())._run("q", runtime=runtime)
    TextbookSearch(graph_db=graph, embedder=_Embedder())._run("q", runtime=runtime)

    assert milvus.calls[0]["top_k"] == 8
    assert graph.calls[0][1]["top_k"] == 8
