from core.config import Retrieve_param
from core.schema.retrieval import RetrievalValidity
from TA.retrieval.policy import resolve_retrieval_context, validate_retrieval_artifacts


def _context(preset: str):
    return resolve_retrieval_context(Retrieve_param.from_preset(preset))


def _artifact(*, chunks=None, error=""):
    return {
        "tool": "semantic_search",
        "source": "semantic",
        "args": {"query": "q"},
        "chunks": chunks or [],
        "error": error,
    }


def test_plain_without_retrieval_is_valid():
    result = validate_retrieval_artifacts(_context("PLAIN"), [])

    assert result.validity is RetrievalValidity.VALID
    assert result.attempted_calls == 0


def test_required_retrieval_without_attempt_is_policy_invalid():
    result = validate_retrieval_artifacts(_context("RAG"), [])

    assert result.validity is RetrievalValidity.POLICY_INVALID


def test_successful_empty_retrieval_is_valid():
    result = validate_retrieval_artifacts(_context("RAG"), [_artifact()])

    assert result.validity is RetrievalValidity.VALID


def test_repository_failure_is_invalid():
    result = validate_retrieval_artifacts(
        _context("FULL"),
        [_artifact(error="TimeoutError: unavailable")],
    )

    assert result.validity is RetrievalValidity.INVALID
    assert result.errors == ("TimeoutError: unavailable",)
