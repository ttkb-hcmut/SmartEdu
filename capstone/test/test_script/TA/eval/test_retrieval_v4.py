import pytest
from pydantic import ValidationError


def test_retrieve_decision_requires_grounded_query_and_sources():
    from core.schema.retrieval import RetrievalToolId
    from TA.retrieval.schema import ClaimSupport, HopAction, HopDecision

    decision = HopDecision(
        action=HopAction.RETRIEVE,
        supported_claims=[ClaimSupport(claim="A founded B", evidence_uris=["p1"])],
        missing_link="B's location",
        query="Where is B located?",
        sources=[RetrievalToolId.SEMANTIC],
        basis_uris=["p1"],
    )

    assert decision.query == "Where is B located?"
    assert decision.basis_uris == ["p1"]


@pytest.mark.parametrize(
    "changes",
    [
        {"query": ""},
        {"sources": []},
        {"basis_uris": []},
        {"missing_link": ""},
    ],
)
def test_retrieve_decision_rejects_missing_grounding(changes):
    from core.schema.retrieval import RetrievalToolId
    from TA.retrieval.schema import HopAction, HopDecision

    values = {
        "action": HopAction.RETRIEVE,
        "missing_link": "missing",
        "query": "focused query",
        "sources": [RetrievalToolId.SEMANTIC],
        "basis_uris": ["p1"],
    }
    values.update(changes)

    with pytest.raises(ValidationError):
        HopDecision(**values)


def test_stop_decision_has_no_retrieval_arguments():
    from TA.retrieval.schema import HopAction, HopDecision, StopReason

    decision = HopDecision(
        action=HopAction.STOP,
        stop_reason=StopReason.CHAIN_COMPLETE,
    )

    assert decision.query == ""
    assert decision.sources == []
    assert decision.basis_uris == []


def test_stop_decision_requires_reason_and_rejects_query():
    from TA.retrieval.schema import HopAction, HopDecision, StopReason

    with pytest.raises(ValidationError):
        HopDecision(action=HopAction.STOP)
    with pytest.raises(ValidationError):
        HopDecision(
            action=HopAction.STOP,
            stop_reason=StopReason.CHAIN_COMPLETE,
            query="should not run",
        )


def test_final_chain_requires_cited_claims_when_answerable():
    from TA.retrieval.schema import ClaimSupport, FinalChain

    chain = FinalChain(
        claims=[ClaimSupport(claim="A caused B", evidence_uris=["p1", "p2"])],
        answerable=True,
    )
    assert chain.claims[0].evidence_uris == ["p1", "p2"]

    with pytest.raises(ValidationError):
        FinalChain(answerable=True)
