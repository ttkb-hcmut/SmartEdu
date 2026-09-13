from enum import Enum

from pydantic import BaseModel, Field, model_validator

from core.schema.retrieval import RetrievalToolId


class HopAction(str, Enum):
    RETRIEVE = "retrieve"
    STOP = "stop"


class StopReason(str, Enum):
    CHAIN_COMPLETE = "chain_complete"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTEXT_LIMIT = "context_limit"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ClaimSupport(BaseModel):
    claim: str = Field(min_length=1)
    evidence_uris: list[str] = Field(min_length=1)


class HopDecision(BaseModel):
    action: HopAction
    supported_claims: list[ClaimSupport] = Field(default_factory=list)
    sub_question: str = ""
    query: str = ""
    sources: list[RetrievalToolId] = Field(default_factory=list)
    basis_uris: list[str] = Field(default_factory=list)
    stop_reason: StopReason | None = None

    @model_validator(mode="after")
    def validate_action(self):
        if self.action is HopAction.RETRIEVE:
            if not self.query.strip() or not self.sub_question.strip():
                raise ValueError("retrieve decision requires query and sub_question")
            if not self.sources or not self.basis_uris:
                raise ValueError("retrieve decision requires sources and basis URIs")
            if self.stop_reason is not None:
                raise ValueError("retrieve decision cannot have stop reason")
        else:
            if self.stop_reason is None:
                raise ValueError("stop decision requires reason")
            if self.query.strip() or self.sources or self.basis_uris:
                raise ValueError("stop decision cannot request retrieval")
        return self


class FinalChain(BaseModel):
    claims: list[ClaimSupport] = Field(default_factory=list)
    answerable: bool
    remaining_uncertainty: str = ""

    @model_validator(mode="after")
    def validate_answerable(self):
        if self.answerable and not self.claims:
            raise ValueError("answerable chain requires cited claims")
        return self
