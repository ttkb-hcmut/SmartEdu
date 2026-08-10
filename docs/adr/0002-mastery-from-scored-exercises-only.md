# ADR-0002: Mastery is written only from scored exercises

**Status:** Accepted (2026-06-12)

## Context

Previously `Teach_Evaluate` judged "passed" from chat history alone — blind to the lecture
it graded against and backed by no exercise. A mastery state built on such verdicts
propagates noise into roadmap and next-topic decisions (LLM-council finding, problem_wf.md).

## Decision

- Mastery (Bloom 0–6) is written ONLY by the Student Judge from a scored exercise attempt.
- Exercises are **retrieved from valid resources, never free-generated**: the MOOCCubeX
  exercise bank (concept-matched import) first, KG `rrole=Quiz|Problem` content as fallback,
  lightly augmented (rotation/paraphrase) against parrot learning.
- The judge grades against the exercise, its reference solution, AND the lecture actually
  taught. LLM chat impressions may only *suggest* weak areas — never write mastery.
- Every scored attempt also appends to the immutable attempt log (DKT substrate).

## Consequences

- Mastery becomes auditable: every level traces to a concrete attempt record.
- Concepts without exercise content cannot raise mastery above unverified levels —
  acceptable; content gaps surface instead of being papered over.
- Deep Knowledge Tracing research gets a dataset from day one (collect now, model later).
