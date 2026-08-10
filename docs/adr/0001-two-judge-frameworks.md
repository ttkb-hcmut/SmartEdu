# ADR-0001: Two distinct judge frameworks (Student Judge ≠ Teacher Judge)

**Status:** Accepted (2026-06-12)

## Context

The system needs evaluation on two axes: whether the *student* learned (grading), and
whether the *teaching system itself* behaves well (workflow quality). One shared
LLM-as-a-judge framework was considered.

## Decision

Two fully distinct judge frameworks that never share prompts, schemas, or pipelines:

- **Student Judge** — in-loop, grades student exercise answers against retrieved
  exercise + solution + the lecture actually taught. Strict, rubric-driven,
  clarify-before-fail. Sole writer of mastery.
- **Teacher Judge** — offline, scores `AgentTracer` traces per workflow node against
  per-node rubrics. Its scores are the verifier for harness evolution (ghost proposer).

## Consequences

- Different objects of judgment, rubrics, and failure costs stay isolated; a change to
  grading strictness cannot silently alter system-quality scoring (and vice versa).
- Two harnesses to maintain instead of one.
- Teacher Judge uses a different model family than the actor (self-enhancement bias control).
