# ADR-0008: The retrieval ablation is a full-system benchmark on a typed, frozen policy

**Status:** Accepted (2026-07-20)

## Context

The retrieval ablation (PLAIN/RAG/FULL) needs a stance on *what* is measured and *how* the
measurement stays honest across a loop of improvements. Two forces shaped it. First, an
independent review repeatedly proposed stripping the tutor down to a bare retrieval function
for the benchmark — cheaper and more isolated, but it measures a library that is not the
product. Second, an ablation that lets prompt wording, tool exposure, and knobs drift between
runs cannot attribute a score change to the thing under study; the earlier fan-out design also
bound tools once at agent startup, so per-arm tool exposure could not be enforced by
configuration alone.

## Decision

- **Full-`TAModule` retrieve-path benchmark.** The benchmark drives the real tutor end-to-end,
  bypassing only the router (via an explicit forced route). SmartEdu *is* the system under
  test; a direct retrieval runner was rejected because it would benchmark code the product
  never runs.
- **Agentic harness is the default; fanout is a frozen control.** Retrieval executes as an
  agent looping over the arm's tools (multi-hop emerges from repeated calls). The deterministic
  fan-out+RRF harness is retained, versioned, and selectable as a control. Every benchmark
  command names its harness; there is **no automatic fallback** between harnesses — a row
  labelled `agentic` can never secretly be `fanout`.
- **Arms are toolsets under one typed policy.** PLAIN → [] · RAG → [semantic] · FULL →
  [semantic, textbook]. The policy is a frozen, code-registered contract (prompt, per-preset
  toolset, minimum calls, limits) with a versioned ID and a content **digest**; a test fails if
  an existing ID's digest changes. Any prompt/policy/model change mints a new ID (a new baseline
  family), never an in-place edit. A run context composing preset+policy+harness+scope+case is
  resolved once and injected; the model cannot choose its own scope or tools. Middleware filters
  the registered tool superset per arm and mechanically caps retrieval calls.
- **Validity is mechanical, not LLM-reported.** A retrieval-tool error marks a row *invalid*
  (excluded, reported separately); RAG/FULL with zero retrieval attempts is *policy-invalid*;
  the agent's self-reported status never overrides this.

## Consequences

- The benchmark exercises production wiring (router bypass aside), so a green run is evidence
  about the shipped tutor, at the cost of more moving parts and longer runs than a bare runner.
- Baselines are comparable within a family and explicitly *incomparable* across policy/harness
  versions — the digest + version stamps on every trace and result row make a silent drift a
  test failure rather than a misread table.
- Genuine graph retrieval, query classification, and section-narrowing (BookRAG operators) are
  future levers, each a measured iteration or a new baseline version — never folded silently
  into the current baseline.
- Full live baselines remain blocked until a benchmark corpus loader dual-writes canonical
  paragraphs 1:1 into both stores (owned by the ingestion plan); the runner enforces this with
  a corpus-ready gate.
