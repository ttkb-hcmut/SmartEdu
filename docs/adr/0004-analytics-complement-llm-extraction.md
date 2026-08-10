# ADR-0004: Graph analytics complement LLM extraction, never rewrite it

**Status:** Accepted (2026-06-12) — implementation deferred (cost); planning-only phase

## Context

KG construction uses LLM extraction, which reasons over *local* (slide-window) relations
but cannot see *macro* structure: global concept importance, topic bridges, or
cross-course relations. The DKG community-detection literature (Aparicio et al. 2024,
docs/Ref/KG/Algo/DKG_Comm.pdf) supplies the macro tools: closeness centrality,
betweenness, Louvain communities, KC2 component stitching.

## Decision

- Division of labor: LLM extraction owns micro relations (BELONGS_TO, PART_OF,
  PREREQUISITE, CONTENT). Analytics own macro signals (closeness, betweenness/gateways,
  cross-course relations) — written as node *properties* and *proposed* edges only.
- Analytics never modify or delete LLM-extracted edges.
- Per-course metrics run in NetworkX in-memory at ingestion (no Neo4j GDS dependency).
- Cross-course pass operates on backbone nodes only (memory-safe by construction):
  embedding-similarity candidates → Louvain communities → KC2 stitch ranking → LLM
  semantic verification. Cross-course `PREREQUISITE` edges are human-gated (they alter
  roadmap routing); `SIMILAR_TO` may auto-write after LLM verification.
- Deferred to the last implementation phase on cost; requires its own design session
  before implementation.

## Consequences

- `CourseTree` ranks by `coalesce(n.closeness, out_degree)` — the upgrade lands with
  zero workflow changes when analytics ship.
- Until then, backbone selection uses weighted out-degree (the documented cold-start
  fallback).
