# ADR-0003: One Neo4j graph (knowledge only); all student data in Mongo

**Status:** Accepted (2026-06-12)

## Context

Student progress was dual-written: Neo4j `(s:Student)-[:MASTERY]->(Entity)` /
`LEARNED_PATH` edges AND Mongo session state via `Student_Tracker` — two sources of
truth. Storing per-student graphs in Neo4j raised cost/scale concerns; Neo4j community
edition is single-database.

## Decision

- Neo4j holds the knowledge layer ONLY (shared Entity nodes + LLM-extracted edges).
- MongoDB single-sources ALL student data: the Learning Tree (per student × course doc
  with per-node mastery, status, left→right order), the attempt log, active exercise,
  and last lecture.
- Neo4j `MASTERY`/`LEARNED_PATH` student edges are deprecated (read-compatible until
  teach-as-search lands, then removed).
- Joins stay cheap: the cached Course Tree JSON carries `requires` annotations, so
  readiness/frontier checks join two Mongo documents in app code; Neo4j is touched only
  for discovery beyond the backbone.

## Consequences

- One source of truth; the dual-write bug class disappears.
- Graph-shaped per-student queries (e.g. Cypher paths filtered by mastery) move to app
  code — acceptable at current scale.
- After course re-ingestion, Mongo concept references can dangle → offline
  dangling-reference check required (deferred analytics phase).
