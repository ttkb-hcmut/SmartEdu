# ADR-0006: Video is an anchor-only substrate; novelty is stored, not decided

**Status:** Accepted (2026-07-17), amended (2026-07-31) — Option 1 is the current contract

## Context

Video ingestion (Video → faster-whisper transcript → valley-cut `:Segment` nodes) needs
a stance on segments that match no existing concept. The field evidence
(docs/description/research-note.md): absolute-threshold novelty is the entity-linking
literature's known-weak baseline (BLINKout, CIKM 2023); proper NIL detection needs a
trained classifier plus labeled data we do not have; LMMs up to 40B measurably struggle
at lecture comprehension (Video-MMLU), so an Ollama-scale LLM judge is indefensible;
the largest deployed lecture-annotation system (X5GON, ~20K lectures) runs anchor-only
against a fixed concept inventory with no generative extraction; SOTA LLM KG-extraction
reaches ~66% fidelity on clean written text before ASR noise (KGGen/MINE).

## Decision

- **Video births no concepts in v1.** Segments are anchored to slide-born concepts via
  batched ANN, mirroring the textbook substrate doctrine. No LLM runs on video content.
  "Extract-on-novelty" moves to v2, to be decided *with* the collected score data.
- **Store the signal; use an advisory floor.** Per segment, persist the ordered top-k candidate profile
  (`candidate_ids`, `candidate_names`, `candidate_scores`, `anchor_model`,
  `anchor_version`, `best_score`). The advisory `novel_candidate` flag currently means
  no candidate exists or `best_score < anchor_score_min`; this absolute floor is
  uncalibrated and does not prove out-of-KB content. The BookRAG-derived gradient walk
  controls how many candidates become anchors, not the flag.
- **Advisory-only.** Nothing downstream mutates the KG from the flag; it surfaces in
  the ingest report (segment counts, novel-candidate timestamps) and is queryable on
  `:Segment`.

## Consequences

- Reversible by construction: candidate identities keep every score attributable, so a
  future rule can back-fill flags and accepted anchors from stored profiles.
- The stored profiles are simultaneously the future NIL classifier's training features
  and the input to BLINKout's KB-pruning calibration eval (hide N slide concepts,
  verify their segments score low) — runnable for free once data accumulates.
- A lecture covering material absent from the slides yields only advisory flags in v1;
  teacher-facing surfacing beyond the ingest report is deferred.

## Amendment (2026-07-31): the gradient walk does not currently affect novelty

Analysis of the shipped implementation (`docs/description/ingest_v1.7.md` §6, verified against
`core/ingest/novelty.py` and `knowledge/ingest/vid.py`) showed that the original "Novelty = B+D"
decision, corrected above, overstated what runs.

`cliff_partition` unconditionally keeps index `0`, and `vid.py` sorts hits descending before calling
it. So every kept index `i` satisfies `scores[i] <= scores[0]`, and the novelty test

```python
novel = not any(scores[i] >= floor for i in kept)
```

reduces exactly to `scores[0] < floor` — i.e. `best_score < anchor_score_min` (0.55). The gradient
ratio `g` changes *which extra candidates are retained as anchors*, and therefore how many
`ANCHORED_IN` edges are written, but it cannot change the boolean. **The shipped novelty rule is
option B (absolute threshold) alone; option D (gradient-cliff) is live for anchor selection only.**

This is not a defect in the code — the code does what it says. It was a defect in this ADR's claim,
and it matters because BLINKout (CIKM 2023) identifies absolute-threshold novelty as the known-weak
baseline this decision was written to avoid.

### Options considered for restoring a shape-sensitive rule

The stored candidate profiles (`candidate_scores`, `best_score`, `anchor_version`) make all of these
evaluable offline against recorded runs, with no Whisper re-run. That is the point of B+D's storage
half, and it survives this amendment intact.

**Option 1 — accept and rename.** Keep `best_score < floor` as the rule and correct this ADR to say
"absolute floor, with candidate profiles stored for later calibration." Zero code change, zero risk,
and honest. The cost is that novelty stays the weak baseline, and the `anchor_gradient_g` knob keeps
a name implying influence it does not have over the flag.

**Option 2 — cliff-shape novelty (makes D real).** Flag novel when the score curve is *flat and low*
rather than merely below a floor: no candidate clears the floor **and** the retained set shows no
cliff (`scores[0] - scores[k]` small across the kept run). A segment matching one concept sharply
(0.72, 0.31, 0.20) is clearly anchored; a segment scoring (0.51, 0.49, 0.48) is flat-and-low and is
the interesting novel case that the current rule and this rule agree on — but a segment at
(0.58, 0.56, 0.55) is a *plateau of moderate* scores, which the current rule calls anchored and this
rule would still call anchored. The real behavioral difference appears just under the floor, where
shape distinguishes "weakly related to several concepts" from "related to nothing." Small change to
`cliff_partition`'s return, no new dependency, and it makes `g` load-bearing for the flag.

**Option 3 — margin-to-second (relative confidence).** Flag on `scores[0] - scores[1]` rather than
`scores[0]` alone. Cheap, well-understood in entity linking, and directly targets the ambiguous case
where a segment sits between two concepts. Weakness: it says nothing when only one candidate returns,
which is common in a small course-scoped collection.

**Option 4 — calibrated NIL detection (the BLINKout-faithful answer).** Run the KB-pruning eval this
ADR already anticipates: hide N slide-born concepts, re-anchor, and check that segments teaching the
hidden concepts score low. That yields a *measured* floor and a precision/recall curve instead of a
guessed 0.55. Requires accumulated lecture data and is the only option that produces a defensible
number rather than a better-shaped guess.

### Decision on the amendment

**Adopt Option 1 now** — this ADR is corrected to describe the shipped rule accurately, and
`anchor_gradient_g` is documented as governing anchor breadth, not novelty. Options 2–4 remain open
and are explicitly *not* chosen on intuition: 0.55 and `g=1.5` are both uncalibrated defaults, so
replacing one uncalibrated heuristic with a more elaborate uncalibrated heuristic buys complexity
without evidence. Option 4 is the intended path once real lecture runs accumulate; Option 2 is the
cheapest upgrade if a shape-sensitive flag is wanted before that data exists.

The advisory-only property is unchanged: nothing downstream mutates the KG from the flag, so a later
rule change re-derives flags from stored profiles without reprocessing video.
