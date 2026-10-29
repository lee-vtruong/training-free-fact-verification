# M1.6.2d — Decision & Status: Error Information Feasibility

**Internal name:** M1.6.3a (user-proposed) — **repository ID: M1.6.2d** to avoid any implication of
activating the preregistered, NO-GO confirmatory M1.6.3. **Classification: EXPLORATORY.** All
M1.6.2-series results remain exploratory; nothing here is confirmatory evidence.

**Date:** 2026-10-28 · **Branch:** `codex/m16-heal-design-integrity`

## Purpose

Establish whether error information (verified correction explanations grounded in evidence) can be
made usable for error-informed prompting **without** speculation, before any new GPU work. Four steps:

1. **Evidence-adequacy triage** (CPU-only) of the 28 assistant-INSUFFICIENT error records.
2. **Human correction review** of the 5 existing drafts; expansion only where grounded.
3. **Lock one error-informed hypothesis** comparing gold-only vs gold+verified-explanation vs
   wrong+gold+verified-explanation exemplars, with concrete exemplar IDs.
4. **New development population** only when memory and protocol are ready.

## Step 1 — Evidence-adequacy triage: COMPLETE (2026-10-28)

Artifact: `docs/audits/M1_6_2D_EVIDENCE_TRIAGE.json` (sha256 `c966fad71fc5bbb3fc19d69ff449d28dbb5a0579af071cbe5eb6cd104f2fb5ac`).
Generators: `scripts/m162d_triage_digest.py` (digest of all 33 error records) and
`scripts/m162d_triage_finalize.py` (quote-verified triage; every bank-level quote is asserted
verbatim against the bank evidence spans at generation time).

### Classes

- **E1 — evidence for the gold genuinely absent / gold reading not derivable** (8):
  train:3, 14, 26, 38, 39, 1873, 1881, 2224. Correction drafts are **not grounded**; forcing them
  would be speculation (forbidden).
- **E2 — evidence present in the target's own store/corpus (or a retrieved passage) but missed by
  extraction/spanning or ranking** (4): train:2, 17, 20, 2449. Draftable **once the identified store
  passage is included**; all four have verbatim decisive quotes plus source URLs and passage IDs.
- **E3 — decisive (or both-sides) evidence already in the Top-8; Base reasoning/ontology error** (16):
  train:8, 12, 18, 19, 21, 24, 27, 29, 31, 35, 37, 47, 48, 49, 979, 1863. Draftable now, pending human
  review; `ONTOLOGY` subflagged records (12, 18, 21, 24, 31, 35, 49, 1863) additionally need an
  explicit human sign-off.

### Headline E2 mechanisms (server store probes, 2026-10-28)

| ID | gold | missed decisive fact | where found |
|---|---|---|---|
| train:2 | Supported | current Federation Account formula "FG (52.68%), SGs (26.72%) and LGs (20.60%)" | own store (CBN chapter; LinkedIn explainer) + raw corpus; absent from Top-8 |
| train:17 | Supported | Politifact title "Yes, Police Were **20.8** More Likely to Kill…" | present in a *retrieved* houstonlawreview passage, but the rendered **span starts mid-title and cuts the number** (span-truncation, not rank miss) |
| train:20 | Refuted | "A refinery **100% fully owned and developed by AIPCC**" (aipccenergy.com) at the same Ologbo site | own store; Top-8 only has co-developer PR framing |
| train:2449 | Conflicting | Russia-specific "telephone justice" (ICJ profile; Ledeneva) | raw corpus + retrieval store (60 hits) — **zero** telephone passages in Top-8 |

Store↔bank identity verified by bank Top-8 `passage_id` membership (8/8 in every probed store);
train:2449's store is `outputs/m161-source-batch25-v1/train-2449/retrieval-attempt-001/train-2449/passages.jsonl`,
raw corpus `…/candidates-attempt-001/train-2449/corpus.jsonl`.

**Structural finding:** all 14 wrong NEI/Conflicting-gold records were INSUFFICIENT with 0 drafts
(the 5 existing drafts are all Supported/Refuted-gold); the E3 set now supplies the NEI/Conflicting
grounding material that was missing.

## Step 2 — Human correction review: READY (awaiting user)

Packet: `docs/M1_6_2D_REVIEW_PACKET.md` (generated verbatim from the bank by
`scripts/m162d_review_packet.py`). Contents: review rules (grounded expansion only; drafting
discipline preserved; ontology sign-offs; ACCEPT / ACCEPT_WITH_EDITS / REJECT), Part A = the 5
existing drafts in full, Part B = the 20 draftable expansion candidates (16 E3 + 4 E2, each with the
exact grounded evidence to use), Part C = the 8 E1 records recorded as not draftable.

**Gate status: 0 / 33 error corrections human-verified; gate ≥30 verified UNMET.** No expansion is
performed by the assistant until reviews arrive; the ≥30 gate must never be reached by speculation.

## Step 3 — Error-informed hypothesis: DRAFTED, NOT LOCKED

Three-arm comparison (all arms EXPLORATORY, frozen prompt template, no sweeps):

- **Arm A (gold-only exemplars):** exemplars drawn from verified corrections of gold-only content.
- **Arm B (gold + verified explanation):** adds the human-verified correction explanation.
- **Arm C (wrong + gold + verified explanation):** additionally includes the Base's historical wrong
  verdict, separating the corrective-reasoning effect (B−A) from the wrong-verdict-history effect (C−B).

Selection rules (to be locked only after step 2):

- Exemplar IDs come **exclusively** from human-verified corrections (target: the 5 existing drafts +
  accepted expansions). Slots left open until the verified pool is known.
- No DEVELOPMENT50 outcome may influence exemplar selection; DEV50 stays frozen; no prompt sweeps.
- Primary metrics: Accuracy and Macro-F1 (AVeriTeC four-class ontology preserved).
- Fixed seeds/batch SHAs; sealed outputs validated by `schemas/m162c_results.schema.json` +
  `m162c.validate_result` before publication.

## Step 4 — New development population: PENDING (rules)

- Only when (a) step-2 gate satisfied and (b) step-3 design locked.
- Population: development claims **unused** in any M1.6.x pilot (not in memory75, not in the DEV50,
  not in evaluation groups); official splits preserved; Historical source knowledge excludes
  evaluation folds.
- Retrieval: **reuse the existing R4 cache when the retrieval fingerprint matches**; otherwise exactly
  **one** retrieval pass; no new source batches; no TEST_QUARANTINE access; no CPU/GPU scope creep
  (GPU runs remain user-executed).

## Gates

| Gate | Status |
|---|---|
| Step-1 triage artifact | DONE (`c966fad7…`) |
| ≥30 human-verified error corrections | **UNMET (0/33)** — blocks step-3 lock |
| Step-3 hypothesis locked | NOT LOCKED (blocked by gate above) |
| Step-4 population switch | PENDING (blocked by both above) |
| M1.6.3 confirmatory activation | NO-GO (unchanged) |

## Limitations

- Store-level quotes for E2 rows were verified in server probes and are reproduced in the artifact;
  they are not re-assertable from the local bank (probe provenance recorded per record).
- train:21/31 carry a recorded alternative E1 reading (gold NEI possibly not derivable); flagged for
  human adjudication rather than silently resolved.
- The E2 set (4) is small: it quantifies "extraction/retrieval missed it" only within this pilot
  population; no coverage rate beyond 28 records is claimable.
- Base-outcome-history effects remain unclaimable (gold-only masks), consistent with the M1.6.2c
  confound separation.

## Provenance

- Bank: `outputs/m161-review-preparation-v1/combined75.json` · Memory: `outputs/m1611-correction-audit-v3/memory75.json` (sha256s in the triage artifact).
- Server stores probed: `outputs/m11-store-r14-pilot100-20261023-162716/train-{2,3,14,17,18,20,38}` and
  `outputs/m161-source-batch25-v1/train-2449/…` (read-only greps, no writes server-side).
- `corpus_scope: per_claim` confirmed in the R1–R4 configs: each target retrieves over its own
  AVeriTeC knowledge store.
