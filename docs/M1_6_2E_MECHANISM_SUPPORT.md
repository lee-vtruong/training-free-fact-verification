# M1.6.2e — Per-Mechanism Verified Support Report

Generated: 2026-10-28T04:45:48Z · Status: **0 human-verified corrections** (assistant drafts are NOT verified).

Candidate pool by mechanism (from `docs/audits/M1_6_2E_CANDIDATE_DRAFTS.json`):

| mechanism | class | candidate IDs | n | verified | status |
|---|---|---|---|---|---|
| causal_attribution_missing | E3 | train:35 | 1 | 0 | PENDING_HUMAN_REVIEW |
| incomplete_integration | E3 | train:979 | 1 | 0 | PENDING_HUMAN_REVIEW |
| metric_mismatch | E3 | train:21, train:31 | 2 | 0 | PENDING_HUMAN_REVIEW |
| misreport_denial_weighting | E3 | train:1863 | 1 | 0 | PENDING_HUMAN_REVIEW |
| missed_direct_match | E3 | train:37 | 1 | 0 | PENDING_HUMAN_REVIEW |
| missed_direct_ranking | E3 | train:47, train:48 | 2 | 0 | PENDING_HUMAN_REVIEW |
| precision_demand | E3 | train:29 | 1 | 0 | PENDING_HUMAN_REVIEW |
| promise_vs_progress | E3 | train:27 | 1 | 0 | PENDING_HUMAN_REVIEW |
| retrieval_miss | E2 | train:2449 | 1 | 0 | PENDING_HUMAN_REVIEW |
| retrieval_rank_miss | E2 | train:2, train:20 | 2 | 0 | PENDING_HUMAN_REVIEW |
| scope_conflation | E3 | train:24 | 1 | 0 | PENDING_HUMAN_REVIEW |
| scope_integration | E3 | train:18 | 1 | 0 | PENDING_HUMAN_REVIEW |
| scope_qualifier | E3 | train:49 | 1 | 0 | PENDING_HUMAN_REVIEW |
| selective_reading | E3 | train:12, train:19 | 2 | 0 | PENDING_HUMAN_REVIEW |
| source_circularity | E3 | train:8 | 1 | 0 | PENDING_HUMAN_REVIEW |
| span_truncation | E2 | train:17 | 1 | 0 | PENDING_HUMAN_REVIEW |

Mechanism → verified support will be updated only from human review bookkeeping. The three-arm feasibility experiment may use an exemplar only after it is human-verified (both reviewers), and E2-quarantined mechanisms (`retrieval_rank_miss`, `span_truncation`, `retrieval_miss`) are excluded from the primary reasoning-error experiment regardless of verification status.