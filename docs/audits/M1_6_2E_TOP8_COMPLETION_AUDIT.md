# M1.6.2e — Top-8 completion audit for five original corrections

Status: CPU-only evidence audit; no retrieval, inference, or review-status mutation. Generated from the immutable rendered Top-8 records in `outputs/m161-review-preparation-v1/combined75.json`.

This resolves the bracketed evidence checks in the submitted `ACCEPT_WITH_EDITS` drafts. It is **not** a human review record and does not set `VERIFIED`. The two-reviewer gate remains fail-closed.

| ID | Top-8 finding | Draft disposition after audit |
|---|---|---|
| `train:6` | State shares are 26.72% of the Federation Account (ranks 5–6), 24% of total revenue (rank 7), and 50% of VAT (rank 2). The only 35% figure is local-government VAT share (rank 2). No rendered passage gives states 35%. | Recipient check resolved; unqualified “revenue” still needs basket/period ontology sign-off. Keep `PENDING_REVIEW`; retain a qualified correction. |
| `train:11` | Federal shares are 52.68% statutory/mineral (rank 5), 52%/at least 50% Federation Account (rank 6), 44% oil after derivation (rank 4), 15% VAT (rank 5), and 14% in another allocation (rank 7). No rendered passage states 45%. | Values checked, but several non-equivalent baskets exist. Keep `PENDING_REVIEW`; Refuted needs reviewer sign-off selecting the Federation Account interpretation. |
| `train:25` | Rank 8 gives 0.69% of the **total national budget** in FY2017/18 and 0.43% in FY2018/19; rank 1 continues with about 0.44% in the then-current financial year; rank 4 independently gives 0.44% of national budget. | Evidence adequate for the qualified sub-1% correction, limited to reported fiscal years. Keep `PENDING_REVIEW` until two actual reviewers decide. |
| `train:978` | The same official Serbia Government passage at rank 2 says weekend movement is banned 15:00–05:00 (14 hours) **and** says the Government withdrew the 20-minute pet walk. | The necessary contemporaneity/linkage is directly present. Keep `PENDING_REVIEW` until two actual reviewers decide. |
| `train:1824` | Rank 4 reproduces the exact quotation under “Quotes Falsely Attributed to Winston Churchill” and says it “appears nowhere in the Churchill canon.” Rank 7 says Society experts found neither it nor a similar phrase in Churchill material. | The exact-quotation check passes, with retained caveat that the retrieved passage is a secondary reproduction of Society material. Keep `PENDING_REVIEW` until two actual reviewers decide. |

## Required wording updates

- **train:6:** State that the full Top-8 contains no 35% state share; its 35% number is local-government VAT. Do not present 26.72% as all Nigerian government revenue.
- **train:11:** State the actual VAT figures (15% federal in rank 5; 14% federal in rank 7) and the 44% oil share. Do not collapse these different baskets into a universal federal share.
- **train:25:** The 0.69% denominator check passes: it is explicitly total national budget. Support is for the reported years, not every year.
- **train:978:** The correction may state that one official notice contains both facts. Do not attribute an unobserved reasoning process to Base; report only Base's stored `Not Enough Evidence` output.
- **train:1824:** Say that retrieved material *reproduces* an International Churchill Society attribution check; do not claim the Society page itself was retrieved.

## Gate consequence

This audit removes the evidence-completeness blockers for `train:25`, `train:978`, and `train:1824`, and resolves the recipient check for `train:6`. It neither creates human-review provenance nor resolves the revenue-basket ontology decisions for `train:6` and `train:11`. No review status changes: verified-rationale count remains **0**.
