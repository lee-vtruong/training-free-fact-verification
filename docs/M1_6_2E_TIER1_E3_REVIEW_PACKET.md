# M1.6.2e — Tier-1 E3 review packet

Purpose: review four SOURCE-only candidate corrections that can satisfy the **coverage** portion of the locked Tier-1 gate if both reviewers accept them. This is an exploratory review packet, not a GPU experiment. Do not use DEVELOPMENT or TEST material.

## Reviewer instructions

For every record, independently inspect the quoted rendered Top-8 passages and choose exactly one:

- `ACCEPT`: evidence supports the gold relation and proposed failure mechanism.
- `ACCEPT_WITH_EDITS`: provide a complete replacement correction in `edited_text`.
- `REJECT`: the displayed evidence cannot ground the gold relation or mechanism.

Also provide `evidence_sufficient`, `gold_relation_supported`, `mechanism_confirmed`, and an explicit `ontology_signoff` for `train:21` and `train:31`. Do not infer hidden model reasoning: describe only its stored Base verdict and the evidence relationship. Two distinct human reviewers must complete the output template independently.

## Coverage target

| mechanism | candidates | why selected |
|---|---|---|
| `metric_mismatch` | `train:21`, `train:31` | two exact Top-8 candidates; both need ontology sign-off for NEI rather than Conflicting. |
| `missed_direct_ranking` | `train:47`, `train:48` | two exact Top-8 candidates; direct rankings refute a per-capita superlative. |

No candidate becomes verified merely by appearing in this packet. E2 records remain excluded, and no threshold is lowered if a reviewer rejects a record.

---

## train:21 — `metric_mismatch`

- Claim: “There is a ‘global average’ for the number of judges and magistrates to number of people in Kenya.”
- Gold: `Not Enough Evidence`; Base: `Supported`.
- Top-8 rank 2 (UNODC): “On average, there are 15,000 inhabitants for every judge or magistrate across the globe.”
- Top-8 rank 1 (University of Birmingham thesis): “about 75 judges 1,000,000 persons.”

Candidate correction: The worldwide judges-and-magistrates ratio and Kenya judges-only ratio use different officer populations and denominators. No displayed passage gives a comparable Kenya judges-and-magistrates figure or a same-metric global comparator; therefore the asserted Kenya relation is not decidable from these spans, supporting `Not Enough Evidence` rather than Base `Supported`.

Required ontology sign-off: Does the worldwide UNODC value alone answer this claim, or does the claim require a comparable Kenya relation? Explain the selected reading.

---

## train:31 — `metric_mismatch`

- Claim: “There is a ‘global average’ for the number of judges and magistrates to number of people in Kenya.”
- Gold: `Not Enough Evidence`; Base: `Refuted`.
- Top-8 rank 5 (Judicial Service Commission presentation): “Kenya with a population of 48 million has a judge to population ratio of 1 judicial”.
- Top-8 rank 7 (ReliefWeb): “well below the national global average of 17”.

Candidate correction: The displayed Kenya ratio is judges/judicial officers, whereas the ReliefWeb “global average of 17” concerns public defenders; the available worldwide judges-and-magistrates value has a separate denominator. These are not comparable measures, so the displayed evidence cannot establish the gold relation by either support or refutation. The appropriate verdict is `Not Enough Evidence`, not Base `Refuted`.

Required ontology sign-off: Confirm that metric mismatch indicates insufficiency, not conflicting evidence, and state why.

---

## train:47 — `missed_direct_ranking`

- Claim: “South African consume more alcohol per person on average than anywhere else in Africa.”
- Gold: `Refuted`; Base: `Not Enough Evidence`.
- Top-8 rank 1 (BusinessTech): “third biggest drinking nation in Africa, and the 19th biggest drinking nation in the world”.

Candidate correction: A third-place African per-capita ranking entails at least two African countries have higher consumption. The displayed ranking therefore directly refutes the superlative “than anywhere else in Africa”; Base `Not Enough Evidence` missed this direct comparison. The statement remains limited to the rendered WHO-2015 per-capita framing.

Review focus: verify that “third biggest drinking nation in Africa” is sufficient evidence for the intended per-person superlative and preserve its data-year/basis limitation.

---

## train:48 — `missed_direct_ranking`

- Claim: “South Africans drink more per person than anywhere else in Africa.”
- Gold: `Refuted`; Base: `Not Enough Evidence`.
- Top-8 rank 4 (BusinessTech): “below Namibia (31.3 litres), Eswatini (32.7 litres), Cook Islands (32.9 litres) and Tunisia (33.4 litres)”.

Candidate correction: Namibia, Eswatini, and Tunisia are African countries listed above South Africa in the displayed per-capita ranking. That directly refutes the Africa-wide superlative, so `Refuted` is grounded rather than Base `Not Enough Evidence`. The comparison retains the source's WHO/2005-era and per-capita framing.

Review focus: verify country membership, ranking direction, and that the claim's per-person scope matches the cited litres comparison.

## Required output

Each reviewer returns one filled copy of [the JSON output template](audits/M1_6_2E_TIER1_E3_REVIEW_OUTPUT_TEMPLATE.json). Do not edit the candidate bank or use a shared response. A record qualifies only when both submissions accept it and both mark evidence/gold/mechanism as grounded.
