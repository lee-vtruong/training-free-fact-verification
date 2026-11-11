# Milestone M2.0i: P2 Exemplar Human Review Worksheet

This worksheet documents the 9 P2-classified exemplars from `CERTIFIED_EXEMPLAR_BANK`.
Each record provides the claim, gold verdict, baseline mistake, evidence quotes, and validation checklist.

---

## Exemplar `train:2956` — Status: `P1 (Validated via Qwen3-14B Judge M1.8g)`

- **Claim:** The Kenyan judiciary receives less than 1% of the national budget.
- **Gold Verdict:** `Supported`
- **Baseline Mistake Verdict:** `Not Enough Evidence`
- **Observed Mistake Statement:** Earlier baseline concluded 'Not Enough Evidence' by overlooking the explicit numerical figure.
- **Neutral / Corrective Explanation:** The passage explicitly confirms the exact percentage; verbatim statement substantiates the claim.
- **Evidence Passage Excerpt:**
  > [1] The Kenyan judiciary receives less than 1% of the national budget, according to official treasury reports and statements by Chief Justice David Maraga.

### Verification Checklist:
- [ ] Quote accurately extracted from Top-8 retrieval?
- [ ] Observed mistake accurately captures the failure mode without hallucination?
- [ ] Corrective resolution logically entails the gold verdict?
- [ ] Reviewer Sign-off: `PENDING`

---

## Exemplar `train:37` — Status: `P2 (Source-Traceable, Pending Human Review)`

- **Claim:** Ghana's economy previously grew at 3.4% under the former President.
- **Gold Verdict:** `Supported`
- **Baseline Mistake Verdict:** `Refuted`
- **Observed Mistake Statement:** Earlier baseline confused local growth projections with final World Bank published growth rates.
- **Neutral / Corrective Explanation:** Official World Bank macroeconomic reports substantiate the recorded 3.4% GDP growth rate.
- **Evidence Passage Excerpt:**
  > [1] World Bank economic indicators confirm Ghana's GDP growth rate was recorded at 3.4% in 2016.

### Verification Checklist:
- [ ] Quote accurately extracted from Top-8 retrieval?
- [ ] Observed mistake accurately captures the failure mode without hallucination?
- [ ] Corrective resolution logically entails the gold verdict?
- [ ] Reviewer Sign-off: `PENDING`

---

## Exemplar `train:622` — Status: `P2 (Source-Traceable, Pending Human Review)`

- **Claim:** A German Shepherd in New York was the first confirmed dog to test positive for SARS-CoV-2 in the US.
- **Gold Verdict:** `Supported`
- **Baseline Mistake Verdict:** `Conflicting Evidence/Cherrypicking`
- **Observed Mistake Statement:** Earlier baseline perceived conflicting media articles rather than prioritizing the primary USDA press release.
- **Neutral / Corrective Explanation:** The USDA official agency report unequivocally verifies the entity, location, and first positive diagnosis.
- **Evidence Passage Excerpt:**
  > [1] The USDA National Veterinary Services Laboratories confirmed that a German Shepherd in New York was the first dog in the United States to test positive for SARS-CoV-2.

### Verification Checklist:
- [ ] Quote accurately extracted from Top-8 retrieval?
- [ ] Observed mistake accurately captures the failure mode without hallucination?
- [ ] Corrective resolution logically entails the gold verdict?
- [ ] Reviewer Sign-off: `PENDING`

---

## Exemplar `train:47` — Status: `P2 (Source-Traceable, Pending Human Review)`

- **Claim:** South Africans consume more alcohol per person on average than anyone else in Africa.
- **Gold Verdict:** `Refuted`
- **Baseline Mistake Verdict:** `Supported`
- **Observed Mistake Statement:** Earlier baseline assumed high regional drinking rates established the absolute continental top ranking.
- **Neutral / Corrective Explanation:** WHO official tables rank Gabon and Nigeria higher, directly contradicting the continental superlative.
- **Evidence Passage Excerpt:**
  > [1] The WHO Global Status Report on Alcohol shows Nigeria and Gabon rank higher than South Africa in pure alcohol consumption per drinker in Africa.

### Verification Checklist:
- [ ] Quote accurately extracted from Top-8 retrieval?
- [ ] Observed mistake accurately captures the failure mode without hallucination?
- [ ] Corrective resolution logically entails the gold verdict?
- [ ] Reviewer Sign-off: `PENDING`

---

## Exemplar `train:48` — Status: `P2 (Source-Traceable, Pending Human Review)`

- **Claim:** South Africans drink more per person than anywhere else in Africa.
- **Gold Verdict:** `Refuted`
- **Baseline Mistake Verdict:** `Supported`
- **Observed Mistake Statement:** Earlier baseline accepted anecdotal claims without verifying cross-national per capita metrics.
- **Neutral / Corrective Explanation:** Official World Health Organization statistics show per capita consumption in neighboring countries exceeds South Africa.
- **Evidence Passage Excerpt:**
  > [1] Data from the World Health Organization indicates per capita consumption in Namibia and Gabon exceeds South Africa.

### Verification Checklist:
- [ ] Quote accurately extracted from Top-8 retrieval?
- [ ] Observed mistake accurately captures the failure mode without hallucination?
- [ ] Corrective resolution logically entails the gold verdict?
- [ ] Reviewer Sign-off: `PENDING`

---

## Exemplar `train:1863` — Status: `P2 (Source-Traceable, Pending Human Review)`

- **Claim:** President Buhari approves the creation of state police in Nigeria.
- **Gold Verdict:** `Refuted`
- **Baseline Mistake Verdict:** `Conflicting Evidence/Cherrypicking`
- **Observed Mistake Statement:** Earlier baseline mistook receiving a committee advisory report for formal presidential executive approval.
- **Neutral / Corrective Explanation:** The official presidency statement directly refutes the claim of executive approval.
- **Evidence Passage Excerpt:**
  > [1] Presidential spokesman Femi Adesina explicitly clarified that President Buhari did not approve state police, explaining the recommendation was merely received for further study.

### Verification Checklist:
- [ ] Quote accurately extracted from Top-8 retrieval?
- [ ] Observed mistake accurately captures the failure mode without hallucination?
- [ ] Corrective resolution logically entails the gold verdict?
- [ ] Reviewer Sign-off: `PENDING`

---

## Exemplar `train:35` — Status: `P2 (Source-Traceable, Pending Human Review)`

- **Claim:** The current Ghana administration has resulted in a decline in foreign direct investment.
- **Gold Verdict:** `Not Enough Evidence`
- **Baseline Mistake Verdict:** `Supported`
- **Observed Mistake Statement:** Earlier baseline declared support based on a single quarter's dip, overlooking incomplete annual data.
- **Neutral / Corrective Explanation:** Preliminary quarterly flows without full audited balance of payments are insufficient to prove a sustained net decline.
- **Evidence Passage Excerpt:**
  > [1] Ghana Investment Promotion Centre released preliminary figures for 2018 showing fluctuating quarterly flows, with full audited annual balance of payments not yet finalized.

### Verification Checklist:
- [ ] Quote accurately extracted from Top-8 retrieval?
- [ ] Observed mistake accurately captures the failure mode without hallucination?
- [ ] Corrective resolution logically entails the gold verdict?
- [ ] Reviewer Sign-off: `PENDING`

---

## Exemplar `train:49` — Status: `P2 (Source-Traceable, Pending Human Review)`

- **Claim:** Only 31% of South Africans drink alcohol.
- **Gold Verdict:** `Not Enough Evidence`
- **Baseline Mistake Verdict:** `Supported`
- **Observed Mistake Statement:** Earlier baseline equated partial provincial sample estimates with a verified nationwide demographic figure.
- **Neutral / Corrective Explanation:** Regional survey estimates with varied sampling methodology cannot definitively substantiate the exact national 31% statistic.
- **Evidence Passage Excerpt:**
  > [1] Surveys cite varying estimates ranging from 30% to 35% among adults in selected urban provinces, lacking a unified national census figure.

### Verification Checklist:
- [ ] Quote accurately extracted from Top-8 retrieval?
- [ ] Observed mistake accurately captures the failure mode without hallucination?
- [ ] Corrective resolution logically entails the gold verdict?
- [ ] Reviewer Sign-off: `PENDING`

---

## Exemplar `train:8` — Status: `P2 (Source-Traceable, Pending Human Review)`

- **Claim:** Joe Biden voted for the Iraq War and supported war in Serbia, Syria and Libya.
- **Gold Verdict:** `Conflicting Evidence/Cherrypicking`
- **Baseline Mistake Verdict:** `Refuted`
- **Observed Mistake Statement:** Earlier baseline rejected the entire statement as Refuted because one of the listed conflicts lacked primary documentation.
- **Neutral / Corrective Explanation:** The Iraq vote is fully verified, but attributing the entire four-conflict interventionist record mixes factual votes with unverified campaign claims.
- **Evidence Passage Excerpt:**
  > [1] Official Senate records confirm Biden voted for the 2002 Iraq War authorization.
[2] Independent analysis notes his record on Serbia and Libya is mixed, while Syria intervention claims rely on unverified campaign speeches.

### Verification Checklist:
- [ ] Quote accurately extracted from Top-8 retrieval?
- [ ] Observed mistake accurately captures the failure mode without hallucination?
- [ ] Corrective resolution logically entails the gold verdict?
- [ ] Reviewer Sign-off: `PENDING`

---

## Exemplar `train:12` — Status: `P2 (Source-Traceable, Pending Human Review)`

- **Claim:** Donald Trump inherited a stagnant economy and then rebuilt it.
- **Gold Verdict:** `Conflicting Evidence/Cherrypicking`
- **Baseline Mistake Verdict:** `Refuted`
- **Observed Mistake Statement:** Earlier baseline labeled the claim outright false, overlooking documented acceleration in specific quarterly GDP metrics.
- **Neutral / Corrective Explanation:** Economic indicators confirm growth rates accelerated, but attributing economic health solely to the transition cherry-picks continuous multi-year trends.
- **Evidence Passage Excerpt:**
  > [1] Bureau of Economic Analysis data shows GDP growth was 1.6% in 2016 and rose to 2.9% in 2018.
[2] Non-partisan economists note low unemployment and job creation trends began steadily under the previous administration.

### Verification Checklist:
- [ ] Quote accurately extracted from Top-8 retrieval?
- [ ] Observed mistake accurately captures the failure mode without hallucination?
- [ ] Corrective resolution logically entails the gold verdict?
- [ ] Reviewer Sign-off: `PENDING`

---
