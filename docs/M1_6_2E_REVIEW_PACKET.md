# M1.6.2e — Correction Review Packet (two reviewers)

Generated: 2026-10-28T04:45:48Z · EXPLORATORY · 5 original drafts (untouched) + 16 E3 candidate drafts + 4 E2 candidates (quarantined) + 8 E1 no-draft records.

**Rules.** (1) Decisions per record: `ACCEPT` / `ACCEPT_WITH_EDITS` / `REJECT`; both reviewers fill `decision`, optional `edited_text`, `rationale`, `ontology_signoff` (mandatory for records marked SIGN-OFF), `reviewed_at`. (2) Grounded expansion only: support must quote the displayed spans (or, for E2, the identified store passages); never fabricate to reach a gate. (3) The assistant never fills reviewer fields and never marks VERIFIED. (4) E2 records are quarantined from the primary reasoning-error experiment (evidence-availability change declared). (5) Current verified count: **0**.

---

### train:6 — ORIGINAL_DRAFT (gold=Refuted, base=Not Enough Evidence)

- **Claim:** 35% of revenue goes to states. (In Nigeria)
- **Mechanism:** —

**Candidate draft:**

> The displayed federal-account formula allocates 26.72% to states, not 35%. A different excerpt places 35% in the VAT allocation to local government, a different recipient and revenue basket. The Base notices countervailing figures but treats lack of confirmation as unresolved; the stated formula contradicts 35% for states under this reading.

**Decisive quotes:** none recorded

---

### train:11 — ORIGINAL_DRAFT (gold=Refuted, base=Not Enough Evidence)

- **Claim:** 45% of Nigeria's revenue goes to the federal government.
- **Mechanism:** —

**Candidate draft:**

> The excerpt explicitly distinguishes an extant federal share of 52.68% of statutory/mineral revenue from a 15% VAT share. These are different baskets, not evidence that 45% is supported. Under the federation-account reading the stated 52.68% contradicts 45%, whereas the Base treats mismatched baskets as a reason to abstain.

**Decisive quotes:** none recorded

---

### train:25 — ORIGINAL_DRAFT (gold=Supported, base=Refuted)

- **Claim:** The Kenyan judiciary does not even get 1% of the national budget.
- **Mechanism:** —

**Candidate draft:**

> The Base correctly observes sub-1% allocations but interprets not even 1% as a minimum of 1%. Not even 1% means below 1%. The displayed 0.43% and 0.44% figures support that comparison rather than refute it.

**Decisive quotes:** none recorded

---

### train:978 — ORIGINAL_DRAFT (gold=Supported, base=Not Enough Evidence)

- **Claim:** In Serbia, the stricter Covid19 curfew measures prevented dog walking for up to 14 hours at a time.
- **Mechanism:** —

**Candidate draft:**

> The official Serbian notice gives weekend movement restrictions from 15.00 to 5.00 and withdraws the 20-minute pet-walk exception. Across midnight that interval is 14 hours. The Base considers only the weekday 17.00-to 5.00 interval, so its claim that no 14-hour interval is shown overlooks the weekend measure.

**Decisive quotes:** none recorded

---

### train:1824 — ORIGINAL_DRAFT (gold=Refuted, base=Conflicting Evidence/Cherrypicking)

- **Claim:** Former British Prime Minister Winston Churchill said, 'Courage is what it takes to stand up and speak; courage is also what it takes to sit down and listen'.
- **Mechanism:** —

**Candidate draft:**

> Generic quote pages repeat the Churchill attribution, but the displayed correction explicitly cites the Churchill Society's falsely-attributed list and says the quotation appears nowhere in the Churchill canon. Repetition is not independent authorship evidence. This specific attribution correction supports Refuted instead of treating all quote pages as equally reliable conflicting accounts.

**Decisive quotes:** none recorded

---

### train:8 — E3_CANDIDATE_DRAFT (gold=Conflicting Evidence/Cherrypicking, base=Supported)

- **Claim:** Joe Biden voted for the Irag War and supported war in Serbia, Syria and Libya.
- **Mechanism:** source_circularity
- **Ontology note (re-review):** Conflicting retained: the dispute is over the same proposition set (Iraq vote; the three-conflict enumeration) across comparable-type sources; the enumeration itself originates in the claim-source speech, which is a sourcing defect, not source disagreement.
- **Limitations:** Syria element unevidenced in displayed spans; Serbia/Libya supported only at general-intervention-record level.

**Candidate draft:**

> The claim mixes a documented vote with elements attested mainly in its own source. Independent passages confirm the Iraq vote and its later repudiation - CFR: "supported the 2003 invasion of Iraq, but he now calls it a mistake" - and describe an interventionist record (Jacobin: "a reliable representative of the national security establishment"; Arab Center: a "mixed record" on US interventions). But the "Serbia, Syria, and Libya" enumeration comes from the CNN passage quoting the attack speech itself, and no displayed passage evidences a Syria war position. The Iraq element is supported while the three-conflict enumeration is partially unevidenced and circularly sourced; that mixed state is Conflicting Evidence/Cherrypicking, not the Base's flat Supported.

**Decisive quotes (top8_span):**
- Top-8 [3] www.cfr.org — "supported the 2003 invasion of Iraq, but he now calls it a “mistake”"
- Top-8 [1] www.cnn.com — "Joe Biden voted for the Iraq War, which President Trump has long called"

---

### train:12 — E3_CANDIDATE_DRAFT (gold=Conflicting Evidence/Cherrypicking, base=Refuted)

- **Claim:** Donald Trump inherited a stagnant economy and then rebuilt it.
- **Mechanism:** selective_reading
- **Ontology note (re-review):** Conflicting retained: the disagreement is between comparable-type sources over the same propositions (inherited condition; rebuild effect). Both sides carry partisan provenance (WH archives vs nwlc/medium), noted; partisan symmetry is not itself grounds for a different verdict.
- **Limitations:** Both sides include partisan framing (administration archives vs advocacy outlets); tracker-level economic data not displayed.

**Candidate draft:**

> Both sides of the dispute are displayed. Recovered-economy side: "Donald Trump inherited an economy that had largely recovered, according to many indicators, by the time he took office" (nwlc) and "Trump inherited the longest economic expansion in" modern history (medium). Stagnation-inheritance side: the White House archives passage frames the inheritance as stagnation - "low growth, or "secular stagnation," would plague the U.S. economy indefinitely" - and the Detroit News passage credits the "pro-growth policy agenda" with policies that "have inspired enormous confidence in the economy". The Base read only the recovered side and returned Refuted. With the inherited-stagnation framing displayed in the administration's own material and the rebuild element contested, the verdict is Conflicting Evidence/Cherrypicking.

**Decisive quotes (top8_span):**
- Top-8 [1] nwlc.org — "Donald Trump inherited an economy that had largely recovered"
- Top-8 [4] trumpwhitehouse.archives.gov — "low growth, or “secular stagnation,” would plague the U.S. economy indefinitely"

---

### train:18 — E3_CANDIDATE_DRAFT (gold=Conflicting Evidence/Cherrypicking, base=Not Enough Evidence)

- **Claim:** A greater proportion of 16 year olds vote in elections in Scotland, Wales and Austria than 18 and 20 year olds.
- **Mechanism:** scope_integration
- **Ontology note (re-review):** Conflicting retained: the tension is about the claim's three-jurisdiction generalization (Austria/Scotland support vs a Wales-specific caution), i.e., claim-scope conflict, not mere source disagreement. Sign-off required.
- **Limitations:** Austria support is bibliography-title-level; no matched 16-vs-18/20 Wales comparison is displayed.

**Candidate draft:**

> Support exists for two of the three jurisdictions and a caution for the third, all inside the displayed spans. The bibliography cites Zeglovits & Aichholzer (2014), "Are people more inclined to vote at 16 than at 18? Evidence for the first-time voting boost among 16- to 25-year-olds in Austria", and the Wales passage itself continues: "Evidence from Scotland shows that in the 2014 independence referendum where 16-17 year olds were enfranchised to vote, 75 per[cent]..." The same passage warns that "Austria has traditionally high turnout rates for elections (Zeglovits & Aichholzer, 2014), which cannot be said for elections in Wales", where turnout is "typically below 50 per cent". The Base returned Not Enough Evidence without integrating the displayed title and caveat. The first-time-voting boost is evidenced for Austria and Scotland while the Wales leg rests on low-turnout context rather than a matched comparison - the three-jurisdiction generalization is supported in part and undermined in part, i.e., Conflicting Evidence/Cherrypicking.

**Decisive quotes (top8_span):**
- Top-8 [3] repositorio.iscte-iul.pt — "Evidence for the first-time voting boost among 16- to 25-year-olds in Austria"
- Top-8 [6] www.gov.wales — "turnout across the country as a whole for National Assembly and local council elections are typically below 50 per cent"

---

### train:19 — E3_CANDIDATE_DRAFT (gold=Conflicting Evidence/Cherrypicking, base=Refuted)

- **Claim:** Donald Trump has kept his promises to voters
- **Mechanism:** selective_reading
- **Ontology note (re-review):** Conflicting retained: tracker snapshots and the research framing are same-type evidence about the same proposition and point in different directions; the mix is genuine, not manufactured.
- **Limitations:** Tracker snapshots differ in date and methodology; "try to keep" (research) vs "kept" (trackers) is not a single metric.

**Candidate draft:**

> Both directions are displayed. Broken/stalled side: "Out of 102 promises on its Trump-o-Meter scorecard, 33 are currently stalled and seven broken, while only nine have been kept and seven ended with a compromise" (Time/PolitiFact), and "Only six promises are listed as "kept," while 45 have seen no action" (Denver Post). Kept-side framing: Time also reports "decades of academic research has actually shown the opposite: that presidents try to keep an average of about two-thirds of their campaign pledges". The Base read only the stalled/broken trackers and returned Refuted. With kept counts and the research framing displayed alongside the stalled counts, the mixed record supports Conflicting Evidence/Cherrypicking rather than outright refutation.

**Decisive quotes (top8_span):**
- Top-8 [1] time.com — "33 are currently stalled and seven broken"
- Top-8 [4] time.com — "decades of academic research has actually shown the opposite"

---

### train:21 — E3_CANDIDATE_DRAFT (gold=Not Enough Evidence, base=Supported)

- **Claim:** There is a ‘global average’ for the number of judges and magistrates to number of people in Kenya.
- **Mechanism:** metric_mismatch
- **Ontology note (re-review):** NEI protected from auto-Conflicting: the displayed ratios (judges-only per million; judicial officers per 100,000; worldwide judges-and-magistrates per 15,000 inhabitants) are different measurements, not opposing verdicts on one proposition. Insufficiency is the ontology-correct reading. Sign-off required.
- **Limitations:** Metric-comparability reading recorded for sign-off; an alternative reading (the UNODC figure itself answers the claim) is disclosed.

**Candidate draft:**

> The displayed figures cannot be combined into the asserted comparison. The only global figure is UNODC's "On average, there are 15,000 inhabitants for every judge or magistrate across the globe" - a worldwide judges-and-magistrates ratio. The Kenya-side passages use different metrics: "about 75 judges 1,000,000 persons" (judges only, per million) and a judicial-officer ratio per 100,000. No displayed passage states a global average on the same metric as the Kenya figures, nor a Kenya figure on the global metric, so the claim's relation cannot be checked. The Base accepted the worldwide figure as if it settled the Kenya relation; the verdict is Not Enough Evidence.

**Decisive quotes (top8_span):**
- Top-8 [2] www.unodc.org — "On average, there are 15,000 inhabitants for every judge or magistrate across the globe"
- Top-8 [1] etheses.bham.ac.uk — "about 75 judges 1,000,000 persons"

---

### train:24 — E3_CANDIDATE_DRAFT (gold=Conflicting Evidence/Cherrypicking, base=Supported)

- **Claim:** Anxiety levels among young teenagers dropped during the coronavirus pandemic, a study has suggested
- **Mechanism:** scope_conflation
- **Ontology note (re-review):** Conflicting retained with scope note: the study finding (13-14-year-olds, South West of England) and the general-population increases (ONS adults; 21-40 age group) address different scopes; the gold reads the displayed tension as conflict. Sign-off required.
- **Limitations:** Sign-off required on the scope reading (study report vs general trend).

**Candidate draft:**

> The displayed evidence both supports the study finding and contradicts the broader trend. Study side: "Younger teenagers in the South West of England felt less anxious and more connected to school when they were away from it during the COVID-19 global pandemic... a first-of-its-kind study has found", with figures that "dropped by nearly 10 per cent among girls to less than half (45 per cent) and to less than one in five of boys (18 per cent)". Counter-side: "average anxiety scores were 5.2 out of 10, a marked increase from 3.0 in the last quarter of 2019" (ONS), and higher anxiety reported for the 21-40 age group. The Base read the study side alone as confirmation. Because the study-specific drop and the general-population increases are both displayed, the verdict is Conflicting Evidence/Cherrypicking; the claim's "a study has suggested" attribution is preserved in the reading.

**Decisive quotes (top8_span):**
- Top-8 [1] www.newswise.com — "felt less anxious and more connected to school when they were away from it"
- Top-8 [8] www.ons.gov.uk — "average anxiety scores were 5.2 out of 10, a marked increase from 3.0"

---

### train:27 — E3_CANDIDATE_DRAFT (gold=Not Enough Evidence, base=Supported)

- **Claim:** The counties are striving to reach the 10% [allocation to agriculture] recommended by the Malabo declaration.
- **Mechanism:** promise_vs_progress
- **Limitations:** No material limitation beyond the promise/progress gap itself.

**Candidate draft:**

> The displayed evidence documents commitments, not striving. Counties "made the following promises to their people: (a) to allocate at least 10% of national budgets to agriculture; and (b) to achieve at least 6% annual agricultural growths" - the 10% target itself being a commitment to double the then average. The Malabo 10% target and expenditure-monitoring frameworks are displayed, but no displayed passage reports county-level allocations or progress toward the 10% mark. The Base equated the promise with progress and returned Supported. With only commitments displayed, there is not enough evidence that the counties are striving to reach the target.

**Decisive quotes (top8_span):**
- Top-8 [2] www.devolution.go.ke — "made the following promises to their people: (a) to allocate at least 10% of national budgets to agriculture"

---

### train:29 — E3_CANDIDATE_DRAFT (gold=Supported, base=Not Enough Evidence)

- **Claim:** The judiciary of Kenya have an average of 400,000 cases being filed in our courts every year.
- **Mechanism:** precision_demand
- **Limitations:** Single fiscal year (FY 2013/14) displayed; no multi-year average shown.

**Candidate draft:**

> The claimed magnitude is displayed directly: "there were 397,243 cases filed in all of the courts of Kenya during FY 2013/14". About 400,000 filings per year is what the claim asserts, and 397,243 is that figure within ordinary rounding of "an average of 400,000". The Base acknowledged the number but demanded an exact 400,000 and a multi-year average; the gold reading accepts the displayed magnitude. The single displayed fiscal year is the only limitation.

**Decisive quotes (top8_span):**
- Top-8 [1] pdf.usaid.gov — "there were 397,243 cases filed in all of the courts of Kenya during FY 2013/14"

---

### train:31 — E3_CANDIDATE_DRAFT (gold=Not Enough Evidence, base=Refuted)

- **Claim:** There is a ‘global average’ for the number of judges and magistrates to number of people in Kenya.
- **Mechanism:** metric_mismatch
- **Ontology note (re-review):** NEI protected from auto-Conflicting (mirror of train:21): divergent officer ratios are different measurements, not opposing verdicts. Sign-off required.
- **Limitations:** Sign-off required; metric-comparability reading disclosed.

**Candidate draft:**

> The Base refuted the claim by contrasting Kenya's ratio - "Kenya with a population of 48 million has a judge to population ratio of 1 judicial officer" - against other countries' ratios. But no displayed passage provides a global average for judges and magistrates on a comparable metric: the worldwide UNODC figure ("15,000 inhabitants for every judge or magistrate across the globe") uses a different denominator, and the "global average of 17" in the displayed reliefweb passage concerns public defenders per 100,000, a different officer class. With no comparable benchmark displayed, the claim can be neither confirmed nor refuted - Not Enough Evidence.

**Decisive quotes (top8_span):**
- Top-8 [5] judiciary.go.ug — "Kenya with a population of 48 million has a judge to population ratio of 1 judicial"
- Top-8 [7] reliefweb.int — "well below the national global average of 17"

---

### train:35 — E3_CANDIDATE_DRAFT (gold=Not Enough Evidence, base=Conflicting Evidence/Cherrypicking)

- **Claim:** The current Ghana administration has resulted in a decline in unemployment rates from 11.9% in 2015 to 7.3% in 2019.
- **Mechanism:** causal_attribution_missing
- **Ontology note (re-review):** NEI protected from auto-Conflicting: the GLSS series (11.9%->7.3%) and the ILO-modelled series (3.46% in 2019) are different measurements with different methodologies, not opposing verdicts; insufficiency of the causal claim is the ontology-correct reading. Sign-off required.
- **Limitations:** Sign-off required; both series are displayed only at single-year resolution.

**Candidate draft:**

> The decline numbers are displayed - "the rate of unemployment, which stood at 11.9% in 2015, dropped to 7.3% in 2019" (Ghana Living Standards Survey, quoted in the presidential-speech passage) - but the claim attributes the decline to the administration's policies. Two displayed facts block that attribution: the ILO-modelled series reports "Ghana unemployment rate for 2019 was 3.46%", materially below the GLSS 7.3%, so the level and size of any decline depend on which measured series is used; and no displayed passage attributes the movement to administration policy rather than survey methodology. The Base returned Conflicting by weighing the series against each other; divergent series are different measurements, not opposing verdicts. The causal claim is unevidenced: Not Enough Evidence.

**Decisive quotes (top8_span):**
- Top-8 [8] www.gbcghanaonline.com — "the rate of unemployment, which stood at 11.9% in 2015, dropped to 7.3% in 2019"
- Top-8 [4] www.macrotrends.net — "Ghana unemployment rate for 2019 was 3.46%"

---

### train:37 — E3_CANDIDATE_DRAFT (gold=Supported, base=Conflicting Evidence/Cherrypicking)

- **Claim:** Ghana's economy previously grew at 3.4% under the form President Mahama's government and on average, for three successive years, at 7% per year with the current administration.
- **Mechanism:** missed_direct_match
- **Limitations:** The figures appear in a presidential-speech report (ghanaweb column); single-source, government-attributed.

**Candidate draft:**

> The claim's figures are displayed verbatim: "annual average of 7%, up from 3.4% in 2016, the lowest in nearly three decades" - matching the claimed 3.4% under the prior government and an average of 7% per year under the current one. The Base returned Conflicting Evidence/Cherrypicking by weighing this passage against unrelated indicators instead of reading the direct match. The gold reads the displayed statement as sufficient support for the claimed growth figures.

**Decisive quotes (top8_span):**
- Top-8 [5] www.ghanaweb.com — "annual average of 7%, up from 3.4% in 2016, the lowest in nearly three decades"

---

### train:47 — E3_CANDIDATE_DRAFT (gold=Refuted, base=Not Enough Evidence)

- **Claim:** South African consume more alcohol per person on average than anywhere else in Africa.
- **Mechanism:** missed_direct_ranking
- **Limitations:** WHO 2015-data framing; per-capita basis.

**Candidate draft:**

> The displayed ranking refutes the superlative: "This pushes South Africa up to the third biggest drinking nation in Africa, and the 19th biggest drinking nation in the world, tied with Poland." Third in Africa entails that at least two African countries consume more per person, so South Africans do not consume more alcohol per person "than anywhere else in Africa." The Base returned Not Enough Evidence despite the direct statement.

**Decisive quotes (top8_span):**
- Top-8 [1] businesstech.co.za — "third biggest drinking nation in Africa, and the 19th biggest drinking nation in the world"

---

### train:48 — E3_CANDIDATE_DRAFT (gold=Refuted, base=Not Enough Evidence)

- **Claim:** South Africans drink more per person than anywhere else in Africa
- **Mechanism:** missed_direct_ranking
- **Limitations:** Passage mixes world-ranking (28.9 litres, 2005-era WHO figures) with the per-capita claim; ranking direction is unambiguous.

**Candidate draft:**

> The displayed passage names the countries above South Africa: "the fifth highest consumption rate in the world, below Namibia (31.3 litres), Eswatini (32.7 litres), Cook Islands (32.9 litres) and Tunisia (33.4 litres)". Namibia, Eswatini and Tunisia are African and rank above South Africa, so South Africans do not drink more per person than anywhere else in Africa. The Base returned Not Enough Evidence despite the explicit comparison.

**Decisive quotes (top8_span):**
- Top-8 [4] businesstech.co.za — "below Namibia (31.3 litres), Eswatini (32.7 litres), Cook Islands (32.9 litres) and Tunisia (33.4 litres)"

---

### train:49 — E3_CANDIDATE_DRAFT (gold=Not Enough Evidence, base=Supported)

- **Claim:** Only 31% of South African drink alcohol
- **Mechanism:** scope_qualifier
- **Ontology note (re-review):** NEI protected from auto-Conflicting: the displayed estimates (31% of 15+; 31% of adults past-year; 35%; 33.1%) differ by scope and denominator, so they are not opposing verdicts on one proposition; insufficiency of the unqualified claim is the ontology-correct reading. Sign-off required.
- **Limitations:** Sign-off required on the scope reading.

**Candidate draft:**

> The displayed estimates diverge by scope and denominator: "an estimated 31% of the population aged 15 and older consume alcohol" (WHO 2016 data), "only a third (31%) of adults report consuming alcohol in the past year", "the 35% of the population that do consume alcohol", and a 33.1% figure in the BMC passage. The unqualified "Only 31% of South African drink alcohol" specifies neither population, year nor survey; with scoped estimates spanning 31-35%, the unqualified figure is not settled. The Base accepted the first 31% without its scope qualifier. Not Enough Evidence.

**Decisive quotes (top8_span):**
- Top-8 [1] businesstech.co.za — "an estimated 31% of the population aged 15 and older consume alcohol"
- Top-8 [2] www.thedtic.gov.za — "of the 35% of the population that do consume alcohol"

---

### train:979 — E3_CANDIDATE_DRAFT (gold=Conflicting Evidence/Cherrypicking, base=Not Enough Evidence)

- **Claim:** COVID-19 virus could be spread from shoes.
- **Mechanism:** incomplete_integration
- **Ontology note (re-review):** Conflicting retained: this is a genuinely contested possibility claim - surface persistence is displayed alongside explicit low-likelihood assessments. The tension is evidential, not a mere difference between source types.
- **Limitations:** Risk assessments are expert commentary, not measurement; possibility vs probability framing disclosed.

**Candidate draft:**

> The displayed evidence genuinely splits on the transmission question. Possibility side: "shoes can be a potential source of contamination, according to family practitioner Georgine Nanos"; persistence is displayed as "coronavirus can remain active on some surfaces, like plastic, for up to two to three days", with shoe soles discussed as virus carriers. Low-risk side: "the likelihood of spreading or contracting the virus from shoes is low", and the Stanford NEJM-based comment: "it sees unlikely that enough virus would land on the floor, then get transmitted to shoes". The Base collapsed this displayed tension into Not Enough Evidence. Both a transmission route treated as possible and an explicit low-likelihood assessment are displayed: Conflicting Evidence/Cherrypicking.

**Decisive quotes (top8_span):**
- Top-8 [2] www.huffpost.com — "shoes can be a potential source of contamination"
- Top-8 [5] www.womenshealthmag.com — "the likelihood of spreading or contracting the virus from shoes is low"

---

### train:1863 — E3_CANDIDATE_DRAFT (gold=Refuted, base=Conflicting Evidence/Cherrypicking)

- **Claim:** President Buhari approves the creation of state police.
- **Mechanism:** misreport_denial_weighting
- **Ontology note (re-review):** Refuted retained: the presidency denial directly addresses the reported approval - this is a refutation of the reported act, not a mere source disagreement. Sign-off required on denial-weighting.
- **Limitations:** Denial-weighting reading recorded for sign-off; no independent confirmation of the denial displayed.

**Candidate draft:**

> The refutation chain is displayed: the report - "President Muhammadu Buhari Monday in Abuja approved the creation of state and local government police" - and the same-day dismissal - "The presidency has dismissed media reports claiming President Muhammadu Buhari approved the establishment of State and Local Government Police." The approval the claim asserts was reported and then dismissed by the presidency. The Base weighted the pair as mere tension and returned Conflicting Evidence/Cherrypicking; the gold reads the dismissal of the underlying report as refuting the claimed approval.

**Decisive quotes (top8_span):**
- Top-8 [1] www.ripplesnigeria.com — "The presidency has dismissed media reports claiming President Muhammadu Buhari approved the establishment of State and Local Government Police"
- Top-8 [3] www.thisdaylive.com — "President Muhammadu Buhari Monday in Abuja approved the creation of state and local government police"

---

### train:2 — E2_CANDIDATE_DRAFT_QUARANTINED (gold=Supported, base=Not Enough Evidence) · QUARANTINED (E2)

- **Claim:** In Nigeria … in terms of revenue share, 20% goes to the local government.
- **Mechanism:** retrieval_rank_miss
- **Limitations:** Formula vintage: the LinkedIn "current" is undated and the CBN chapter is 2013-era; the claim itself is undated.

**Candidate draft:**

> The target store contains the current statutory split: "The Federation Account revenue sharing formula is as follows: FG (52.68%), SGs (26.72%) and LGs (20.60%)" (CBN chapter five), and "In the current revenue allocation formula, the FG gets 52.68%, State Governments get 26.72% and the Local Governments get 20.60%" (LinkedIn explainer). A 20% share to local government is grounded by the current formula (20.60%); the Top-8 missed both passages, leaving only the historical 1976-1992 increase. Supported per the gold, conditional on including the store passages.

**Decisive quotes (store_probe):**
- store probe: https://www.cbn.gov.ng/out/2013/ccd/chapter%20five.pdf — "The Federation Account revenue sharing formula is as follows: FG (52.68%), SGs (26.72%) and LGs (20.60%)"
- store probe: https://www.linkedin.com/pulse/how-revenue-being-shared-from-federation-account- — "In the current revenue allocation formula, the FG gets 52.68%, State Governments get 26.72% and the Local Governments get 20.60%"

---

### train:17 — E2_CANDIDATE_DRAFT_QUARANTINED (gold=Supported, base=Conflicting Evidence/Cherrypicking) · QUARANTINED (E2)

- **Claim:** In 2019 the US Police were 20.8 times more likely to kill than be killed by a criminal
- **Mechanism:** span_truncation
- **Limitations:** Attestation is a fact-check title cited inside a law review; not primary 2019 data.

**Candidate draft:**

> The retrieved law-review passage contains the complete citation: "Madeline Heim, Yes, Police Were 20.8 More Likely to Kill than Be Killed by a Criminal in 2019, POLITIFACT (Aug. 27, 2020)". The ratio the claim asserts is attested bibliographically; the rendered span began mid-title and cut the number, which is why the Base saw no usable figure. Supported per the gold, conditional on reading the full passage rather than the truncated span.

**Decisive quotes (store_probe):**
- store probe: https://houstonlawreview.org/article/22273.pdf — "Madeline Heim, Yes, Police Were 20.8 More Likely to Kill than Be Killed by a Criminal in 2019, POLITIFACT (Aug. 27, 2020)"

---

### train:20 — E2_CANDIDATE_DRAFT_QUARANTINED (gold=Refuted, base=Supported) · QUARANTINED (E2)

- **Claim:** The Edo Modular Refinery is Governor Obaseki's project.
- **Mechanism:** retrieval_rank_miss
- **Limitations:** Company self-report; the state-sponsored framing coexists and the ownership/sponsorship line is the decisive distinction.

**Candidate draft:**

> The operator's own project page states: "A refinery 100% fully owned and developed by AIPCC. It is located in Ologbo, Ikpoba Okha LGA of Edo State." The Ologbo modular refinery is AIPCC's project; the displayed Top-8 framing ("partly sponsored by the Governor Godwin Obaseki-led administration... being developed by AIPCC Energy Limited") describes sponsorship and facilitation, not ownership. "Governor Obaseki's project" is refuted by the ownership statement. Refuted per the gold, conditional on including the store passage.

**Decisive quotes (store_probe):**
- store probe: https://www.aipccenergy.com/operations-projects — "A refinery 100% fully owned and developed by AIPCC. It is located in Ologbo, Ikpoba Okha LGA of Edo State."

---

### train:2449 — E2_CANDIDATE_DRAFT_QUARANTINED (gold=Conflicting Evidence/Cherrypicking, base=Refuted) · QUARANTINED (E2)

- **Claim:** Russia has a lower prison population than America because of the presence of telephone justice in Russia
- **Mechanism:** retrieval_miss
- **Limitations:** Causal link unevidenced; telephone-justice evidence is qualitative; rate comparisons span different years/definitions.

**Candidate draft:**

> The causal premise's existence is recoverable: the ICJ Russia profile reports "the phenomenon of "telephone justice" remained widespread: reportedly, judges are often - as a matter of routine - directly instructed by the court president how to resolve a specific case", and the retrieved "Telephone Law and the Rule of Law: The Russian Case" material cites Ledeneva's "Telephone Justice in Russia" (Post-Soviet Affairs, 2008). But the displayed prison-rate passages show Russia's rates high and period-dependent ("The incarceration rate (per 100,000 of the population) was 528 in 2004"), with no passage linking telephone justice to prison population. The premise is real while the because-clause is unevidenced: Conflicting Evidence/Cherrypicking per the gold, conditional on including the store/corpus passages.

**Decisive quotes (store_probe):**
- store probe: https://media.law.wisc.edu/m/nmytc/telephone_law_and_rol.pdf — "'Telephone Law' and the 'Rule of Law': The Russian Case (Alena V. Ledeneva, 'Telephone Justice in Russia', 24 Post-Soviet Affairs (2008))"
