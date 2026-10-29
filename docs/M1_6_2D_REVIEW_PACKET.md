# M1.6.2d — Step 2 Human Correction Review Packet

Generated: 2026-10-28 04:32Z | Scope: EXPLORATORY (internal M1.6.3a Error Information Feasibility; the confirmatory M1.6.3 remains NO-GO).

## Review rules (locked)

1. **Grounded expansion only.** A correction may be expanded/accepted only where its support quotes verbatim from displayed evidence (Top-8) or, for E2 candidates, from the identified per-claim store passage. Never fabricate or extrapolate to reach the ≥30-verified gate; an unmet gate is reported as unmet.
2. **Drafting discipline preserved.** Do not consult raw-train justifications or any gold QA/justification material for SOURCE records; drafts stand on evidence only (this matches the M1.6.1 discipline).
3. **Ontology.** Verdicts use the AVeriTeC four-class ontology; records flagged `ONTOLOGY` require an explicit human sign-off line in the review.
4. **Verdicts per draft:** `ACCEPT` / `ACCEPT_WITH_EDITS` (with edited text) / `REJECT` (with reason). Per-review bookkeeping increments the verified-error count; current status **0 verified / 33 error corrections in scope; gate ≥30 unmet**.
5. No DEVELOPMENT50 outcomes may be used to tune exemplar selection or prompts; no further DEV50 prompt sweeps; results remain EXPLORATORY.

## Part A — Existing correction drafts (5, verbatim from bank)

### train:6 (gold=Refuted, base=Not Enough Evidence)

- **Claim:** 35% of revenue goes to states. (In Nigeria)
- **Draft status (bank metadata):** PENDING_REVIEW

**Draft text (verbatim, from memory record):**

> The displayed federal-account formula allocates 26.72% to states, not 35%. A different excerpt places 35% in the VAT allocation to local government, a different recipient and revenue basket. The Base notices countervailing figures but treats lack of confirmation as unresolved; the stated formula contradicts 35% for states under this reading.

**Evidence inventory (Top-8 rendered spans):**

- [1] www.uclg.org — own-source revenue (tariffs and fees, property income) represent 65% of SNG revenue while grants and subsidies account for 35% (but respectively 43% and 57% for the municipal level alone). tax revenue. SNGs have considerable tax autonomy, g
- [2] www.sng-wofi.org — mining rents and royalties, and petroleum profit tax, are generally exclusively collected by the federal government. In FY2016, the share of VAT revenues for the three tiers of government was 35% for local governments, 50% for state governm
- [3] www.etd.ceu.edu — states or twenty-nine percent of the states received forty-one percent of the revenue. This analysis is even more interesting when one looks at the North which controls the greatest number of states – fourteen states or thirty-seven percent
- [4] documents.worldbank.org — of the revenues allocated to all state governments in the ﬁnal horizontal distribution (see table 4). One of the reasons that they receive this high share is that among these nine produc- ing states are some of the most highly populated sta
- [5] www.stears.co — money it has generated from crude oil sales, customs and taxes, and so on. 13% of any revenue made from natural resources is returned to the origin state of those resources, in line with the derivation principle. Other deductions are made f
- [6] www.nigerianstat.gov.ng — for the States and Local governments and accordingly which captures factors/principles and percentage is as follows: Equality – 40%. Here 40% of disbursable revenue from the share of the States 26.72% from the Federation account is is divid
- [7] eiti.org — beneficiaries of the 13% derivation, (an exclusive preserve only for oil producing states). From 2007 to 2011, N22.35 trillion (US $125 billion) of oil, gas and mining revenues arrived in state coffers. That sum is equivalent to almost 15% 
- [8] www.etd.ceu.edu — of state exercise. The group wrote that “the people of Rivers State feel that they are losing something vital because, in the past, they were one out of 12 and, today, they are one out of 30. They feel that the other parts of Nigeria are en

### train:11 (gold=Refuted, base=Not Enough Evidence)

- **Claim:** 45% of Nigeria's revenue goes to the federal government.
- **Draft status (bank metadata):** PENDING_REVIEW

**Draft text (verbatim, from memory record):**

> The excerpt explicitly distinguishes an extant federal share of 52.68% of statutory/mineral revenue from a 15% VAT share. These are different baskets, not evidence that 45% is supported. Under the federation-account reading the stated 52.68% contradicts 45%, whereas the Base treats mismatched baskets as a reason to abstain.

**Evidence inventory (Top-8 rendered spans):**

- [1] kingcenter.stanford.edu — government to 50%), 30% to thestate, 15% to the local, 8% to special funds(changedby the federal government to 5%), 1% to special funds, andso on. Horizontal allocation was to be distributed as 40% equality of states, 30% by population, 10%
- [2] www.stears.co — Nigeria is a federation. In practice, this means that the Nigerian government operates at three levels; federal, state, and local, with each responsible for different functions and responsibilities. For example, the Federal Government (FG) 
- [3] kingcenter.stanford.edu — the table exposesthat the federal level receives all oil revenue, which in turn comprisesover 70% of total federal revenue. Further, since oil as a percentageof total revenuehas consistently beenabove50% of total revenue(with a high of 86% 
- [4] documents.worldbank.org — Nigeria, with the largest oil production being in the states of Akwa-Ibom, Rivers, Delta, Bayelsa and Cross River. The revenues from oil production are distributed across all levels of government—federal, state and local. The sharing formul
- [5] tribuneonlineng.com — from the percentage difference. According to the extant revenue formula, the Federal Government takes 52.68 per cent, states got 27.72 per cent, while councils were entitled to 20.60 per cent of statutory and mineral revenue. In the distrib
- [6] www.etd.ceu.edu — Table 4.6 below shows that despite several amendments to the vertical revenue sharing formula since 1970, the federal government consistently maintained its control over almost a half of the revenue in the Federation Account. As can be seen
- [7] www.imf.org — as follows: Federal Government (14 percent), Federal Capital Territory (1 percent), State Governments (50 percent), and Local Governments (35 percent). 3This has been a contentious issue over the years. Initially the law was interpreted as 
- [8] www.oxfordmartin.ox.ac.uk — revenues, far from the figures often quoted. • According to available data, Nigeria’s Federal Government collects 5.53% of GDP in revenue, while if the taxes collected by State governments are included, the figure rises to 6.27%. As state g

### train:25 (gold=Supported, base=Refuted)

- **Claim:** The Kenyan judiciary does not even get 1% of the national budget.
- **Draft status (bank metadata):** PENDING_REVIEW

**Draft text (verbatim, from memory record):**

> The Base correctly observes sub-1% allocations but interprets not even 1% as a minimum of 1%. Not even 1% means below 1%. The displayed 0.43% and 0.44% figures support that comparison rather than refute it.

**Evidence inventory (Top-8 rendered spans):**

- [1] s3-eu-west-1.amazonaws.com — 0.69% of the total national budget. In the year 2018/2019, the allocation was Kshs. 14.5B translating to 0.43% of the total national budget. In the current financial year, the allocation stands at 18.8B, which is about 0.44% of the budget. 
- [2] icj-kenya.org — Independence of the Judiciary provide that the State must provide adequate resources to enable the Judiciary as an institution to perform its functions properly. Indeed, even while the Judiciary Fund has not yet been actualised, the fact th
- [3] elfafrica.org — first drastically reduced to KES 17.4 billion, and later suffered a further reduction to KES 14.5 billion, an over 50% deficit cut. In the previous financial year, the Judiciary received only 40% of its budgetary requirement with a resultan
- [4] s3-eu-west-1.amazonaws.com — the adequacy of resources to the Judiciary as an arm of government, are crucial determinants to its independence and accountability. Yet, the Judiciary receives a meagre 0.44% of the national budget, and the establishment of the Judiciary F
- [5] elfafrica.org — The close of last year (2019) saw the Judiciary revolt against what it termed as ‘control by the executive’ due to budgetary cuts from treasury. Chief Justice David Maraga lashed out at the executive and the current finance Cabinet Secretar
- [6] www.aljazeera.com — Kenya’s top judge: Court system is being ‘starved’ of funds The judiciary was allocated just over half of its requested budget for the current fiscal year. Kenya‘s chief justice on Monday lashed out at budget cuts that he said were intended
- [7] s3-eu-west-1.amazonaws.com — analysis of Judiciaries budget. The analysis focuses on: Judiciaries budget as a percentage of total Government budget; Budget as a percentage of GDP; and Judiciary per capita budget Memorandum of the Judicial Service Commission (JSC) to th
- [8] s3-eu-west-1.amazonaws.com — of 50 million Kenyans. With the Sustaining Judiciary Transformation (SJT) blueprint’s objective to establish a magistrate’s court in all the sub counties in the country, the financial outlay required would be significant. It would require f

### train:978 (gold=Supported, base=Not Enough Evidence)

- **Claim:** In Serbia, the stricter Covid19 curfew measures prevented dog walking for up to 14 hours at a time.
- **Draft status (bank metadata):** PENDING_REVIEW

**Draft text (verbatim, from memory record):**

> The official Serbian notice gives weekend movement restrictions from 15.00 to 5.00 and withdraws the 20-minute pet-walk exception. Across midnight that interval is 14 hours. The Base considers only the weekday 17.00-to 5.00 interval, so its claim that no 14-hour interval is shown overlooks the weekend measure.

**Evidence inventory (Top-8 rendered spans):**

- [1] www.washingtontimes.com — BELGRADE, Serbia — Pet owners in Serbia are furious over the populist government’s decision to ban even a brief walk for people with dogs during an evening curfew to contain the spread of the new coronavirus. Angry dog owners have flooded s
- [2] www.srbija.gov.rs — a residence permit, without exception, are obliged to respect the extended period of home self-isolation, which the competent authorities will regularly check in the coming period. The duration of the ban on citizens' movement for weekend d
- [3] www.boston.com — on Saturday, revoking a previously introduced 20-minute permission for dog owners to walk their pets. Serbia has imposed some of the harshest measures in Europe against the spread of the new coronavirus, including a total ban on movement fo
- [4] guernseypress.com — the virus and 38,042 infections. North Macedonia has reported two more deaths to raise the death toll to six. They are both men in their 30s. More than 9,000 people in the country of 2.1 million are in quarantine or in self-isolation. The c
- [5] library.fes.de — Since March 22 a “Stay at Home” campaign has been active for the whole population and movement restriction was made mandatory for everybody under 18 (FBiH) and over 65 (24hrs). The cur- few hours are from 8pm – 5am (FBiH, RS) and from 9pm –
- [6] library.fes.de — despite a lack of notice given, etc.). Moreover, there are indications that legal measures on isolation were imposed retroactively, against the Constitution. Restrictive measures on the free- dom of movement increased in almost day by day c
- [7] guernseypress.com — in people’s homes.” Serbia’s government made the decision on Saturday, revoking a previously introduced 20-minute permission for dog owners to walk their pets. Serbia has imposed some of the harshest measures in Europe against the spread of
- [8] www.aa.com.tr — Serbia on Wednesday imposed a night curfew to curb the coronavirus outbreak. President Aleksandar Vucic in a live televised address to the nation asked people aged above 60 years to stay at home at all times. The curfew will be in place fro

### train:1824 (gold=Refuted, base=Conflicting Evidence/Cherrypicking)

- **Claim:** Former British Prime Minister Winston Churchill said, 'Courage is what it takes to stand up and speak; courage is also what it takes to sit down and listen'.
- **Draft status (bank metadata):** PENDING_REVIEW

**Draft text (verbatim, from memory record):**

> Generic quote pages repeat the Churchill attribution, but the displayed correction explicitly cites the Churchill Society's falsely-attributed list and says the quotation appears nowhere in the Churchill canon. Repetition is not independent authorship evidence. This specific attribution correction supports Refuted instead of treating all quote pages as equally reliable conflicting accounts.

**Evidence inventory (Top-8 rendered spans):**

- [1] www.turningmanagersintoleaders.com — “Courage is what it takes to stand up and speak, Courage is also what it takes to sit down and listen.” — Sir Winston Churchill, British politician (1874 – 1965) at a conference in Washington DC I received this quote and photo from a client
- [2] www.beaninspirer.com — Everybody wants to speak. Everybody wants to show off their knowledge. Yes, it requires courage to do so. But people think that silence is weakness. Wrong! It is not a weakness. When a person is silent, it does not mean he is in the dormant
- [3] www.startmarriageright.com — Courage is what it takes to stand up and speak; courage is also what it takes to sit down and listen. –Winston Churchill This insightful quote sums up marriage in a very succinct manner. It is relatively easy to get married but it takes imm
- [4] falschzitate.blogspot.com — und zu sprechen Mut ist auch, was braucht, um sich hinzusetzen und zuzuhören." (Dieser Satz wurde wohl von künstlicher Intelligenz übersetzt.) International Chuchill Society: "Quotes Falsely Attributed to Winston Churchill" (winstonchurchil
- [5] www.pingree.org — Winston Churchill said, “Courage is what it takes to stand up and speak; courage is also what it takes to sit down and listen.” Knowing when to do each is an art form learned over a lifetime. As the Director of Prep@Pingree it is my great p
- [6] equip.sbts.edu — Winston Churchill is one of the most quoted leaders of all time, yet many modern citations put words in the lion’s mouth. You’ve probably heard this one: “Courage is what it takes to stand up and speak; courage is also what it takes to sit 
- [7] falschzitate.blogspot.com — Experten von der International Churchill Society haben diesen Satz weder so noch so ähnlich in einem Text oder Interview Churchills entdecken können (Link). Winston Churchill hat also niemals Leute als mutig bezeichnet, weil sie sich nieder
- [8] www.masslive.com — British Prime Minister Winston Churchill. Elwell makes use of a Churchill quote on the school website: “Courage is what it takes to stand up and speak; courage is also what it takes to sit down and listen.”

## Part B — Draftable expansion candidates (from step-1 triage)

These are **candidates**, not drafts. Expansion proceeds only after human go, and only with the listed grounded evidence. E2 rows additionally require including the identified store passage before drafting.

| ID | gold | base | class | subflags | decisive grounded evidence |
|---|---|---|---|---|---|
| train:2 | Supported | Not Enough Evidence | E2 | RETRIEVAL_MISS | Top-8[1]: "Between 1976 and 1992 the revenue allocation by the federal government to the local govern…" |
| train:8 | Conflicting Evidence/Cherrypicking | Supported | E3 | SOURCE_CIRCULARITY | Top-8[3]: "supported the 2003 invasion of Iraq, but he now calls it a “mistake”…" <br>Top-8[1]: "Joe Biden voted for the Iraq War, which President Trump has long called…" |
| train:12 | Conflicting Evidence/Cherrypicking | Refuted | E3 | — | Top-8[1]: "Donald Trump inherited an economy that had largely recovered…" <br>Top-8[4]: "low growth, or “secular stagnation,” would plague the U.S. economy indefinitely…" |
| train:17 | Supported | Conflicting Evidence/Cherrypicking | E2 | SPAN_TRUNCATION, BIBLIOGRAPHIC_ONLY | Top-8[1]: "More Likely to Kill than Be Killed by a Criminal in 2019, POLITIFACT (Aug. 27, 2020)…" |
| train:18 | Conflicting Evidence/Cherrypicking | Not Enough Evidence | E3 | ONTOLOGY | Top-8[3]: "Evidence for the first-time voting boost among 16- to 25-year-olds in Austria…" <br>Top-8[6]: "turnout across the country as a whole for National Assembly and local council elections ar…" |
| train:19 | Conflicting Evidence/Cherrypicking | Refuted | E3 | — | Top-8[1]: "33 are currently stalled and seven broken…" <br>Top-8[4]: "decades of academic research has actually shown the opposite…" |
| train:20 | Refuted | Supported | E2 | RETRIEVAL_MISS | Top-8[3]: "a project partly sponsored by the Governor Godwin Obaseki-led administration in Edo State …" |
| train:21 | Not Enough Evidence | Supported | E3 | ONTOLOGY | Top-8[2]: "On average, there are 15,000 inhabitants for every judge or magistrate across the globe…" <br>Top-8[1]: "about 75 judges 1,000,000 persons…" |
| train:24 | Conflicting Evidence/Cherrypicking | Supported | E3 | ONTOLOGY | Top-8[1]: "felt less anxious and more connected to school when they were away from it…" <br>Top-8[8]: "average anxiety scores were 5.2 out of 10, a marked increase from 3.0…" |
| train:27 | Not Enough Evidence | Supported | E3 | PROMISE_VS_PROGRESS | Top-8[2]: "made the following promises to their people: (a) to allocate at least 10% of national budg…" |
| train:29 | Supported | Not Enough Evidence | E3 | PRECISION_DEMAND | Top-8[1]: "there were 397,243 cases filed in all of the courts of Kenya during FY 2013/14…" |
| train:31 | Not Enough Evidence | Refuted | E3 | ONTOLOGY | Top-8[5]: "Kenya with a population of 48 million has a judge to population ratio of 1 judicial…" <br>Top-8[7]: "well below the national global average of 17…" |
| train:35 | Not Enough Evidence | Conflicting Evidence/Cherrypicking | E3 | CAUSAL_ATTRIBUTION_MISSING, SERIES_DIVERGENCE | Top-8[8]: "the rate of unemployment, which stood at 11.9% in 2015, dropped to 7.3% in 2019…" <br>Top-8[4]: "Ghana unemployment rate for 2019 was 3.46%…" |
| train:37 | Supported | Conflicting Evidence/Cherrypicking | E3 | — | Top-8[5]: "annual average of 7%, up from 3.4% in 2016, the lowest in nearly three decades…" |
| train:47 | Refuted | Not Enough Evidence | E3 | — | Top-8[1]: "third biggest drinking nation in Africa, and the 19th biggest drinking nation in the world…" |
| train:48 | Refuted | Not Enough Evidence | E3 | — | Top-8[4]: "below Namibia (31.3 litres), Eswatini (32.7 litres), Cook Islands (32.9 litres) and Tunisi…" |
| train:49 | Not Enough Evidence | Supported | E3 | SCOPE_QUALIFIER | Top-8[1]: "an estimated 31% of the population aged 15 and older consume alcohol…" <br>Top-8[2]: "of the 35% of the population that do consume alcohol…" |
| train:979 | Conflicting Evidence/Cherrypicking | Not Enough Evidence | E3 | — | Top-8[2]: "shoes can be a potential source of contamination…" <br>Top-8[5]: "the likelihood of spreading or contracting the virus from shoes is low…" |
| train:1863 | Refuted | Conflicting Evidence/Cherrypicking | E3 | ONTOLOGY | Top-8[1]: "The presidency has dismissed media reports claiming President Muhammadu Buhari approved th…" <br>Top-8[3]: "President Muhammadu Buhari Monday in Abuja approved the creation of state and local govern…" |
| train:2449 | Conflicting Evidence/Cherrypicking | Refuted | E2 | RETRIEVAL_MISS, CAUSAL_LINK_UNEVIDENCED | Top-8[2]: "prison population is substantially inflated by the presence of drug smugglers who are nati…" |

ONTOLOGY sign-off required for: train:18, train:1863, train:21, train:24, train:31.

## Part C — Not draftable (E1, no action)

- **train:3** (gold=Supported, base=Refuted; GOLD_TENSION): Gold Supported is not derivable from the evidence: the displayed NBC fact-check ("This is misleading") and the store’s own full text ("he has hardly proposed amnesty and free health care for all undocumented immigrants; Biden supports allowing undocumented immigrants to purchase health care with their own money") support a mixed/refuted reading of the health-care element. A Supported draft would contradict the store’s decisive passage.
- **train:14** (gold=Supported, base=Not Enough Evidence; EVIDENCE_ELEMENTS_SPLIT): Claim elements (Buhari signing; new Police bill; warrantless-arrest power) never appear together: Top-8 has generic warrantless-arrest law fragments [2][7] and a 2017/2019 gunshot-treatment bill [1]; store and raw corpus contain no Buhari+Police-Act-2020+arrest link. Gold Supported cannot be grounded without speculation.
- **train:26** (gold=Not Enough Evidence, base=Supported; GOLD_READING_UNSUPPORTED): Both pledge elements are directly evidenced ([1][2][3][7] "no new taxes" pledge; [4] hikes "limited to people who make more than $400,000"). The NEI gold rests on an unstated pledge-vs-enacted-incidence reading (original limitation: "Need whether the claim verifies a speech act, proposed policy or enacted tax incidence"); forcing NEI would be speculation.
- **train:38** (gold=Supported, base=Not Enough Evidence; GOLD_BASELINE_ABSENT): The store’s only ~300,000 figure is a backlog count, not annual settlements ("the number of cases classified as back-log stood at just over 300,000"; "553,187 cases pending, 327,928 of which were backlogged"); no passage gives an annual disposal total. Grounding the Supported gold would require turning backlog into settlements — speculation.
- **train:39** (gold=Not Enough Evidence, base=Supported; GOLD_READING_UNSUPPORTED): [4] explicitly reports "the global average of 16" (judges and magistrates per 100,000), which grounds the claim; the original assessor already concluded the NEI label "is not entailed by denying all benchmark evidence". A NEI draft would have to dismiss displayed benchmark evidence.
- **train:1873** (gold=Refuted, base=Supported; PRIMARY_SOURCE_ABSENT): The claim attributes quotes to a then-Sen. Biden op-ed; the op-ed is absent from the corpus (original limitation) and the quoted phrases do not appear in any displayed span, so the Refuted gold (quote attribution wrong) cannot be grounded without speculation.
- **train:1881** (gold=Supported, base=Not Enough Evidence; NON_PROPOSITIONAL_CLAIM): The claim is a headline ("Pinning Down Prescription Drug Prices"), not a verifiable proposition; the displayed abstract supports a related thesis ("prescription drugs are priced in the United States primarily on the basis of what the market will bear") but the gold-Supported reading of the title itself cannot be grounded.
- **train:2224** (gold=Refuted, base=Conflicting Evidence/Cherrypicking; GOLD_READING_UNSUPPORTED): Displayed [2] supports the claim (embassy "reversed its earlier decision to ban visa applications by Kenyans without digital passport... will place a US visa in a non-digital passport until further notice"); the Refuted gold rests on an unstated temporal scope (later policy change not in evidence).

## Provenance

- `outputs/m161-review-preparation-v1/combined75.json` sha256 `8ec5d9fbc496b48c1483b64942a8b50bf7e41cee56d71eaec07b884c81047618`
- `outputs/m1611-correction-audit-v3/memory75.json` sha256 `b3f7275112aa49b71595d94c7313a4ae101b3ed263c277bb277f97def2565581`
- `docs/audits/M1_6_2D_EVIDENCE_TRIAGE.json` sha256 `c966fad71fc5bbb3fc19d69ff449d28dbb5a0579af071cbe5eb6cd104f2fb5ac`
