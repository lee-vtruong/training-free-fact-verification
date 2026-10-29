"""M1.6.2e section C: candidate correction drafts + two-reviewer review records.

CPU-only, read-only over immutable artifacts. Rules honored:
- 5 original drafts passed through verbatim (untouched).
- E3 candidate drafts quote EXACT Top-8 spans only (every quote asserted verbatim).
- E2 drafts are quarantined (evidence_availability_change=true; excluded from the
  primary reasoning-error experiment); their quotes come from the 2026-10-28 server
  store probes and are marked as such.
- E1 records stay no-draft.
- No human decisions are created; no VERIFIED marks anywhere.
Outputs: docs/audits/M1_6_2E_CANDIDATE_DRAFTS.json,
docs/audits/M1_6_2E_REVIEW_RECORDS_REVIEWER1.json / _REVIEWER2.json,
docs/M1_6_2E_REVIEW_PACKET.md, docs/M1_6_2E_MECHANISM_SUPPORT.md.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BANK = json.loads(Path('outputs/m161-review-preparation-v1/combined75.json').read_text('utf-8-sig'))
MEM = json.loads(Path('outputs/m1611-correction-audit-v3/memory75.json').read_text('utf-8-sig'))
TRI = json.loads(Path('docs/audits/M1_6_2D_EVIDENCE_TRIAGE.json').read_text('utf-8-sig'))
B = {r['sample_id']: r for r in BANK['records']}
M = {r['sample_id']: r for r in MEM['records']}
T = {r['sample_id']: r for r in TRI['records']}
DRAFT_IDS = ['train:6', 'train:11', 'train:25', 'train:978', 'train:1824']
NOW = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def norm(s):
    return ' '.join((s or '').split())


def bank_quote(sid, quote):
    q = norm(quote)
    for e in B[sid]['evidence']:
        t = norm(e.get('rendered_text'))
        if t.find(q) >= 0:
            return {'source': 'top8_span', 'rank': e['rank'], 'url': e.get('source_url', ''),
                    'quote': quote, 'span': [e.get('start_char'), e.get('end_char')],
                    'verified_in': 'bank_top8_asserted'}
    raise AssertionError('%s quote not verbatim in Top-8: %r' % (sid, quote[:60]))


E3 = {
 'train:8': dict(mechanism='source_circularity', ontology_note=(
    'Conflicting retained: the dispute is over the same proposition set (Iraq vote; the three-conflict '
    'enumeration) across comparable-type sources; the enumeration itself originates in the claim-source '
    'speech, which is a sourcing defect, not source disagreement.'),
  draft=(
    'The claim mixes a documented vote with elements attested mainly in its own source. Independent '
    'passages confirm the Iraq vote and its later repudiation - CFR: "supported the 2003 invasion of '
    'Iraq, but he now calls it a mistake" - and describe an interventionist record (Jacobin: "a reliable '
    'representative of the national security establishment"; Arab Center: a "mixed record" on US '
    'interventions). But the "Serbia, Syria, and Libya" enumeration comes from the CNN passage quoting '
    'the attack speech itself, and no displayed passage evidences a Syria war position. The Iraq element '
    'is supported while the three-conflict enumeration is partially unevidenced and circularly sourced; '
    'that mixed state is Conflicting Evidence/Cherrypicking, not the Base\'s flat Supported.'),
  limitations='Syria element unevidenced in displayed spans; Serbia/Libya supported only at general-intervention-record level.'),
 'train:12': dict(mechanism='selective_reading', ontology_note=(
    'Conflicting retained: the disagreement is between comparable-type sources over the same propositions '
    '(inherited condition; rebuild effect). Both sides carry partisan provenance (WH archives vs nwlc/medium), '
    'noted; partisan symmetry is not itself grounds for a different verdict.'),
  draft=(
    'Both sides of the dispute are displayed. Recovered-economy side: "Donald Trump inherited an economy '
    'that had largely recovered, according to many indicators, by the time he took office" (nwlc) and '
    '"Trump inherited the longest economic expansion in" modern history (medium). Stagnation-inheritance '
    'side: the White House archives passage frames the inheritance as stagnation - "low growth, or '
    '"secular stagnation," would plague the U.S. economy indefinitely" - and the Detroit News passage '
    'credits the "pro-growth policy agenda" with policies that "have inspired enormous confidence in the '
    'economy". The Base read only the recovered side and returned Refuted. With the inherited-stagnation '
    'framing displayed in the administration\'s own material and the rebuild element contested, the '
    'verdict is Conflicting Evidence/Cherrypicking.'),
  limitations='Both sides include partisan framing (administration archives vs advocacy outlets); tracker-level economic data not displayed.'),
 'train:18': dict(mechanism='scope_integration', ontology_note=(
    'Conflicting retained: the tension is about the claim\'s three-jurisdiction generalization (Austria/Scotland '
    'support vs a Wales-specific caution), i.e., claim-scope conflict, not mere source disagreement. Sign-off required.'),
  draft=(
    'Support exists for two of the three jurisdictions and a caution for the third, all inside the '
    'displayed spans. The bibliography cites Zeglovits & Aichholzer (2014), "Are people more inclined to '
    'vote at 16 than at 18? Evidence for the first-time voting boost among 16- to 25-year-olds in Austria", '
    'and the Wales passage itself continues: "Evidence from Scotland shows that in the 2014 independence '
    'referendum where 16-17 year olds were enfranchised to vote, 75 per[cent]..." The same passage warns '
    'that "Austria has traditionally high turnout rates for elections (Zeglovits & Aichholzer, 2014), which '
    'cannot be said for elections in Wales", where turnout is "typically below 50 per cent". The Base '
    'returned Not Enough Evidence without integrating the displayed title and caveat. The first-time-voting '
    'boost is evidenced for Austria and Scotland while the Wales leg rests on low-turnout context rather '
    'than a matched comparison - the three-jurisdiction generalization is supported in part and undermined '
    'in part, i.e., Conflicting Evidence/Cherrypicking.'),
  limitations='Austria support is bibliography-title-level; no matched 16-vs-18/20 Wales comparison is displayed.'),
 'train:19': dict(mechanism='selective_reading', ontology_note=(
    'Conflicting retained: tracker snapshots and the research framing are same-type evidence about the same '
    'proposition and point in different directions; the mix is genuine, not manufactured.'),
  draft=(
    'Both directions are displayed. Broken/stalled side: "Out of 102 promises on its Trump-o-Meter '
    'scorecard, 33 are currently stalled and seven broken, while only nine have been kept and seven ended '
    'with a compromise" (Time/PolitiFact), and "Only six promises are listed as "kept," while 45 have seen '
    'no action" (Denver Post). Kept-side framing: Time also reports "decades of academic research has '
    'actually shown the opposite: that presidents try to keep an average of about two-thirds of their '
    'campaign pledges". The Base read only the stalled/broken trackers and returned Refuted. With kept '
    'counts and the research framing displayed alongside the stalled counts, the mixed record supports '
    'Conflicting Evidence/Cherrypicking rather than outright refutation.'),
  limitations='Tracker snapshots differ in date and methodology; "try to keep" (research) vs "kept" (trackers) is not a single metric.'),
 'train:21': dict(mechanism='metric_mismatch', ontology_note=(
    'NEI protected from auto-Conflicting: the displayed ratios (judges-only per million; judicial officers '
    'per 100,000; worldwide judges-and-magistrates per 15,000 inhabitants) are different measurements, not '
    'opposing verdicts on one proposition. Insufficiency is the ontology-correct reading. Sign-off required.'),
  draft=(
    'The displayed figures cannot be combined into the asserted comparison. The only global figure is '
    'UNODC\'s "On average, there are 15,000 inhabitants for every judge or magistrate across the globe" - '
    'a worldwide judges-and-magistrates ratio. The Kenya-side passages use different metrics: "about 75 '
    'judges 1,000,000 persons" (judges only, per million) and a judicial-officer ratio per 100,000. No '
    'displayed passage states a global average on the same metric as the Kenya figures, nor a Kenya figure '
    'on the global metric, so the claim\'s relation cannot be checked. The Base accepted the worldwide '
    'figure as if it settled the Kenya relation; the verdict is Not Enough Evidence.'),
  limitations='Metric-comparability reading recorded for sign-off; an alternative reading (the UNODC figure itself answers the claim) is disclosed.'),
 'train:24': dict(mechanism='scope_conflation', ontology_note=(
    'Conflicting retained with scope note: the study finding (13-14-year-olds, South West of England) and '
    'the general-population increases (ONS adults; 21-40 age group) address different scopes; the gold reads '
    'the displayed tension as conflict. Sign-off required.'),
  draft=(
    'The displayed evidence both supports the study finding and contradicts the broader trend. Study side: '
    '"Younger teenagers in the South West of England felt less anxious and more connected to school when '
    'they were away from it during the COVID-19 global pandemic... a first-of-its-kind study has found", '
    'with figures that "dropped by nearly 10 per cent among girls to less than half (45 per cent) and to '
    'less than one in five of boys (18 per cent)". Counter-side: "average anxiety scores were 5.2 out of '
    '10, a marked increase from 3.0 in the last quarter of 2019" (ONS), and higher anxiety reported for the '
    '21-40 age group. The Base read the study side alone as confirmation. Because the study-specific drop '
    'and the general-population increases are both displayed, the verdict is Conflicting '
    'Evidence/Cherrypicking; the claim\'s "a study has suggested" attribution is preserved in the reading.'),
  limitations='Sign-off required on the scope reading (study report vs general trend).'),
 'train:27': dict(mechanism='promise_vs_progress', ontology_note=None, draft=(
    'The displayed evidence documents commitments, not striving. Counties "made the following promises to '
    'their people: (a) to allocate at least 10% of national budgets to agriculture; and (b) to achieve at '
    'least 6% annual agricultural growths" - the 10% target itself being a commitment to double the then '
    'average. The Malabo 10% target and expenditure-monitoring frameworks are displayed, but no displayed '
    'passage reports county-level allocations or progress toward the 10% mark. The Base equated the '
    'promise with progress and returned Supported. With only commitments displayed, there is not enough '
    'evidence that the counties are striving to reach the target.'),
  limitations='No material limitation beyond the promise/progress gap itself.'),
 'train:29': dict(mechanism='precision_demand', ontology_note=None, draft=(
    'The claimed magnitude is displayed directly: "there were 397,243 cases filed in all of the courts of '
    'Kenya during FY 2013/14". About 400,000 filings per year is what the claim asserts, and 397,243 is '
    'that figure within ordinary rounding of "an average of 400,000". The Base acknowledged the number but '
    'demanded an exact 400,000 and a multi-year average; the gold reading accepts the displayed magnitude. '
    'The single displayed fiscal year is the only limitation.'),
  limitations='Single fiscal year (FY 2013/14) displayed; no multi-year average shown.'),
 'train:31': dict(mechanism='metric_mismatch', ontology_note=(
    'NEI protected from auto-Conflicting (mirror of train:21): divergent officer ratios are different '
    'measurements, not opposing verdicts. Sign-off required.'),
  draft=(
    'The Base refuted the claim by contrasting Kenya\'s ratio - "Kenya with a population of 48 million has '
    'a judge to population ratio of 1 judicial officer" - against other countries\' ratios. But no displayed '
    'passage provides a global average for judges and magistrates on a comparable metric: the worldwide '
    'UNODC figure ("15,000 inhabitants for every judge or magistrate across the globe") uses a different '
    'denominator, and the "global average of 17" in the displayed reliefweb passage concerns public '
    'defenders per 100,000, a different officer class. With no comparable benchmark displayed, the claim '
    'can be neither confirmed nor refuted - Not Enough Evidence.'),
  limitations='Sign-off required; metric-comparability reading disclosed.'),
 'train:35': dict(mechanism='causal_attribution_missing', ontology_note=(
    'NEI protected from auto-Conflicting: the GLSS series (11.9%->7.3%) and the ILO-modelled series (3.46% '
    'in 2019) are different measurements with different methodologies, not opposing verdicts; insufficiency '
    'of the causal claim is the ontology-correct reading. Sign-off required.'),
  draft=(
    'The decline numbers are displayed - "the rate of unemployment, which stood at 11.9% in 2015, dropped '
    'to 7.3% in 2019" (Ghana Living Standards Survey, quoted in the presidential-speech passage) - but the '
    'claim attributes the decline to the administration\'s policies. Two displayed facts block that '
    'attribution: the ILO-modelled series reports "Ghana unemployment rate for 2019 was 3.46%", materially '
    'below the GLSS 7.3%, so the level and size of any decline depend on which measured series is used; '
    'and no displayed passage attributes the movement to administration policy rather than survey '
    'methodology. The Base returned Conflicting by weighing the series against each other; divergent series '
    'are different measurements, not opposing verdicts. The causal claim is unevidenced: Not Enough Evidence.'),
  limitations='Sign-off required; both series are displayed only at single-year resolution.'),
 'train:37': dict(mechanism='missed_direct_match', ontology_note=None, draft=(
    'The claim\'s figures are displayed verbatim: "annual average of 7%, up from 3.4% in 2016, the lowest '
    'in nearly three decades" - matching the claimed 3.4% under the prior government and an average of 7% '
    'per year under the current one. The Base returned Conflicting Evidence/Cherrypicking by weighing this '
    'passage against unrelated indicators instead of reading the direct match. The gold reads the displayed '
    'statement as sufficient support for the claimed growth figures.'),
  limitations='The figures appear in a presidential-speech report (ghanaweb column); single-source, government-attributed.'),
 'train:47': dict(mechanism='missed_direct_ranking', ontology_note=None, draft=(
    'The displayed ranking refutes the superlative: "This pushes South Africa up to the third biggest '
    'drinking nation in Africa, and the 19th biggest drinking nation in the world, tied with Poland." '
    'Third in Africa entails that at least two African countries consume more per person, so South Africans '
    'do not consume more alcohol per person "than anywhere else in Africa." The Base returned Not Enough '
    'Evidence despite the direct statement.'),
  limitations='WHO 2015-data framing; per-capita basis.'),
 'train:48': dict(mechanism='missed_direct_ranking', ontology_note=None, draft=(
    'The displayed passage names the countries above South Africa: "the fifth highest consumption rate in '
    'the world, below Namibia (31.3 litres), Eswatini (32.7 litres), Cook Islands (32.9 litres) and Tunisia '
    '(33.4 litres)". Namibia, Eswatini and Tunisia are African and rank above South Africa, so South '
    'Africans do not drink more per person than anywhere else in Africa. The Base returned Not Enough '
    'Evidence despite the explicit comparison.'),
  limitations='Passage mixes world-ranking (28.9 litres, 2005-era WHO figures) with the per-capita claim; ranking direction is unambiguous.'),
 'train:49': dict(mechanism='scope_qualifier', ontology_note=(
    'NEI protected from auto-Conflicting: the displayed estimates (31% of 15+; 31% of adults past-year; 35%; '
    '33.1%) differ by scope and denominator, so they are not opposing verdicts on one proposition; '
    'insufficiency of the unqualified claim is the ontology-correct reading. Sign-off required.'),
  draft=(
    'The displayed estimates diverge by scope and denominator: "an estimated 31% of the population aged 15 '
    'and older consume alcohol" (WHO 2016 data), "only a third (31%) of adults report consuming alcohol in '
    'the past year", "the 35% of the population that do consume alcohol", and a 33.1% figure in the BMC '
    'passage. The unqualified "Only 31% of South African drink alcohol" specifies neither population, year '
    'nor survey; with scoped estimates spanning 31-35%, the unqualified figure is not settled. The Base '
    'accepted the first 31% without its scope qualifier. Not Enough Evidence.'),
  limitations='Sign-off required on the scope reading.'),
 'train:979': dict(mechanism='incomplete_integration', ontology_note=(
    'Conflicting retained: this is a genuinely contested possibility claim - surface persistence is displayed '
    'alongside explicit low-likelihood assessments. The tension is evidential, not a mere difference between '
    'source types.'),
  draft=(
    'The displayed evidence genuinely splits on the transmission question. Possibility side: "shoes can be a '
    'potential source of contamination, according to family practitioner Georgine Nanos"; persistence is '
    'displayed as "coronavirus can remain active on some surfaces, like plastic, for up to two to three '
    'days", with shoe soles discussed as virus carriers. Low-risk side: "the likelihood of spreading or '
    'contracting the virus from shoes is low", and the Stanford NEJM-based comment: "it sees unlikely that '
    'enough virus would land on the floor, then get transmitted to shoes". The Base collapsed this displayed '
    'tension into Not Enough Evidence. Both a transmission route treated as possible and an explicit '
    'low-likelihood assessment are displayed: Conflicting Evidence/Cherrypicking.'),
  limitations='Risk assessments are expert commentary, not measurement; possibility vs probability framing disclosed.'),
 'train:1863': dict(mechanism='misreport_denial_weighting', ontology_note=(
    'Refuted retained: the presidency denial directly addresses the reported approval - this is a refutation '
    'of the reported act, not a mere source disagreement. Sign-off required on denial-weighting.'),
  draft=(
    'The refutation chain is displayed: the report - "President Muhammadu Buhari Monday in Abuja approved '
    'the creation of state and local government police" - and the same-day dismissal - "The presidency has '
    'dismissed media reports claiming President Muhammadu Buhari approved the establishment of State and '
    'Local Government Police." The approval the claim asserts was reported and then dismissed by the '
    'presidency. The Base weighted the pair as mere tension and returned Conflicting Evidence/Cherrypicking; '
    'the gold reads the dismissal of the underlying report as refuting the claimed approval.'),
  limitations='Denial-weighting reading recorded for sign-off; no independent confirmation of the denial displayed.'),
}

E2 = {
 'train:2': dict(mechanism='retrieval_rank_miss', evidence_availability_change=True, draft=(
    'The target store contains the current statutory split: "The Federation Account revenue sharing formula '
    'is as follows: FG (52.68%), SGs (26.72%) and LGs (20.60%)" (CBN chapter five), and "In the current '
    'revenue allocation formula, the FG gets 52.68%, State Governments get 26.72% and the Local Governments '
    'get 20.60%" (LinkedIn explainer). A 20% share to local government is grounded by the current formula '
    '(20.60%); the Top-8 missed both passages, leaving only the historical 1976-1992 increase. '
    'Supported per the gold, conditional on including the store passages.'),
  limitations='Formula vintage: the LinkedIn "current" is undated and the CBN chapter is 2013-era; the claim itself is undated.'),
 'train:17': dict(mechanism='span_truncation', evidence_availability_change=True, draft=(
    'The retrieved law-review passage contains the complete citation: "Madeline Heim, Yes, Police Were 20.8 '
    'More Likely to Kill than Be Killed by a Criminal in 2019, POLITIFACT (Aug. 27, 2020)". The ratio the '
    'claim asserts is attested bibliographically; the rendered span began mid-title and cut the number, '
    'which is why the Base saw no usable figure. Supported per the gold, conditional on reading the full '
    'passage rather than the truncated span.'),
  limitations='Attestation is a fact-check title cited inside a law review; not primary 2019 data.'),
 'train:20': dict(mechanism='retrieval_rank_miss', evidence_availability_change=True, draft=(
    'The operator\'s own project page states: "A refinery 100% fully owned and developed by AIPCC. It is '
    'located in Ologbo, Ikpoba Okha LGA of Edo State." The Ologbo modular refinery is AIPCC\'s project; the '
    'displayed Top-8 framing ("partly sponsored by the Governor Godwin Obaseki-led administration... being '
    'developed by AIPCC Energy Limited") describes sponsorship and facilitation, not ownership. '
    '"Governor Obaseki\'s project" is refuted by the ownership statement. Refuted per the gold, conditional '
    'on including the store passage.'),
  limitations='Company self-report; the state-sponsored framing coexists and the ownership/sponsorship line is the decisive distinction.'),
 'train:2449': dict(mechanism='retrieval_miss', evidence_availability_change=True, draft=(
    'The causal premise\'s existence is recoverable: the ICJ Russia profile reports "the phenomenon of '
    '"telephone justice" remained widespread: reportedly, judges are often - as a matter of routine - '
    'directly instructed by the court president how to resolve a specific case", and the retrieved '
    '"Telephone Law and the Rule of Law: The Russian Case" material cites Ledeneva\'s "Telephone Justice in '
    'Russia" (Post-Soviet Affairs, 2008). But the displayed prison-rate passages show Russia\'s rates high '
    'and period-dependent ("The incarceration rate (per 100,000 of the population) was 528 in 2004"), with '
    'no passage linking telephone justice to prison population. The premise is real while the because-clause '
    'is unevidenced: Conflicting Evidence/Cherrypicking per the gold, conditional on including the '
    'store/corpus passages.'),
  limitations='Causal link unevidenced; telephone-justice evidence is qualitative; rate comparisons span different years/definitions.'),
}

records = []
# --- Part A: originals (verbatim passthrough) ---
for sid in DRAFT_IDS:
    b, m = B[sid], M[sid]
    dec = T[sid]['decisive_evidence'] if sid in T else []
    records.append({
        'record_kind': 'ORIGINAL_DRAFT', 'sample_id': sid, 'claim': b['claim'],
        'gold_verdict': b['gold_verdict'], 'base_verdict': b['base_prediction']['stored_verdict'],
        'draft_text': m.get('corrected_explanation'),
        'draft_text_provenance': 'outputs/m1611-correction-audit-v3/memory75.json corrected_explanation (verbatim, untouched)',
        'mechanism': None, 'limitations': None, 'ontology_note': None,
        'decisive_quotes': dec, 'evidence_availability_change': False,
        'excluded_from_primary_experiment': False,
        'triage_class': 'ORIGINAL', 'author': 'assistant (M1.6.1, untouched per M1.6.2e C.1)',
        'drafted_at': NOW, 'review_status': 'PENDING_REVIEW', 'verified': False,
        'reviews': {'reviewer_1': None, 'reviewer_2': None},
    })
# --- Part B: E3 candidates (exact Top-8 spans) ---
for sid, d in sorted(E3.items(), key=lambda kv: int(kv[0].split(':')[1])):
    t = T[sid]
    quotes = [dict(bank_quote(sid, q['quote']), note=q.get('note')) for q in t['decisive_evidence']
              if q['source'] == 'bank_top8']
    records.append({
        'record_kind': 'E3_CANDIDATE_DRAFT', 'sample_id': sid, 'claim': t['claim'],
        'gold_verdict': t['gold_verdict'], 'base_verdict': t['base_verdict'],
        'draft_text': d['draft'], 'draft_text_provenance': 'authored from exact Top-8 spans (M1.6.2e C.2)',
        'mechanism': d['mechanism'], 'limitations': d['limitations'], 'ontology_note': d['ontology_note'],
        'decisive_quotes': quotes, 'evidence_availability_change': False,
        'excluded_from_primary_experiment': False,
        'triage_class': 'E3', 'subflags': t['subflags'],
        'author': 'assistant (M1.6.2e, CPU only)', 'drafted_at': NOW,
        'review_status': 'PENDING_REVIEW', 'verified': False,
        'reviews': {'reviewer_1': None, 'reviewer_2': None},
    })
# --- Part C: E2 candidates (quarantined) ---
for sid, d in sorted(E2.items(), key=lambda kv: int(kv[0].split(':')[1])):
    t = T[sid]
    sp = t['store_probe']
    quotes = []
    for h in sp['decisive_hits']:
        quotes.append({'source': 'store_probe', 'url': h['url'], 'quote': h['quote'],
                       'passage_id': h.get('passage_id'), 'in_bank_top8': h.get('in_bank_top8'),
                       'verified_in': 'server_store_probe_20261028'})
    records.append({
        'record_kind': 'E2_CANDIDATE_DRAFT_QUARANTINED', 'sample_id': sid, 'claim': t['claim'],
        'gold_verdict': t['gold_verdict'], 'base_verdict': t['base_verdict'],
        'draft_text': d['draft'], 'draft_text_provenance': 'authored from server store-probe passages (M1.6.2e C.6)',
        'mechanism': d['mechanism'], 'limitations': d['limitations'], 'ontology_note': None,
        'decisive_quotes': quotes, 'evidence_availability_change': True,
        'excluded_from_primary_experiment': True,
        'triage_class': 'E2', 'subflags': t['subflags'],
        'store_probe': {'store': sp['store'], 'lines': sp['lines'], 'corpus_probe': sp.get('corpus_probe')},
        'author': 'assistant (M1.6.2e, CPU only)', 'drafted_at': NOW,
        'review_status': 'PENDING_REVIEW', 'verified': False,
        'reviews': {'reviewer_1': None, 'reviewer_2': None},
    })

# --- Part D: E1 no-draft stubs (recorded, not drafted) ---
for sid in sorted(T, key=lambda s: int(s.split(':')[1])):
    t = T[sid]
    if t['triage_class'] != 'E1':
        continue
    records.append({
        'record_kind': 'E1_NO_DRAFT', 'sample_id': sid, 'claim': t['claim'],
        'gold_verdict': t['gold_verdict'], 'base_verdict': t['base_verdict'],
        'draft_text': None, 'mechanism': None, 'limitations': t['rationale'],
        'ontology_note': None, 'decisive_quotes': t['decisive_evidence'],
        'evidence_availability_change': False, 'excluded_from_primary_experiment': True,
        'triage_class': 'E1', 'subflags': t['subflags'],
        'review_status': 'NOT_DRAFTABLE', 'verified': False,
        'reviews': {'reviewer_1': None, 'reviewer_2': None},
    })

out = {
    'schema_version': 'm162e-candidate-drafts-v1',
    'generated_utc': NOW,
    'scope': ('M1.6.2e section C candidate correction drafts + two-reviewer review records. '
              'EXPLORATORY. No human decisions created; verified=false everywhere; '
              'assistant drafts are NOT human-verified.'),
    'counts': {
        'original_drafts': sum(1 for r in records if r['record_kind'] == 'ORIGINAL_DRAFT'),
        'e3_candidates': sum(1 for r in records if r['record_kind'] == 'E3_CANDIDATE_DRAFT'),
        'e2_candidates_quarantined': sum(1 for r in records if r['record_kind'].startswith('E2_')),
        'e1_no_draft': sum(1 for r in records if r['record_kind'] == 'E1_NO_DRAFT'),
    },
    'review_protocol': {
        'reviewers': ['reviewer_1', 'reviewer_2'],
        'decisions': ['ACCEPT', 'ACCEPT_WITH_EDITS', 'REJECT'],
        'blank_fields': ['decision', 'edited_text', 'rationale', 'ontology_signoff', 'reviewed_at'],
        'verification_rule': ('A record counts as verified only after BOTH reviewers submit decisions and '
                              'the human review bookkeeping records it. The assistant never fills these fields.'),
        'verified_count': 0,
        'gate_note': 'See docs/M1_6_2E_GATE_POLICY.md for the corrected two-tier gate.',
    },
    'records': records,
}
dest = Path('docs/audits/M1_6_2E_CANDIDATE_DRAFTS.json')
dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')

# reviewer sheets (identical blank records; decisions are filled by humans, never by the assistant)
for who in ('REVIEWER1', 'REVIEWER2'):
    p = Path('docs/audits/M1_6_2E_REVIEW_RECORDS_%s.json' % who)
    p.write_text(json.dumps({'reviewer': who.lower(), 'generated_utc': NOW,
                             'instructions': 'Fill decision/edited_text/rationale/ontology_signoff/reviewed_at per record. Assistant must not fill these.',
                             'records': out['records']}, indent=1, ensure_ascii=False), encoding='utf-8')

# --- mechanism support report ---
sup = []
A = sup.append
A('# M1.6.2e — Per-Mechanism Verified Support Report')
A('')
A('Generated: %s · Status: **0 human-verified corrections** (assistant drafts are NOT verified).'
  % NOW)
A('')
A('Candidate pool by mechanism (from `docs/audits/M1_6_2E_CANDIDATE_DRAFTS.json`):')
A('')
A('| mechanism | class | candidate IDs | n | verified | status |')
A('|---|---|---|---|---|---|')
mech = {}
for r in records:
    if r['mechanism']:
        mech.setdefault((r['mechanism'], r['triage_class']), []).append(r['sample_id'])
for (mname, cls), ids in sorted(mech.items()):
    A('| %s | %s | %s | %d | 0 | PENDING_HUMAN_REVIEW |' % (mname, cls, ', '.join(ids), len(ids)))
A('')
A('Mechanism → verified support will be updated only from human review bookkeeping. The three-arm '
  'feasibility experiment may use an exemplar only after it is human-verified (both reviewers), and '
  'E2-quarantined mechanisms (`retrieval_rank_miss`, `span_truncation`, `retrieval_miss`) are excluded '
  'from the primary reasoning-error experiment regardless of verification status.')
Path('docs/M1_6_2E_MECHANISM_SUPPORT.md').write_text('\n'.join(sup), encoding='utf-8')

# --- review packet (rendered) ---
pk = []
P = pk.append
P('# M1.6.2e — Correction Review Packet (two reviewers)')
P('')
P('Generated: %s · EXPLORATORY · 5 original drafts (untouched) + 16 E3 candidate drafts + '
  '4 E2 candidates (quarantined) + 8 E1 no-draft records.' % NOW)
P('')
P('**Rules.** (1) Decisions per record: `ACCEPT` / `ACCEPT_WITH_EDITS` / `REJECT`; both reviewers fill '
  '`decision`, optional `edited_text`, `rationale`, `ontology_signoff` (mandatory for records marked '
  'SIGN-OFF), `reviewed_at`. (2) Grounded expansion only: support must quote the displayed spans '
  '(or, for E2, the identified store passages); never fabricate to reach a gate. (3) The assistant '
  'never fills reviewer fields and never marks VERIFIED. (4) E2 records are quarantined from the '
  'primary reasoning-error experiment (evidence-availability change declared). (5) Current verified '
  'count: **0**.')
P('')
for r in records:
    if r['record_kind'] == 'E1_NO_DRAFT':
        continue
    P('---')
    P('')
    P('### %s — %s (gold=%s, base=%s)%s' % (
        r['sample_id'], r['record_kind'], r['gold_verdict'], r['base_verdict'],
        ' · QUARANTINED (E2)' if r['excluded_from_primary_experiment'] and r['record_kind'].startswith('E2') else ''))
    P('')
    P('- **Claim:** %s' % r['claim'])
    P('- **Mechanism:** %s' % (r['mechanism'] or '—'))
    if r.get('ontology_note'):
        P('- **Ontology note (re-review):** %s' % r['ontology_note'])
    if r['limitations']:
        P('- **Limitations:** %s' % r['limitations'])
    P('')
    P('**Candidate draft:**')
    P('')
    P('> ' + (r['draft_text'] or '').replace('\n', '\n> '))
    P('')
    P('**Decisive quotes (%s):**' % r['decisive_quotes'][0]['source'] if r['decisive_quotes'] else '**Decisive quotes:** none recorded')
    for q in r['decisive_quotes']:
        src = ('Top-8 [%s] %s' % (q.get('rank'), (q.get('url') or '').split('/')[2] if q.get('url') else '-')
               if q['source'] == 'top8_span' else
               'store probe: %s' % ((q.get('url') or '')[:80]))
        P('- %s — "%s"' % (src, (q['quote'] or '')[:220]))
    P('')
Path('docs/M1_6_2E_REVIEW_PACKET.md').write_text('\n'.join(pk), encoding='utf-8')

print('records:', len(records), '| counts:', out['counts'])
for f in ('docs/audits/M1_6_2E_CANDIDATE_DRAFTS.json', 'docs/audits/M1_6_2E_REVIEW_RECORDS_REVIEWER1.json',
          'docs/audits/M1_6_2E_REVIEW_RECORDS_REVIEWER2.json', 'docs/M1_6_2E_REVIEW_PACKET.md',
          'docs/M1_6_2E_MECHANISM_SUPPORT.md'):
    print(f, Path(f).stat().st_size, 'bytes | sha256', hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16])
