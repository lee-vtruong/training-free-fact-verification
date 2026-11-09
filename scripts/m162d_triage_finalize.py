"""M1.6.2d step-1 finalize: write docs/audits/M1_6_2D_EVIDENCE_TRIAGE.json.

CPU-only, read-only over immutable artifacts (combined75 bank + m1611-v3 memory).
Every bank-level quote is asserted verbatim against the bank evidence spans;
store-level quotes were verified on hvtham-server probes on 2026-10-28 (paths
recorded in store_probes). Deterministic; EXPLORATORY artifact, not confirmatory.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BANK_PATH = Path('outputs/m161-review-preparation-v1/combined75.json')
MEM_PATH = Path('outputs/m1611-correction-audit-v3/memory75.json')
OUT_PATH = Path('docs/audits/M1_6_2D_EVIDENCE_TRIAGE.json')

bank = json.loads(BANK_PATH.read_text('utf-8-sig'))
mem = json.loads(MEM_PATH.read_text('utf-8-sig'))
B = {r['sample_id']: r for r in bank['records']}
M = {r['sample_id']: r for r in mem['records']}


def norm(s):
    return ' '.join((s or '').split())


def find_quote(sid, quote):
    """Locate a verbatim quote across all bank evidence items; assert presence."""
    q = norm(quote)
    for e in B[sid]['evidence']:
        t = norm(e.get('rendered_text'))
        i = t.find(q)
        if i >= 0:
            return {'source': 'bank_top8', 'rank': e['rank'],
                    'url': e.get('source_url', ''), 'quote': quote,
                    'span': [e.get('start_char'), e.get('end_char')]}
    raise AssertionError('%s: quote not found in bank Top-8: %r' % (sid, quote[:80]))


STORE_PROBES = [
    {'sample_id': 'train:2', 'store': 'outputs/m11-store-r14-pilot100-20261023-162716/train-2/passages.jsonl',
     'lines': 32979, 'bank_top8_passage_ids_present': 8,
     'decisive_hits': [
         {'passage_id': 'd9f3169eb0c4492d2cefb8276b4d7d98a7974de1f2d921de6a69aaf6a4fa77af',
          'url': 'https://www.cbn.gov.ng/out/2013/ccd/chapter%20five.pdf', 'in_bank_top8': False,
          'quote': 'The Federation Account revenue sharing formula is as follows: FG (52.68%), SGs (26.72%) and LGs (20.60%)'},
         {'passage_id': '480c026c8564c9434a3234e70e28727b312d8755da21a2174751736a3c8fc22e',
          'url': 'https://www.linkedin.com/pulse/how-revenue-being-shared-from-federation-account-victor-ejechi',
          'in_bank_top8': False,
          'quote': 'In the current revenue allocation formula, the FG gets 52.68%, State Governments get 26.72% and the Local Governments get 20.60%'}],
     'corpus_probe': 'outputs/m11-candidates-pilot100-20261023-161557/train-2/corpus.jsonl (423 docs): 5x 20.60 hits, 8x 52.68 hits incl. UBE doc "allocates 52.68 percent to the federal government, 26.72 percent to state governments, and 20.6 percent to local governments"'},
    {'sample_id': 'train:17', 'store': 'outputs/m11-store-r14-pilot100-20261023-162716/train-17/passages.jsonl',
     'lines': 82965, 'bank_top8_passage_ids_present': 8,
     'decisive_hits': [
         {'passage_id': 'be6c1a25dd04', 'url': 'https://houstonlawreview.org/article/22273.pdf',
          'in_bank_top8': True,
          'quote': 'Madeline Heim, Yes, Police Were 20.8 More Likely to Kill than Be Killed by a Criminal in 2019, POLITIFACT (Aug. 27, 2020)'}],
     'corpus_probe': None},
    {'sample_id': 'train:20', 'store': 'outputs/m11-store-r14-pilot100-20261023-162716/train-20/passages.jsonl',
     'lines': 43100, 'bank_top8_passage_ids_present': 8,
     'decisive_hits': [
         {'passage_id': 'ad50e08df283538e9ef2d71c28d2bf4554d3138b8322c9ef9f09a5ae9e79ac38',
          'url': 'https://www.aipccenergy.com/operations-projects', 'in_bank_top8': False,
          'quote': 'A refinery 100% fully owned and developed by AIPCC. It is located in Ologbo, Ikpoba Okha LGA of Edo State.'}],
     'corpus_probe': None},
    {'sample_id': 'train:2449', 'store': 'outputs/m161-source-batch25-v1/train-2449/retrieval-attempt-001/train-2449/passages.jsonl',
     'lines': 96440, 'bank_top8_passage_ids_present': 8,
     'decisive_hits': [
         {'passage_id': None, 'url': 'https://media.law.wisc.edu/m/nmytc/telephone_law_and_rol.pdf',
          'in_bank_top8': False,
          'quote': "'Telephone Law' and the 'Rule of Law': The Russian Case (Alena V. Ledeneva, 'Telephone Justice in Russia', 24 Post-Soviet Affairs (2008))"}],
     'corpus_probe': 'outputs/m161-source-batch25-v1/train-2449/candidates-attempt-001/train-2449/corpus.jsonl (1218 docs): 22x "telephone justice" hits incl. ICJ Russia profile "A 2010 ICJ mission learned that the phenomenon of \u201ctelephone justice\u201d remained widespread: reportedly, judges are often \u2013 as a matter of routine \u2013 directly instructed by the court president how to resolve a specific case" (https://www.icj.org/cijlcountryprofiles/russian-federation/) and Cambridge "Telephone justice in the global age" chapter'},
    {'sample_id': 'train:3', 'store': 'outputs/m11-store-r14-pilot100-20261023-162716/train-3/passages.jsonl',
     'lines': 110306, 'bank_top8_passage_ids_present': None,
     'decisive_hits': [
         {'passage_id': 'd25fae756b24f9', 'url': 'https://www.nbcnews.com/politics/2020-election/fact-checking-night-2-republican-national-convention',
          'in_bank_top8': None,
          'quote': 'he has hardly proposed amnesty and free health care for all undocumented immigrants. Biden supports allowing undocumented immigrants to purchase health care with their own money; he does not support using taxpayer-funded subsidies'}],
     'corpus_probe': 'outputs/m11-candidates-pilot100-20261023-161557/train-3/corpus.jsonl (1040 docs): same NBC full text present'},
    {'sample_id': 'train:14', 'store': 'outputs/m11-store-r14-pilot100-20261023-162716/train-14/passages.jsonl',
     'lines': 82887, 'bank_top8_passage_ids_present': 8,
     'decisive_hits': [],
     'negative_findings': 'No passage links Buhari + a new Police bill + warrantless arrest: "Nigeria Police Act, 2020" mentions (punchng) concern a later PSC-amendment bill; assent passages concern disability/finance/other bills; corpus.jsonl (650 docs) likewise has no decisive link (74x generic "Police Act", 31x Buhari hits all about other bills).'},
    {'sample_id': 'train:38', 'store': 'outputs/m11-store-r14-pilot100-20261023-162716/train-38/passages.jsonl',
     'lines': 88907, 'bank_top8_passage_ids_present': 8,
     'decisive_hits': [],
     'negative_findings': 'No Kenya annual case-disposal figure: all Kenya+300,000 co-occurrences are unrelated (SIM-card penalties, forest hectares, Adventists, abortions, displacement); corpus.jsonl (610 docs) only has backlog framing: "the Kenyan Judiciary revealed that there were 553,187 cases pending, 327,928 of which were backlogged" and "the number of cases classified as back-log stood at just over 300,000".'},
]

# --- triage table (id -> class, subflags, draftability, rationale, bank quotes) ---
T = {
 'train:2': ('E2', ['RETRIEVAL_MISS'], 'requires_store_passage_inclusion',
   'Original INSUFFICIENT rested on "the only explicit 20% excerpt is historical (1976-1992)". Store probe proves the target\u2019s own store contains the current statutory formula (CBN chapter: FG 52.68/SG 26.72/LG 20.60; LinkedIn explainer: "In the current revenue allocation formula... Local Governments get 20.60%"), decisive for gold Supported and absent from the Top-8 span set. Draftable once the store passage is included; grounded, no speculation.',
   [('Between 1976 and 1992 the revenue allocation by the federal government to the local government increased from 10% to 20% of the federation account', 'the historical passage Base leaned on; the only explicit 20% excerpt per the original assessment')]),
 'train:3': ('E1', ['GOLD_TENSION'], 'not_grounded',
   'Gold Supported is not derivable from the evidence: the displayed NBC fact-check ("This is misleading") and the store\u2019s own full text ("he has hardly proposed amnesty and free health care for all undocumented immigrants; Biden supports allowing undocumented immigrants to purchase health care with their own money") support a mixed/refuted reading of the health-care element. A Supported draft would contradict the store\u2019s decisive passage.',
   [('Eric Trump says Biden wants \'amnesty and health care\' for undocumented immigrants. That\'s misleading.', 'displayed fact-check framing'),
     ('While it\'s true that Biden has pledged to stop construction of the border', 'partial-support element only')]),
 'train:8': ('E3', ['SOURCE_CIRCULARITY'], 'pending_human_review',
   'Base treated the partisan attack speech [1] (the claim\u2019s own source) as confirmation. Displayed evidence supports a Conflicting draft: Iraq vote documented and later regretted [3]; Serbia/Syria/Libya support only asserted in the claim-source; jacobin [5] characterises his record as interventionist. Grounded without speculation.',
   [('supported the 2003 invasion of Iraq, but he now calls it a \u201cmistake\u201d', 'independent assessment'),
     ('Joe Biden voted for the Iraq War, which President Trump has long called', 'claim-source passage (circularity flag)')]),
 'train:12': ('E3', [], 'pending_human_review',
   'Base cited only the recovered-economy side. Both sides are displayed: recovered ([1] nwlc, [2] medium) vs inherited-stagnation narrative ([4] Trump WH archives "secular stagnation", [6] detroitnews). Conflicting gold draftable.',
   [('Donald Trump inherited an economy that had largely recovered', 'anti-claim side'),
     ('low growth, or \u201csecular stagnation,\u201d would plague the U.S. economy indefinitely', 'pro-claim side (WH archives)')]),
 'train:14': ('E1', ['EVIDENCE_ELEMENTS_SPLIT'], 'not_grounded',
   'Claim elements (Buhari signing; new Police bill; warrantless-arrest power) never appear together: Top-8 has generic warrantless-arrest law fragments [2][7] and a 2017/2019 gunshot-treatment bill [1]; store and raw corpus contain no Buhari+Police-Act-2020+arrest link. Gold Supported cannot be grounded without speculation.',
   [('any police officer may, without an order from a magistrate and without a warrant, arrest', 'generic CPL fragment, not the named bill')]),
 'train:17': ('E2', ['SPAN_TRUNCATION', 'BIBLIOGRAPHIC_ONLY'], 'requires_store_passage_inclusion',
   'The decisive datum (Politifact title "Yes, Police Were 20.8 More Likely to Kill than Be Killed by a Criminal in 2019") sits inside a passage already retrieved for the Top-8, but the rendered span starts mid-title and cuts the number; Base therefore saw an unusable citation fragment. Store full passage (houstonlawreview.org footnotes) contains the complete title. Draftable with the store passage; note the ratio is attested bibliographically, not by primary data.',
   [('More Likely to Kill than Be Killed by a Criminal in 2019, POLITIFACT (Aug. 27, 2020)', 'Top-8 span begins mid-title; "20.8" falls before span start 110751')]),
 'train:18': ('E3', ['ONTOLOGY'], 'pending_human_review',
   'Both sides of the Conflicting reading were already displayed: the Zeglovits & Aichholzer (2014) bibliography title asserting the first-time voting boost [3], and the Wales caveat "Austria has traditionally high turnout rates... which cannot be said for elections in Wales... typically below 50 per cent" [6]. Base (NEI) failed to combine them. Store additionally holds direct comparative passages (vox: Scotland 16-year-olds "75 percent turnout rate \u2014 higher than voters three times their age"; viecer: findings "corroborate earlier evidence reported by Zeglovits and Aichholzer"). ONTOLOGY sign-off required for the three-jurisdiction scope.',
   [('Evidence for the first-time voting boost among 16- to 25-year-olds in Austria', 'supporting title in bibliography'),
     ('turnout across the country as a whole for National Assembly and local council elections are typically below 50 per cent', 'Wales caveat')]),
 'train:19': ('E3', [], 'pending_human_review',
   'Base read only the broken/stalled trackers. Kept-side evidence is displayed ([4]: "decades of academic research has actually shown the opposite: that presidents try to keep an average of about two-thirds of their campaign pledges"). Conflicting gold draftable from the mixed trackers + research framing.',
   [('33 are currently stalled and seven broken', 'broken side'),
     ('decades of academic research has actually shown the opposite', 'kept side')]),
 'train:20': ('E2', ['RETRIEVAL_MISS'], 'requires_store_passage_inclusion',
   'Top-8 is uniformly administration-PR framing ("key legacy project by the Obaseki-led administration"), with AIPCC appearing only as co-developer. The store\u2019s aipccenergy.com/operations-projects page states the refinery is "100% fully owned and developed by AIPCC" at the same Ologbo site \u2014 decisive for gold Refuted (ownership is the consortium\u2019s, not the Governor\u2019s) and absent from the Top-8 span set.',
   [('a project partly sponsored by the Governor Godwin Obaseki-led administration in Edo State and being developed by AIPCC Energy Limited', 'Top-8 framing that masked the ownership fact')]),
 'train:21': ('E3', ['ONTOLOGY'], 'pending_human_review',
   'Base (Supported) read [2]\u2019s worldwide figure ("On average, there are 15,000 inhabitants for every judge or magistrate across the globe") as settling the claim; gold NEI. A grounded NEI draft exists from the displayed metric mismatches: judges-only ratios [1] (~75 per 1,000,000) vs judges+magistrates global figure [2] vs judicial-officer ratios [5] vs public-defender "average of 17" [7] use different denominators, and no passage evidences the specific relation the claim asserts. Human sign-off required; an alternative E1 reading (gold NEI not derivable) is recorded for transparency.',
   [('On average, there are 15,000 inhabitants for every judge or magistrate across the globe', 'the figure Base leaned on'),
     ('about 75 judges 1,000,000 persons', 'Kenya judges-only ratio, different metric')]),
 'train:24': ('E3', ['ONTOLOGY'], 'pending_human_review',
   'Both sides displayed: the Bristol study\u2019s 13-14yo drop ([1][3]) vs general-population anxiety increases ([8] ONS 5.2/10 vs 3.0; [4][7]). Conflicting gold draftable; ONTOLOGY flag on whether "a study has suggested" scopes the claim to the study finding (Supported-reading) or the general assertion.',
   [('felt less anxious and more connected to school when they were away from it', 'study finding'),
     ('average anxiety scores were 5.2 out of 10, a marked increase from 3.0', 'ONS counter-side')]),
 'train:26': ('E1', ['GOLD_READING_UNSUPPORTED'], 'not_grounded',
   'Both pledge elements are directly evidenced ([1][2][3][7] "no new taxes" pledge; [4] hikes "limited to people who make more than $400,000"). The NEI gold rests on an unstated pledge-vs-enacted-incidence reading (original limitation: "Need whether the claim verifies a speech act, proposed policy or enacted tax incidence"); forcing NEI would be speculation.',
   [('he will not raise taxes for Americans making less than $400,000', 'pledge element 1'),
     ('tax hikes will be limited to people who make more than $400,000', 'pledge element 2')]),
 'train:27': ('E3', ['PROMISE_VS_PROGRESS'], 'pending_human_review',
   'Displayed evidence documents commitments, not progress: counties "made the following promises... to allocate at least 10% of national budgets to agriculture" [2]; Malabo 10% target [1][4]; monitoring frameworks [5]. Base (Supported) equated the promise with striving-to-reach; a grounded NEI draft notes no county-level allocation/progress data is displayed.',
   [('made the following promises to their people: (a) to allocate at least 10% of national budgets to agriculture', 'commitment, not progress')]),
 'train:29': ('E3', ['PRECISION_DEMAND'], 'pending_human_review',
   'Base demanded an exact 400,000 figure; [1] displays 397,243 cases filed in all Kenyan courts in FY 2013/14 \u2014 grounds the "about 400,000 per year" reading (gold Supported). Single-year caveat is the only residual limitation.',
   [('there were 397,243 cases filed in all of the courts of Kenya during FY 2013/14', 'decisive figure')]),
 'train:31': ('E3', ['ONTOLOGY'], 'pending_human_review',
   'Mirror of train:21 (same claim, Base Refuted). Grounded NEI draft from the displayed metric mismatches ([1][5][7] vs [2]); human sign-off required; alternative E1 reading recorded.',
   [('Kenya with a population of 48 million has a judge to population ratio of 1 judicial', 'Kenya ratio, judges-only metric'),
     ('well below the national global average of 17', 'unrelated public-defender average')]),
 'train:35': ('E3', ['CAUSAL_ATTRIBUTION_MISSING', 'SERIES_DIVERGENCE'], 'pending_human_review',
   'The 11.9%\u21927.3% GLSS series is displayed [8], but [4] shows a materially different ILO-modelled series (3.46% in 2019) and no passage attributes the decline to the administration\u2019s policies. NEI gold (on the causal claim) draftable; ONTOLOGY sign-off on the causal scope.',
   [('the rate of unemployment, which stood at 11.9% in 2015, dropped to 7.3% in 2019', 'GLSS series'),
     ('Ghana unemployment rate for 2019 was 3.46%', 'divergent series')]),
 'train:37': ('E3', [], 'pending_human_review',
   'The exact claimed figures are displayed verbatim in [5] ("annual average of 7%, up from 3.4% in 2016, the lowest in nearly three decades"); Base returned Conflicting by comparing unrelated indicators instead of the direct passage. Supported gold draftable.',
   [('annual average of 7%, up from 3.4% in 2016, the lowest in nearly three decades', 'decisive passage')]),
 'train:38': ('E1', ['GOLD_BASELINE_ABSENT'], 'not_grounded',
   'The store\u2019s only ~300,000 figure is a backlog count, not annual settlements ("the number of cases classified as back-log stood at just over 300,000"; "553,187 cases pending, 327,928 of which were backlogged"); no passage gives an annual disposal total. Grounding the Supported gold would require turning backlog into settlements \u2014 speculation.',
   [('Kenya\u2019s judiciary has a massive backlog of civil ca', 'Top-8 [8] backlog framing')]),
 'train:39': ('E1', ['GOLD_READING_UNSUPPORTED'], 'not_grounded',
   '[4] explicitly reports "the global average of 16" (judges and magistrates per 100,000), which grounds the claim; the original assessor already concluded the NEI label "is not entailed by denying all benchmark evidence". A NEI draft would have to dismiss displayed benchmark evidence.',
   [('significantly below the global average of 16', 'displayed global average')]),
 'train:47': ('E3', [], 'pending_human_review',
   'Claim: SA consumes more per person than anywhere else in Africa; gold Refuted. [1] displays the ranking directly ("third biggest drinking nation in Africa, and the 19th biggest... in the world") \u2014 African countries rank above SA. Base returned NEI despite the direct statement. Refuted draftable.',
   [('third biggest drinking nation in Africa, and the 19th biggest drinking nation in the world', 'decisive ranking')]),
 'train:48': ('E3', [], 'pending_human_review',
   'Same family as train:47; the Top-8 span displays the countries above SA ("the fifth highest consumption rate in the world, below Namibia (31.3 litres), Eswatini (32.7 litres), Cook Islands (32.9 litres) and Tunisia (33.4 litres)"), grounding Refuted (Namibia/Eswatini/Tunisia are African and higher). Base returned NEI.',
   [('below Namibia (31.3 litres), Eswatini (32.7 litres), Cook Islands (32.9 litres) and Tunisia (33.4 litres)', 'countries above SA')]),
 'train:49': ('E3', ['SCOPE_QUALIFIER'], 'pending_human_review',
   'Displayed figures genuinely diverge with different scopes: "31% of the population aged 15 and older" [1], "only a third (31%) of adults report consuming alcohol in the past year" [3], "the 35% of the population that do consume alcohol" [2], 33.1% [7]. A grounded NEI draft for the unqualified "Only 31%" notes the divergence/scope qualifiers; ONTOLOGY sign-off required.',
   [('an estimated 31% of the population aged 15 and older consume alcohol', 'qualified 31%'),
     ('of the 35% of the population that do consume alcohol', 'divergent figure')]),
 'train:979': ('E3', [], 'pending_human_review',
   'Both sides displayed: shoes as "a potential source of contamination" [4] / virus persistence on soles [7][8] vs "the likelihood of spreading or contracting the virus from shoes is low" [5] and NEJM-based "unlikely that enough virus would land on the floor" [6]. Base (NEI) ignored the contested possibility side; Conflicting gold draftable.',
   [('shoes can be a potential source of contamination', 'possibility side'),
     ('the likelihood of spreading or contracting the virus from shoes is low', 'low-risk side')]),
 'train:1863': ('E3', ['ONTOLOGY'], 'pending_human_review',
   'The displayed pair already contains the refutation chain: [3] carries the original "approved the creation of state and local government police" report and [1] the presidency\u2019s dismissal of that report ("The presidency has dismissed media reports claiming President Muhammadu Buhari approved..."). Base returned Conflicting instead of weighting the denial; Refuted gold draftable. ONTOLOGY flag: denial-vs-report weighting.',
   [('The presidency has dismissed media reports claiming President Muhammadu Buhari approved the establishment of State and Local Government Police', 'denial'),
     ('President Muhammadu Buhari Monday in Abuja approved the creation of state and local government police', 'the misreported claim')]),
 'train:1873': ('E1', ['PRIMARY_SOURCE_ABSENT'], 'not_grounded',
   'The claim attributes quotes to a then-Sen. Biden op-ed; the op-ed is absent from the corpus (original limitation) and the quoted phrases do not appear in any displayed span, so the Refuted gold (quote attribution wrong) cannot be grounded without speculation.',
   []),
 'train:1881': ('E1', ['NON_PROPOSITIONAL_CLAIM'], 'not_grounded',
   'The claim is a headline ("Pinning Down Prescription Drug Prices"), not a verifiable proposition; the displayed abstract supports a related thesis ("prescription drugs are priced in the United States primarily on the basis of what the market will bear") but the gold-Supported reading of the title itself cannot be grounded.',
   [('prescription drugs are priced in the United States primarily on the basis of what the market will bear', 'related thesis, not the title claim')]),
 'train:2224': ('E1', ['GOLD_READING_UNSUPPORTED'], 'not_grounded',
   'Displayed [2] supports the claim (embassy "reversed its earlier decision to ban visa applications by Kenyans without digital passport... will place a US visa in a non-digital passport until further notice"); the Refuted gold rests on an unstated temporal scope (later policy change not in evidence).',
   [('reversed its earlier decision to ban visa applications by Kenyans without digital passport', 'supports the claim, conflicts with gold')]),
 'train:2449': ('E2', ['RETRIEVAL_MISS', 'CAUSAL_LINK_UNEVIDENCED'], 'requires_store_passage_inclusion',
   'Top-8 contains zero "telephone" passages while the retrieval store holds 60 telephone-justice passages (incl. the Russia-specific "Telephone Law and the Rule of Law: The Russian Case" with Ledeneva citations) and the raw corpus holds the Russia-specific ICJ profile ("the phenomenon of \u201ctelephone justice\u201d remained widespread: judges are often... directly instructed by the court president"). The telephone-justice existence premise is recoverable from the store; the causal "because" element remains unevidenced, supporting a Conflicting draft (prison-rate comparability conflicts displayed in Top-8 [1][2][7]).',
   [('prison population is substantially inflated by the presence of drug smugglers who are nationals of other countries', 'Top-8 comparability conflict (UNODC)')]),
}

CLS_ORDER = ['E1', 'E2', 'E3']
records = []
for sid in sorted(T, key=lambda s: int(s.split(':')[1])):
    cls, subflags, draftability, rationale, quotes = T[sid]
    b = B[sid]
    m = M[sid]
    assert b['is_base_error'], sid
    da = m.get('draft_assessment') or {}
    dec = [dict(find_quote(sid, q), note=note) for q, note in quotes]
    rec = {
        'sample_id': sid, 'gold_verdict': b['gold_verdict'],
        'base_verdict': b['base_prediction']['stored_verdict'], 'claim': b['claim'],
        'triage_class': cls, 'subflags': subflags, 'draftability': draftability,
        'rationale': rationale, 'decisive_evidence': dec,
        'original_draft_assessment': {k: da.get(k) for k in ('status', 'reason', 'limitations')},
    }
    sp = [p for p in STORE_PROBES if p['sample_id'] == sid]
    if sp:
        rec['store_probe'] = sp[0]
    records.append(rec)

by_class = {c: sum(1 for r in records if r['triage_class'] == c) for c in CLS_ORDER}
by_gold = {}
for r in records:
    by_gold.setdefault(r['gold_verdict'], {}).setdefault(r['triage_class'], 0)
    by_gold[r['gold_verdict']][r['triage_class']] += 1
by_draftability = {}
for r in records:
    by_draftability[r['draftability']] = by_draftability.get(r['draftability'], 0) + 1

out = {
    'schema_version': 'm162d-triage-v1',
    'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'milestone': 'M1.6.2d (internal M1.6.3a: Error Information Feasibility) \u2014 step 1',
    'scope': ('CPU-only evidence-adequacy triage of the 28 assistant-INSUFFICIENT error records of the '
              'memory75 bank. EXPLORATORY, not confirmatory; no M1.6.3 activation; no gold of any '
              'DEVELOPMENT/TEST target read; SOURCE bank labels are the records\u2019 own gold fields; '
              'no prompt tuning, no model changes.'),
    'class_definitions': {
        'E1': 'Evidence for the gold verdict genuinely absent from the displayed Top-8 AND from the target\u2019s per-claim store/corpus, or the gold reading is not derivable from displayed evidence without speculation \u2014 correction draft not grounded.',
        'E2': 'Decisive evidence present in the target\u2019s own store/corpus (or in a retrieved passage) but missed by extraction/spanning or by retrieval ranking \u2014 draftable once the store passage is included.',
        'E3': 'Decisive (or both-sides) evidence already displayed in the Top-8; Base reasoning/ontology error \u2014 draftable now, pending human review.',
    },
    'inputs': {
        'bank': {'path': str(BANK_PATH), 'sha256': hashlib.sha256(BANK_PATH.read_bytes()).hexdigest()},
        'memory': {'path': str(MEM_PATH), 'sha256': hashlib.sha256(MEM_PATH.read_bytes()).hexdigest()},
    },
    'store_probes_note': ('Server-side greps run 2026-10-28 on hvtham-server against per-claim stores '
                          '(bank Top-8 passage_id membership verified 8/8 where checked; train:3 store '
                          'identity inferred from same pilot100 batch). Store-level quotes were verified '
                          'in those probes and are reproduced here; they are not re-assertable locally.'),
    'summary': {
        'records': len(records),
        'by_class': by_class,
        'by_gold_x_class': by_gold,
        'by_draftability': by_draftability,
    },
    'records': records,
}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print('records:', len(records))
print('by_class:', by_class)
print('by_draftability:', by_draftability)
print('sha256:', hashlib.sha256(OUT_PATH.read_bytes()).hexdigest())
