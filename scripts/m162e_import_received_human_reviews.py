"""Import the two user-submitted human-review packets for five original drafts.

This is bookkeeping, not an automated review: decision text was supplied by the
user in two distinct review submissions.  The resulting qualified records remain
exploratory SOURCE memory; the Tier-1 mechanism-coverage gate still applies.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/audits/M1_6_2E_CANDIDATE_DRAFTS.json'
DEST = ROOT / 'docs/audits/M1_6_2E_REVIEWED_RECORDS.json'

NOW = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

# The IDs are opaque provenance labels, deliberately not invented personal names.
# User attested that these are the two human reviews; their distinct submission
# records are preserved below.
R1 = {
    'reviewer_id': 'human-review-1:user-submission',
    'reviewer_type': 'human', 'independent_review': True,
    'reviewed_at': NOW, 'submission_ref': 'chat review table + revised corrections',
}
R2 = {
    'reviewer_id': 'human-review-2:user-submission',
    'reviewer_type': 'human', 'independent_review': True,
    'reviewed_at': NOW, 'submission_ref': 'attached C1-C5 review packet',
}

EDITS_1 = {
 'train:6': '''The Base returned Not Enough Evidence despite a displayed passage specifying that Nigerian state governments receive 26.72% of the Federation Account allocation. Under this statutory allocation context, the claim that 35% of revenue goes to states is contradicted by the documented state share. Another passage assigns 35% to local governments under a VAT distribution arrangement, which concerns a different recipient and revenue basket and must not be treated as support for the claim. The correction therefore supports Refuted specifically under the Federation Account interpretation. Because the claim does not explicitly specify the revenue basket or reference year, this scope qualification must be preserved rather than presenting 26.72% as a universal share of all Nigerian government revenue.''',
 'train:11': '''The Base returned Not Enough Evidence even though a displayed source reports a federal government share of 52.68% under Nigeria's statutory revenue-allocation formula, rather than the claimed 45%. Other displayed percentages refer to different revenue streams, including VAT distributions, and are not directly interchangeable with the Federation Account allocation. Under the statutory allocation interpretation, 52.68% contradicts the claimed 45%, supporting the Refuted verdict. However, the phrase "Nigeria's revenue" does not explicitly identify the revenue basket, accounting period, or denominator. The correction must therefore be understood as conditional on the statutory Federation Account interpretation, not as proof that the federal government receives 52.68% of every category of Nigerian public revenue.''',
 'train:978': '''The Base focused on the weekday curfew interval of 17:00–05:00, which lasts 12 hours, and therefore failed to identify the longer weekend restriction. The displayed Serbian government notice describes a weekend movement ban running from 15:00 to 05:00 the following morning, a continuous interval of 14 hours. The accompanying reporting states that the previously permitted 20-minute exception for dog walking was withdrawn during the stricter restrictions. When these measures apply to the same restriction period, dog owners could not walk their dogs for up to 14 consecutive hours, supporting the claim. The correction concerns this specific weekend measure and the withdrawal of the dog-walking exception; it does not imply that every Serbian curfew lasted 14 hours.''',
 'train:1824': '''The Base treated repeated attribution of the quotation to Winston Churchill across general quotation pages as evidence of competing accounts. However, the displayed fact-checking material refers to the International Churchill Society's collection of quotations falsely attributed to Churchill and reports that the statement has not been located in Churchill's documented writings or interviews. The repeated attribution by secondary quotation websites does not establish authorship and is not equivalent to independent primary-source confirmation. Under the displayed attribution-checking evidence, the claim that Churchill said these exact words is therefore better classified as Refuted than as Conflicting Evidence/Cherrypicking. The conclusion rests on the documented lack of authenticated attribution and the specialist correction, not on independently proving that Churchill could never have spoken similar words.''',
}

META = {
 'train:6': ('revenue_basket_recipient_disambiguation', 'ACCEPT_WITH_EDITS',
             '26.72% differs from 35%, but scope, revenue type, and timing require qualification.'),
 'train:11': ('revenue_basket_disambiguation', 'ACCEPT_WITH_EDITS',
              '52.68% supports the statutory-formula reading; Nigeria revenue needs a denominator/basket qualifier.'),
 'train:25': ('negation_polarity', 'ACCEPT',
              'Sub-1% figures directly support “does not even get 1%”.'),
 'train:978': ('temporal_interval_integration', 'ACCEPT_WITH_EDITS',
               '14-hour interval is arithmetically correct; contemporaneity was subsequently checked in Top-8.'),
 'train:1824': ('source_authority_attribution', 'ACCEPT_WITH_EDITS',
                'Attribution correction needs the specialist-source limitation stated precisely.'),
}

R2_REASON = {
 'train:6': 'C1/C3/C4/C5 partial: preserves the Federation Account condition and flags the unqualified-denominator ontology issue.',
 'train:11': 'C1–C5 partial: numeric mismatch is conditional on the statutory allocation; alternative revenue baskets remain material.',
 'train:25': 'C1/C2 pass, remaining scope partial: reported sub-1% values support only their stated fiscal periods.',
 'train:978': 'Provisional C1–C5 partial: weekend duration is correct; linkage to pet-walk withdrawal required Top-8 verification.',
 'train:1824': 'C2/C5 pass, other criteria partial: conclusion depends on the specialist material applying to this exact quotation.',
}

payload = json.loads(SOURCE.read_text(encoding='utf-8-sig'))
for record in payload['records']:
    sid = record['sample_id']
    if sid not in META:
        continue
    mechanism, r1_decision, r1_reason = META[sid]
    r1 = dict(R1, decision=r1_decision, rationale=r1_reason,
              ontology_signoff='QUALIFIED: use the stated scope limitation.')
    if sid in EDITS_1:
        r1['edited_text'] = EDITS_1[sid]
    r2 = dict(R2, decision='ACCEPT_WITH_EDITS', rationale=R2_REASON[sid],
              ontology_signoff='QUALIFIED: see Top-8 completion audit for resolved checks and residual scope.')
    record['mechanism'] = mechanism
    record['reviews'] = {'reviewer_1': r1, 'reviewer_2': r2}
    record['review_status'] = 'VERIFIED'
    record['verified'] = True
    record['verification_scope'] = ('Qualified human-reviewed correction: source evidence is limited to the '
                                    'rendered Top-8 and the stated scope; not a universal factual claim.')
    record['review_import_provenance'] = ('Two human reviews supplied by the user in separate submissions; '
                                          'imported with opaque source labels, not fabricated personal identities.')

payload['review_import'] = {
    'imported_at': NOW,
    'source': 'two human-review submissions provided in conversation',
    'imported_ids': sorted(META, key=lambda x: int(x.split(':')[1])),
    'verified_count': 5,
    'note': 'No assistant decision was created; all five records retain qualified limitations.'
}
DEST.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(DEST)
