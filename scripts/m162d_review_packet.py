"""M1.6.2d step-2: generate docs/M1_6_2D_REVIEW_PACKET.md from the immutable bank.

CPU-only, read-only. Renders the 5 existing correction drafts verbatim from
combined75.json plus the triage-derived expansion/candidate sections. No gold of
any DEVELOPMENT/TEST target is read; SOURCE bank labels are the records' own gold.
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
DRAFTS = ['train:6', 'train:11', 'train:25', 'train:978', 'train:1824']


def host(url):
    try:
        return url.split('/')[2]
    except Exception:
        return '-'


L = []
L.append('# M1.6.2d — Step 2 Human Correction Review Packet')
L.append('')
L.append('Generated: %s | Scope: EXPLORATORY (internal M1.6.3a Error Information Feasibility; '
         'the confirmatory M1.6.3 remains NO-GO).' % datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%MZ'))
L.append('')
L.append('## Review rules (locked)')
L.append('')
L.append('1. **Grounded expansion only.** A correction may be expanded/accepted only where its support '
         'quotes verbatim from displayed evidence (Top-8) or, for E2 candidates, from the identified '
         'per-claim store passage. Never fabricate or extrapolate to reach the ≥30-verified gate; '
         'an unmet gate is reported as unmet.')
L.append('2. **Drafting discipline preserved.** Do not consult raw-train justifications or any gold QA/'
         'justification material for SOURCE records; drafts stand on evidence only (this matches the '
         'M1.6.1 discipline).')
L.append('3. **Ontology.** Verdicts use the AVeriTeC four-class ontology; records flagged `ONTOLOGY` '
         'require an explicit human sign-off line in the review.')
L.append('4. **Verdicts per draft:** `ACCEPT` / `ACCEPT_WITH_EDITS` (with edited text) / `REJECT` '
         '(with reason). Per-review bookkeeping increments the verified-error count; current status '
         '**0 verified / 33 error corrections in scope; gate ≥30 unmet**.')
L.append('5. No DEVELOPMENT50 outcomes may be used to tune exemplar selection or prompts; no further '
         'DEV50 prompt sweeps; results remain EXPLORATORY.')
L.append('')
L.append('## Part A — Existing correction drafts (5, verbatim from bank)')
L.append('')
for sid in DRAFTS:
    b = B[sid]
    m = M[sid]
    da = m.get('draft_assessment') or {}
    L.append('### %s (gold=%s, base=%s)' % (sid, b['gold_verdict'], b['base_prediction']['stored_verdict']))
    L.append('')
    L.append('- **Claim:** %s' % b['claim'])
    L.append('- **Draft status (bank metadata):** %s' % (m.get('review_status') or da.get('status', 'n/a')))
    L.append('')
    L.append('**Draft text (verbatim, from memory record):**')
    L.append('')
    L.append('> ' + (m.get('corrected_explanation') or '(empty)').replace('\n', '\n> '))
    L.append('')
    L.append('**Evidence inventory (Top-8 rendered spans):**')
    L.append('')
    for e in b['evidence']:
        head = ' '.join((e.get('rendered_text') or '').split())[:240]
        L.append('- [%d] %s — %s' % (e['rank'], host(e.get('source_url') or ''), head))
    L.append('')
L.append('## Part B — Draftable expansion candidates (from step-1 triage)')
L.append('')
L.append('These are **candidates**, not drafts. Expansion proceeds only after human go, and only with '
         'the listed grounded evidence. E2 rows additionally require including the identified store '
         'passage before drafting.')
L.append('')
L.append('| ID | gold | base | class | subflags | decisive grounded evidence |')
L.append('|---|---|---|---|---|---|')
for sid in sorted(T, key=lambda s: int(s.split(':')[1])):
    t = T[sid]
    if t['triage_class'] == 'E1':
        continue
    ev = t['decisive_evidence']
    desc = []
    for d in ev[:2]:
        src = 'store' if d['source'] == 'store_probe' else 'Top-8[%s]' % d.get('rank')
        desc.append('%s: "%s…"' % (src, (d['quote'] or '')[:90]))
    L.append('| %s | %s | %s | %s | %s | %s |' % (
        sid, t['gold_verdict'], t['base_verdict'], t['triage_class'],
        ', '.join(t['subflags']) or '—', ' <br>'.join(desc)))
L.append('')
L.append('ONTOLOGY sign-off required for: %s.' % ', '.join(
    sorted(sid for sid, t in T.items() if 'ONTOLOGY' in t['subflags'])))
L.append('')
L.append('## Part C — Not draftable (E1, no action)')
L.append('')
for sid in sorted(T, key=lambda s: int(s.split(':')[1])):
    t = T[sid]
    if t['triage_class'] != 'E1':
        continue
    L.append('- **%s** (gold=%s, base=%s; %s): %s' % (
        sid, t['gold_verdict'], t['base_verdict'], ', '.join(t['subflags']) or '—', t['rationale']))
L.append('')
L.append('## Provenance')
L.append('')
for p in ('outputs/m161-review-preparation-v1/combined75.json',
          'outputs/m1611-correction-audit-v3/memory75.json',
          'docs/audits/M1_6_2D_EVIDENCE_TRIAGE.json'):
    L.append('- `%s` sha256 `%s`' % (p, hashlib.sha256(Path(p).read_bytes()).hexdigest()))
L.append('')

dest = Path('docs/M1_6_2D_REVIEW_PACKET.md')
dest.write_text('\n'.join(L), encoding='utf-8')
print('packet bytes:', dest.stat().st_size)
