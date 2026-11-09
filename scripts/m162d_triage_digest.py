"""M1.6.2d step-1 digest: dump the 33 historical error records for evidence-adequacy triage.

CPU-only, read-only over existing immutable artifacts. Deterministic; no gold of any
DEVELOPMENT/TEST target is read; SOURCE bank labels are the records' own gold fields.
"""
import json
from pathlib import Path

BANK = json.loads(Path('outputs/m161-review-preparation-v1/combined75.json').read_text('utf-8-sig'))
MEM = json.loads(Path('outputs/m1611-correction-audit-v3/memory75.json').read_text('utf-8-sig'))
BANK_R = {r['sample_id']: r for r in BANK['records']}
MEM_R = {r['sample_id']: r for r in MEM['records']}
INSUFFICIENT = ['train:2', 'train:3', 'train:8', 'train:12', 'train:14', 'train:17', 'train:18',
                'train:19', 'train:20', 'train:21', 'train:24', 'train:26', 'train:27', 'train:29',
                'train:31', 'train:35', 'train:37', 'train:38', 'train:39', 'train:47', 'train:48',
                'train:49', 'train:1873', 'train:1863', 'train:2224', 'train:979', 'train:2449',
                'train:1881']
DRAFTS = ['train:6', 'train:11', 'train:25', 'train:978', 'train:1824']

out = []
for group, ids in (('INSUFFICIENT', INSUFFICIENT), ('DRAFT', DRAFTS)):
    for sid in sorted(ids, key=lambda s: int(s.split(':')[1])):
        b, m = BANK_R[sid], MEM_R[sid]
        assert b['gold_verdict'] == m['gold_verdict'] and b['is_base_error'], sid
        assert b['claim'] == m['claim'], sid
        assert b['gold_verdict'] == m['gold_verdict'] and b['is_base_error'], sid
        da = m.get('draft_assessment') or {}
        rows = []
        for e in b['evidence']:
            head = ' '.join((e.get('rendered_text') or '').split())[:210]
            rows.append({'rank': e['rank'], 'url': e.get('source_url', ''),
                         'span': [e.get('start_char'), e.get('end_char')], 'head': head})
        out.append({
            'group': group, 'sample_id': sid, 'gold': b['gold_verdict'],
            'base_verdict': b['base_prediction']['stored_verdict'],
            'claim': b['claim'],
            'base_reasoning_head': ' '.join(
                (b['base_prediction'].get('raw_generation') or '').split())[:420],
            'candidate_mechanisms': b.get('candidate_mechanisms'),
            'draft_status': da.get('status'), 'draft_reason': da.get('reason'),
            'draft_limitations': da.get('limitations'),
            'draft_text': b.get('corrected_explanation'),
            'evidence': rows,
        })

dest = Path('temp_m162d/triage_digest.json')
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print('records:', len(out), '| bytes:', dest.stat().st_size)
print('INSUFFICIENT by gold:', {g: sum(1 for o in out if o['group'] == 'INSUFFICIENT' and o['gold'] == g)
      for g in sorted({o['gold'] for o in out})})
