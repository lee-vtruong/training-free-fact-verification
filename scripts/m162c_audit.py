"""M1.6.2c: CPU-only SOURCE coverage audit and M1.6.2b pilot failure analysis.

No retrieval, no inference, and no active learning; reads sealed local
artifacts only and writes one machine-readable audit JSON. Section-B analysis
is descriptive; gold of the exposed DEVELOPMENT50 is only used read-only for
confusion/A-B interpretation, never for demonstration selection.
"""

import argparse, collections, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_SHA256 = 'ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c'
CLASSES = ('Supported', 'Not Enough Evidence', 'Refuted',
           'Conflicting Evidence/Cherrypicking')
CANONICAL_ARMS = ('P0', 'P1', 'P2', 'P3', 'N0')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines()
            if s.strip()]


def evidence_ok(record):
    """Gold-only block usability: guarded evidence context plus citable URL on
    each evidence item (the record-level citations field is an optional extra)."""
    items = record.get('evidence') or []
    return bool(items) and all(i.get('source_url') and i.get('rendered_text') for i in items)


def gold_class_counts(partitions_dir):
    return read(partitions_dir / 'development_manifest.json')


def source_coverage(bank_path, partitions_dir):
    """A: 75-case gold x base-history coverage with usability and review facts."""
    bank = read(bank_path)
    records = bank['records']
    dev_groups = {e['group_id'] for e in gold_class_counts(partitions_dir)}
    crosstab = {'x_' + gold + ('__wrong' if wrong else '__correct'): 0
                for gold in CLASSES for wrong in (False, True)}
    by_class = {gold: {'correct_ids': [], 'wrong_ids': [], 'usable_correct': [],
                       'usable_wrong': []} for gold in CLASSES}
    block_sizes = collections.Counter()
    zero_evidence = []
    for r in records:
        gold = r['gold_verdict']
        history = 'wrong' if r['is_base_error'] else 'correct'
        bucket = by_class[gold]
        bucket['%s_ids' % history].append(r['sample_id'])
        crosstab['x_' + gold + '__' + history] += 1
        block_sizes[r['group_id']] += 1
        if not r.get('evidence'):
            zero_evidence.append(r['sample_id'])
        if evidence_ok(r):
            bucket['usable_%s' % history].append(r['sample_id'])
    return {
        'case_count': len(records),
        'crosstab_gold_x_base_history': crosstab,
        'by_class': {g: {k: sorted(v) for k, v in b.items()} for g, b in by_class.items()},
        'records_with_zero_evidence': sorted(zero_evidence),
        'records_with_zero_citations': sorted(r['sample_id'] for r in records
                                              if not (r.get('citations') or [])),
        'review_statuses': dict(collections.Counter(r['review_status'] for r in records)),
        'records_with_assistant_candidate_text':
            sorted(r['sample_id'] for r in records if r.get('corrected_explanation')),
        'records_with_any_human_review': sorted(r['sample_id'] for r in records if r.get('reviews')),
        'distinct_leakage_groups': len({r['group_id'] for r in records}),
        'bank_groups_shared_by_multiple_records':
            {g: n for g, n in sorted(block_sizes.items()) if n > 1},
        'development_group_overlap': sorted({r['group_id'] for r in records} & dev_groups),
        'usability_definition': 'SOURCE-admitted record with >=1 guarded evidence item '
                                '(nonempty article text + source URL), >=1 citation URL, known gold',
    }


def load_arm_verdicts(canonical_dir, mixed_dir, sample_ids):
    """Exact verdicts and prompt-token means per arm from immutable receipts."""
    verdicts = {a: {} for a in CANONICAL_ARMS + ('MIXED_GOLD_ONLY_K2',)}
    prompt_tokens = {a: [] for a in verdicts}
    for sid in sample_ids:
        cdir = Path(canonical_dir) / sid.replace(':', '-')
        for row in jsonl(cdir / 'predictions-attempt-001' / 'predictions.jsonl'):
            if row['condition'] not in verdicts or row['sample_id'] != sid:
                raise SystemExit('Canonical receipt identity mismatch: ' + sid)
            verdicts[row['condition']][sid] = row['predicted_verdict']
            prompt_tokens[row['condition']].append(row['prompt_tokens'])
        attempts = sorted((Path(mixed_dir) / sid.replace(':', '-'))
                          .glob('prediction-attempt-*/prediction.json'))
        if len(attempts) != 1:
            raise SystemExit('Expected exactly one mixed attempt: ' + sid)
        row = read(attempts[-1])
        if row['sample_id'] != sid or row['condition'] != 'MIXED_GOLD_ONLY_K2':
            raise SystemExit('Mixed receipt identity mismatch: ' + sid)
        verdicts['MIXED_GOLD_ONLY_K2'][sid] = row['predicted_verdict']
        prompt_tokens['MIXED_GOLD_ONLY_K2'].append(row['prompt_tokens'])
    return verdicts, {a: sum(v) / len(v) for a, v in prompt_tokens.items()}


def confusion(gold, pred):
    matrix = {g: {p: 0 for p in CLASSES + ('Unparseable',)} for g in CLASSES}
    for g, p in zip(gold, pred):
        matrix[g][p] += 1
    return matrix


def accuracy(arm, gold_by_sid):
    return sum(1 for sid in arm if arm[sid] == gold_by_sid[sid]) / max(1, len(arm))


def pilot_failure(canonical_dir, mixed_dir, batch_path, train_path, bank_path,
                  partitions_dir):
    """B: exact-verdict Mixed-vs-N0 comparison, priors and composition breaks."""
    batch = read(batch_path)
    sample_ids = batch['sample_ids']
    raw = read(train_path)
    gold_by_sid = {sid: raw[int(sid.split(':')[1])]['label'] for sid in sample_ids}
    verdicts, token_means = load_arm_verdicts(canonical_dir, mixed_dir, sample_ids)
    bank_by_sid = {r['sample_id']: r for r in read(bank_path)['records']}
    compositions = {'supported_pair': [], 'refuted_pair': []}
    for slate in batch['slates']:
        gold = bank_by_sid[slate['correct_id']]['gold_verdict']
        if gold != bank_by_sid[slate['error_id']]['gold_verdict']:
            raise SystemExit('Pair shares source gold violated: ' + slate['sample_id'])
        compositions['supported_pair' if gold == 'Supported' else 'refuted_pair'].append(slate['sample_id'])
    corrections = sorted(sid for sid in sample_ids
                         if verdicts['MIXED_GOLD_ONLY_K2'][sid] == gold_by_sid[sid] and
                         verdicts['P2'][sid] != gold_by_sid[sid])
    regressions = sorted(sid for sid in sample_ids
                         if verdicts['MIXED_GOLD_ONLY_K2'][sid] != gold_by_sid[sid] and
                         verdicts['P2'][sid] == gold_by_sid[sid])
    exact_diffs = []
    for sid in sample_ids:
        m, n0 = verdicts['MIXED_GOLD_ONLY_K2'][sid], verdicts['N0'][sid]
        if m != n0:
            exact_diffs.append({'sample_id': sid, 'mixed': m, 'n0': n0,
                                'gold': gold_by_sid[sid],
                                'both_wrong_other_class': (m != gold_by_sid[sid]) and
                                                          (n0 != gold_by_sid[sid])})
    pair_gold = {sid: bank_by_sid[slate['error_id']]['gold_verdict']
                 for sid, slate in {s['sample_id']: s for s in batch['slates']}.items()}
    composition_breakdown = {}
    for gold_class, members in compositions.items():
        composition_breakdown[gold_class] = {
            'targets': len(members),
            'mixed_accuracy': accuracy({s: verdicts['MIXED_GOLD_ONLY_K2'][s] for s in members},
                                       {s: gold_by_sid[s]}if False else {s: gold_by_sid[s] for s in members}),
            'n0_accuracy': accuracy({s: verdicts['N0'][s] for s in members},
                                    {s: gold_by_sid[s] for s in members}),
            'mixed_member_predictions_sample': members[:6],
        }
    return {
        'arm_accuracy': {a: accuracy(verdicts[a], gold_by_sid)
                         for a in ('P2', 'N0', 'MIXED_GOLD_ONLY_K2')},
        'mean_prompt_tokens': token_means,
        'pair_gold_composition': compositions,
        'composition_breakdown': composition_breakdown,
        'targets_nei_conflicting_against_pair_gold':
            {sid: pair_gold[sid] for sid in sample_ids
             if gold_by_sid[sid] not in ('Supported', 'Refuted')},
        'confusion_matrices': {a: confusion([gold_by_sid[s] for s in sample_ids],
                                            [verdicts[a][s] for s in sample_ids])
                               for a in ('P2', 'N0', 'MIXED_GOLD_ONLY_K2')},
        'predicted_verdict_priors':
            {a: dict(sorted(collections.Counter(verdicts[a][s] for s in sample_ids).items()))
             for a in ('P2', 'N0', 'MIXED_GOLD_ONLY_K2')},
        'mixed_vs_n0_exact_verdict_differences': exact_diffs,
        'mixed_vs_p2_correction_ids': corrections,
        'mixed_vs_p2_regression_ids': regressions,
        'note': 'exact verdicts read from immutable receipts; descriptive only, '
                'no causal conclusion, no tuning',
    }


def source_pool_gold_counts(partitions_dir, train_path):
    """Branch-2 planning parameter: gold-class availability among admitted
    SOURCE-partition claims, for potential expansion batches. Labels are used
    for availability planning only; they are never shown to any model before a
    separately reviewed memory-batch pipeline and never used to select DEV/TEST."""
    manifest = read(Path(partitions_dir) / 'source_manifest.json')
    raw = read(train_path)
    pool = [e['sample_id'] for e in manifest if e['source_memory_eligible']]
    counts = collections.Counter(raw[int(s.split(':')[1])]['label'] for s in pool)
    return {'total_source_claims': len(manifest), 'memory_eligible_claims': len(pool),
            'memory_eligible_gold': dict(sorted(counts.items())),
            'ineligible_claims_without_known_gold': len(pool) - sum(counts.values())}


def correction_status(bank_path):
    """D: 33 historical errors; drafts and human-review progress."""
    records = read(bank_path)['records']
    errors = [r for r in records if r['is_base_error']]
    return {
        'historical_errors': len(errors),
        'pending_error_review_ids': sorted(r['sample_id'] for r in errors
                                           if r['review_status'] == 'PENDING_REVIEW'),
        'errors_with_assistant_draft_text': sorted(r['sample_id'] for r in errors
                                                   if r.get('corrected_explanation')),
        'assistant_insufficient_error_ids': sorted(r['sample_id'] for r in errors
                                                   if r['review_status'] == 'INSUFFICIENT_EVIDENCE'),
        'records_with_any_human_review': sorted(r['sample_id'] for r in records
                                                if r.get('reviews')),
        'verified_rationales': sorted(r['sample_id'] for r in records
                                      if r['review_status'] == 'VERIFIED'),
        'note': 'assistant drafts are never human verification; full suite of 42 '
                'correct controls also remains structurally audited only',
    }


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--source-bank', type=Path,
                   default=ROOT / 'outputs/m161-review-preparation-v1/combined75.json')
    a.add_argument('--correction-bank', type=Path,
                   default=ROOT / 'outputs/m1611-correction-audit-v3/memory75.json',
                   help='post-replay bank; authoritative for assistant dispositions')
    a.add_argument('--canonical-dir', type=Path,
                   default=ROOT / 'outputs/m162c-analysis-v1/outputs/m162a-dev50-v1')
    a.add_argument('--mixed-dir', type=Path,
                   default=ROOT / 'outputs/m162c-analysis-v1/outputs/m162b-mixed-v1')
    a.add_argument('--batch', type=Path,
                   default=ROOT / 'outputs/m162a-dev50-preparation-v4/batch.json')
    a.add_argument('--train', type=Path, default=ROOT / 'data/raw/averitec/train.json')
    a.add_argument('--partitions', type=Path, default=ROOT / 'docs/partitions/m16-v2')
    a.add_argument('--output', type=Path, default=ROOT / 'docs/audits/M1_6_2C_AUDIT.json')
    args = a.parse_args()
    import hashlib
    digest = hashlib.sha256(Path(args.train).read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise SystemExit('Raw train.json SHA mismatch: ' + digest)
    result = {
        'schema_version': 'm162c-audit-v1',
        'scope': 'CPU_ONLY_AUDIT_NO_INFERENCE',
        'raw_train_sha256': digest,
        'source_coverage': source_coverage(args.source_bank, args.partitions),
        'pilot_failure': pilot_failure(args.canonical_dir, args.mixed_dir, args.batch,
                                       args.train, args.source_bank, args.partitions),
        'correction_status': correction_status(args.correction_bank),
        'branch2_source_pool_gold_counts': source_pool_gold_counts(args.partitions,
                                                                  args.train),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True,
                                            ensure_ascii=False) + '\n',
                                 encoding='utf-8')
    print('written', args.output)


if __name__ == '__main__':
    main()
