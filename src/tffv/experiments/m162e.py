"""M1.6.2e: verified error-memory feasibility selection (CPU-only, no inference).

Pure selection/protocol functions for the three-arm (+length-matched control D)
feasibility study defined in docs/M1_6_2E_PROTOCOL.md and locked in
configs/m162e_feasibility_protocol.json. No GPU, no retrieval, no model calls.

Fail-closed by construction: the exemplar pool is derived at runtime from
two-reviewer review records and is EMPTY unless both reviewers accepted a record
(decision in ACCEPT/ACCEPT_WITH_EDITS, verified flag set by human bookkeeping).
Assistant drafts are never counted as verified; E2-quarantined records never
enter the primary reasoning-error pool. A non-empty pool may still fail closed
at the Tier-1 two-exemplars-per-mechanism coverage gate.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / 'configs' / 'm162e_feasibility_protocol.json'
RECORDS_PATH = ROOT / 'docs' / 'audits' / 'M1_6_2E_REVIEWED_RECORDS.json'
REVIEW_SHEETS = (ROOT / 'docs' / 'audits' / 'M1_6_2E_REVIEW_RECORDS_REVIEWER1.json',
                 ROOT / 'docs' / 'audits' / 'M1_6_2E_REVIEW_RECORDS_REVIEWER2.json')

ARMS = ('A', 'B', 'C', 'D')
ARM_NAMES = {'A': 'gold_only', 'B': 'gold_verified',
             'C': 'wrong_gold_verified', 'D': 'gold_filler_length_matched'}
ELIGIBLE_KINDS = ('ORIGINAL_DRAFT', 'E3_CANDIDATE_DRAFT')
ACCEPT_DECISIONS = ('ACCEPT', 'ACCEPT_WITH_EDITS')
LABEL_MENU = ('Verdict options (choose exactly one): Supported | Refuted | '
              'Not Enough Evidence | Conflicting Evidence/Cherrypicking')

# Neutral placebo bank for arm D: procedure/format statements with no
# verdict-relevant content (no verdict labels, no mechanism words).
FILLER_BANK = (
    'This reference is displayed only to illustrate the required response format.',
    'The response should state one chosen option from the listed menu.',
    'Historical references are independent of the target claim.',
    'The displayed evidence block is fixed and identical for every condition.',
    'The response format matches the examples shown in this session.',
    'No additional reference material will be introduced after this point.',
)


def load_config(path=None):
    cfg = json.loads(Path(path or CONFIG_PATH).read_text('utf-8-sig'))
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    """Locked shape; any drift raises so a silent protocol edit cannot pass."""
    if cfg.get('version') != 'm162e-feasibility-v1':
        raise ValueError('protocol version changed without re-registration')
    g = cfg['gate']
    if (g['full_memory_gate_verified'], g['max_plausible_candidates_pre_rejection'],
            g['unchanged_top8_ceiling'], g['feasibility_min_verified'],
            g['feasibility_min_classes'], g['feasibility_min_per_class']) != (30, 25, 21, 4, 2, 2):
        raise ValueError('gate arithmetic drifted from the locked M1.6.2e policy')
    if not (g['e2_excluded_from_primary_experiment'] and
            g['verification_source'].startswith('human review bookkeeping')):
        raise ValueError('E2 quarantine / human-only verification must stay locked')
    d = cfg['decoding']
    if (d['temperature'], d['max_new_tokens'], d['greedy']) != (0.0, 512, True):
        raise ValueError('decoding must stay frozen at temperature 0.0 / 512 tokens / greedy')
    t = cfg['target_selection']
    if (t['n_per_class'], t['min_total_targets']) != (6, 12):
        raise ValueError('target selection counts drifted from the locked plan')
    if t['ordering'] != 'count-then-ID: per covered class, ascending sample_id ordinal':
        raise ValueError('target ordering drifted from the locked count-then-ID rule')
    s = cfg['stats']
    if s['contrasts'] != ['A_vs_B', 'A_vs_C', 'B_vs_D'] or s['test'] != 'exact_mcnemar_two_sided':
        raise ValueError('stats contrasts drifted from the locked plan')
    e = cfg['environment_lock']
    if (e['gpu_executions_in_m162e'], e['new_retrieval_in_m162e'],
            e['test_sealed_access'], e['m163_activation'],
            e['prompt_sweeps'], e['development50_tuning']) != (0, 0, False, False, False, False):
        raise ValueError('environment locks drifted from the M1.6.2e scope')
    return True


def load_review_records(path=None):
    return json.loads(Path(path or RECORDS_PATH).read_text('utf-8-sig'))['records']


def _decision_ok(slot):
    return isinstance(slot, dict) and slot.get('decision') in ACCEPT_DECISIONS


def effective_draft(record):
    """Text an arm would display: reviewer-accepted edits replace the draft."""
    for who in ('reviewer_1', 'reviewer_2'):
        slot = record.get('reviews', {}).get(who)
        if _decision_ok(slot) and slot.get('edited_text'):
            return slot['edited_text']
    return record.get('draft_text')


def verified_pool(records=None):
    """Human-verified, primary-experiment-eligible pool. Empty today (fail-closed).

    A record qualifies only when: eligible kind (original or E3 candidate),
    verified flag set by human bookkeeping, BOTH reviewer decisions present and
    accepting, not excluded from the primary experiment, and no
    evidence-availability change (E2 stays quarantined per C.6).
    """
    records = load_review_records() if records is None else records
    pool = []
    for r in records:
        if r.get('record_kind') not in ELIGIBLE_KINDS:
            continue
        if not r.get('verified') or r.get('excluded_from_primary_experiment'):
            continue
        if r.get('evidence_availability_change'):
            continue
        reviews = r.get('reviews') or {}
        if not (_decision_ok(reviews.get('reviewer_1')) and _decision_ok(reviews.get('reviewer_2'))):
            continue
        if not r.get('mechanism'):
            continue
        pool.append(r)
    pool.sort(key=lambda r: int(r['sample_id'].split(':')[1]))
    return pool


def class_coverage(pool, min_per_class):
    """Classes with >= min_per_class verified exemplars (Tier-1 F3)."""
    counts = {}
    for r in pool:
        counts[r['mechanism']] = counts.get(r['mechanism'], 0) + 1
    return {k: v for k, v in sorted(counts.items()) if v >= min_per_class}


def feasibility_ok(pool, config):
    """Tier-1 gate F1-F3 from docs/M1_6_2E_GATE_POLICY.md."""
    g = config['gate']
    cov = class_coverage(pool, g['feasibility_min_per_class'])
    return (len(pool) >= g['feasibility_min_verified']
            and len(cov) >= g['feasibility_min_classes'])


def require_feasible(pool, config):
    if not feasibility_ok(pool, config):
        cov = class_coverage(pool, config['gate']['feasibility_min_per_class'])
        raise ValueError('Tier-1 feasibility gate not met: verified=%d classes>=2/class=%s '
                         '(see docs/M1_6_2E_GATE_POLICY.md); fail-closed, never lowered'
                         % (len(pool), cov))


def ordinal(sample_id):
    return int(sample_id.split(':')[1])


def leakage_check(target_ids, dev50_ids, group_of):
    """Hard leakage guard: targets never touch DEV50 IDs or DEV50 groups."""
    dev_groups = {group_of(s) for s in dev50_ids}
    for t in target_ids:
        if t in set(dev50_ids):
            raise ValueError('leakage: target %s is a DEV50 member' % t)
        if group_of(t) in dev_groups:
            raise ValueError('leakage: target %s shares a group with DEV50' % t)


def select_targets(candidates, predicted_class_of, covered_classes, config):
    """Locked count-then-ID selection (gold-blind, deterministic).

    candidates: dicts with sample_id and optional manifest flags. Predicted
    classes come from the frozen triage-lineage heuristic; gold is never read.
    """
    t = config['target_selection']
    frame = []
    for c in candidates:
        if c.get('artifact_outcome_exposed_directly') or c.get('artifact_outcome_exposure_propagated'):
            continue
        cls = predicted_class_of.get(c['sample_id'])
        if cls is None or cls not in covered_classes:
            continue
        frame.append(c['sample_id'])
    chosen = []
    for cls in covered_classes:
        ids = sorted((s for s in frame if predicted_class_of[s] == cls), key=ordinal)
        chosen.extend(ids[:t['n_per_class']])
    chosen.sort(key=ordinal)
    if len(chosen) < t['min_total_targets']:
        raise ValueError('target minimum not met: %d < %d; blocked, never backfilled'
                         % (len(chosen), t['min_total_targets']))
    return chosen


def assign_exemplars(target_ids, pool, predicted_class_of):
    """Deterministic within-class rotation; same inputs -> same outputs."""
    by_class = {}
    for r in pool:
        by_class.setdefault(r['mechanism'], []).append(r)
    for cls in by_class:
        by_class[cls].sort(key=lambda r: ordinal(r['sample_id']))
    assignment = {}
    seen = {}
    for t in sorted(target_ids, key=ordinal):
        cls = predicted_class_of[t]
        bucket = by_class.get(cls)
        if not bucket:
            raise ValueError('no verified exemplar for predicted class %s' % cls)
        i = seen.get(cls, 0)
        assignment[t] = bucket[i % len(bucket)]['sample_id']
        seen[cls] = i + 1
    return assignment


def neutral_filler(explanation_text):
    """Deterministic placebo matched to the B-arm explanation word count."""
    n = len((explanation_text or '').split())
    if n == 0:
        return ''
    words = []
    while len(words) < n:
        for sentence in FILLER_BANK:
            words.extend(sentence.split())
            if len(words) >= n:
                break
    return ' '.join(words[:n])


def _exemplar_quotes(exemplar):
    lines = []
    for q in exemplar.get('decisive_quotes') or []:
        lines.append('Evidence: "%s"' % q['quote'])
    return lines


def exemplar_history(arm, exemplar, filler_text=None):
    """Exemplar-history segment; shared lines identical across arms by contract."""
    text = effective_draft(exemplar)
    lines = ['Historical reference (independent of the target):',
             'Claim: %s' % exemplar['claim']]
    lines.extend(_exemplar_quotes(exemplar))
    if arm == 'C':
        lines.append('Base verdict (wrong): %s' % exemplar['base_verdict'])
    lines.append('Gold verdict: %s' % exemplar['gold_verdict'])
    if arm in ('B', 'C'):
        lines.append('Verified corrective explanation: %s' % text)
    if arm == 'D':
        lines.append('Reference note: %s' % filler_text)
    return lines


def render_target_prompt(arm, exemplar, target, evidence_lines, filler_text=None):
    """Full user-turn block; the skeleton is shared across arms."""
    if arm not in ARMS:
        raise ValueError('unknown arm %r' % arm)
    if filler_text is None and arm == 'D':
        filler_text = neutral_filler(effective_draft(exemplar))
    lines = ['Task: verify the target claim using only the displayed evidence.',
             LABEL_MENU]
    lines.extend(exemplar_history(arm, exemplar, filler_text))
    lines.append('Target claim: %s' % target['claim'])
    lines.append('Target evidence (frozen R4 Top-8):')
    lines.extend(evidence_lines)
    return {'arm': arm, 'arm_name': ARM_NAMES[arm], 'content': '\n'.join(lines),
            'exemplar_id': exemplar['sample_id'], 'target_id': target['sample_id']}


def gate_report(records=None, config=None):
    """Machine-checked gate arithmetic (mirrors docs/M1_6_2E_GATE_POLICY.md)."""
    records = load_review_records() if records is None else records
    config = load_config() if config is None else config
    counts = {k: 0 for k in ('ORIGINAL_DRAFT', 'E3_CANDIDATE_DRAFT',
                             'E2_CANDIDATE_DRAFT_QUARANTINED', 'E1_NO_DRAFT')}
    for r in records:
        counts[r['record_kind']] += 1
    originals, e3, e2 = (counts['ORIGINAL_DRAFT'], counts['E3_CANDIDATE_DRAFT'],
                         counts['E2_CANDIDATE_DRAFT_QUARANTINED'])
    if len(records) != 33 or originals + e3 + e2 + counts['E1_NO_DRAFT'] != 33:
        raise ValueError('record population drifted from the 33-record M1.6.2e frame')
    g = config['gate']
    if originals + e3 + e2 != g['max_plausible_candidates_pre_rejection']:
        raise ValueError('max-plausible arithmetic drifted from the locked policy')
    if originals + e3 != g['unchanged_top8_ceiling']:
        raise ValueError('unchanged-Top-8 ceiling drifted from the locked policy')
    pool = verified_pool(records)
    cov = class_coverage(pool, g['feasibility_min_per_class'])
    return {
        'total_records': len(records),
        'counts': counts,
        'max_plausible_candidates_pre_rejection': originals + e3 + e2,
        'unchanged_top8_ceiling': originals + e3,
        'full_memory_gate_verified': g['full_memory_gate_verified'],
        'verified_now': len(pool),
        'covered_classes': cov,
        'feasibility_ok': feasibility_ok(pool, config),
        'gate': 'TIER1_UNMET_FAIL_CLOSED' if not feasibility_ok(pool, config)
                else 'TIER1_MET_PENDING_LOCKED_EXECUTION',
    }
