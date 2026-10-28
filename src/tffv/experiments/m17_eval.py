"""M1.7 Exploratory Feasibility Evaluation Runner (24 Development Claims).

Four arms evaluated under frozen Qwen3-8B verifier:
  Arm A: gold_only
  Arm B: gold_validated (Gold + LLM-validated corrective rationale)
  Arm C: wrong_gold_validated (Base wrong verdict + Gold + same LLM-validated corrective rationale)
  Arm D: gold_filler_length_matched (Gold + length-matched neutral placebo)

Protocol constraints:
  - 24 clean, mutually disjoint development claims (zero DEV50 leakage).
  - Frozen R4 Top-8 evidence retrieved once per claim and shared across all 4 arms.
  - Frozen decoding: temperature 0.0, max_new_tokens 512, greedy.
  - Target gold accessed strictly at final scoring time.
"""
from collections import Counter
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from tffv.experiments import m161, m162e
from tffv.provenance import file_hash
from tffv.verification import heal_pilot as hp
from tffv.verification.ontology import VERDICTS, normalize_verdict

def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')

ROOT = Path(__file__).resolve().parents[3]
MEMORY_PATH = ROOT / 'outputs' / 'm17-llm-validated-memory-v1' / 'memory.json'
DEV_MANIFEST = ROOT / 'outputs' / 'm16-partitions-v1' / 'development_manifest.json'
CLAIMS_MANIFEST = ROOT / 'outputs' / 'm16-partitions-v1' / 'claims_manifest.json'
DEV50_BATCH = ROOT / 'outputs' / 'm162a-dev50-preparation-v4' / 'batch.json'
ACQUISITION_PLAN = ROOT / 'configs' / 'm161_acquisition_plan.json'
CONFIG_PATH = ROOT / 'configs' / 'm161_source_batch25.json'
TOKENIZER_DIR = Path('/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')

TARGET_COUNT = 24


def select_clean_development_targets(n=TARGET_COUNT):
    """Select n clean, mutually disjoint development claims with zero DEV50 collision."""
    dev_claims = json.loads(DEV_MANIFEST.read_text('utf-8-sig'))
    all_claims = json.loads(CLAIMS_MANIFEST.read_text('utf-8-sig'))
    dev50_ids = set(json.loads(DEV50_BATCH.read_text('utf-8-sig'))['sample_ids'])

    claim_map = {r['sample_id']: r for r in all_claims}
    dev50_groups = {claim_map[s]['group_id'] for s in dev50_ids}

    clean = []
    seen_groups = set()
    for r in dev_claims:
        sid = r['sample_id']
        if sid in dev50_ids:
            continue
        g = claim_map[sid]['group_id']
        if g in dev50_groups or g in seen_groups:
            continue
        if r.get('artifact_outcome_exposed_directly') or r.get('artifact_outcome_exposure_propagated'):
            continue
        clean.append(sid)
        seen_groups.add(g)

    clean.sort(key=lambda s: int(s.split(':')[1]))
    if len(clean) < n:
        raise ValueError(f'Insufficient clean disjoint development claims: {len(clean)} < {n}')
    return clean[:n]


def prepare(output_dir, memory_path=MEMORY_PATH, n_targets=TARGET_COUNT):
    """Prepare 24-claim batch, inputs, and exemplar assignments."""
    out = Path(output_dir).resolve()
    if out.exists() and (out / 'batch.json').exists():
        print(f'Using existing batch at {out}')
        return out

    out.mkdir(parents=True, exist_ok=True)
    targets = select_clean_development_targets(n_targets)
    all_claims = {r['sample_id']: r for r in json.loads(CLAIMS_MANIFEST.read_text('utf-8-sig'))}
    mem = json.loads(Path(memory_path).read_text('utf-8-sig'))
    exemplars = mem['records']

    # Deterministic exemplar assignment (rotation across 21 LLM-validated exemplars)
    assignments = {}
    for i, sid in enumerate(targets):
        ex = exemplars[i % len(exemplars)]
        assignments[sid] = ex['sample_id']

    # Setup per-claim input directory
    for sid in targets:
        claim_data = all_claims[sid]
        claim_dir = out / sid.replace(':', '-') / 'input'
        claim_dir.mkdir(parents=True, exist_ok=True)
        (claim_dir / 'claims.jsonl').write_text(json.dumps(claim_data) + '\n', encoding='utf-8')
        save(claim_dir / 'claims_manifest.json', {
            'dataset': 'AVeriTeC', 'split': 'train', 'sample_id': sid,
            'role': 'DEVELOPMENT_RESEARCH_NOT_HISTORICAL_SOURCE',
            'annotation_values_used': False,
            'claims_sha256': file_hash(claim_dir / 'claims.jsonl')
        })

    batch = {
        'schema_version': 'm17-eval-24dev-v1',
        'sample_ids': targets,
        'exemplar_assignments': assignments,
        'memory_path': str(Path(memory_path).resolve()),
        'memory_sha256': file_hash(memory_path),
        'arms': ['A', 'B', 'C', 'D'],
        'arm_names': {'A': 'gold_only', 'B': 'gold_validated',
                      'C': 'wrong_gold_validated', 'D': 'gold_filler_length_matched'},
        'configuration_path': str(CONFIG_PATH.resolve()),
        'acquisition_plan_path': str(ACQUISITION_PLAN.resolve())
    }
    save(out / 'batch.json', batch)
    save(out / 'state.json', {
        'spent_seconds': 0.0,
        'retrieval_completed': [],
        'prediction_completed': []
    })
    print(f'Prepared {len(targets)} targets in {out}')
    return out


def retrieve_one_claim(run_dir, sid, config):
    """Run AVeriTeC R1..R4 retrieval pipeline for a single target claim."""
    folder = run_dir / sid.replace(':', '-')
    receipt_path = folder / 'retrieval.receipt.json'
    if receipt_path.exists():
        return folder / 'retrieval' / 'R4.jsonl'

    from tffv.retrieval.acquire_subset import acquire
    from tffv.retrieval.audit_subset import audit
    from tffv.retrieval.prepare_store_candidates import prepare_candidates
    from tffv.retrieval.store_retrieval import run as retrieve_store

    ordinal = int(sid.split(':')[1])
    # 1. Acquire
    acq_out = folder / 'acquire'
    acquire(config['acquisition_plan_path'], acq_out, [ordinal], execute=True)

    # 2. Audit
    aud_out = folder / 'audit'
    audit(acq_out, aud_out, max_record_mib=64)

    # 3. Candidates
    cm = folder / 'input' / 'claims_manifest.json'
    ca_out = folder / 'candidates'
    prepare_candidates(aud_out, cm, ca_out, policy=config.get('candidate_policy', 'quarantine_review_flags_v1'))

    # 4. Retrieval R1..R4
    ret_out = folder / 'retrieval'
    r4_file = retrieve_store(ca_out, ret_out, methods=('R1', 'R2', 'R3', 'R4'),
                             config=config['retrieval'], index_cache=ret_out / 'embedding_cache') / 'R4.jsonl'

    save(receipt_path, {
        'sample_id': sid, 'stage': 'retrieval',
        'r4_file': str(r4_file.relative_to(run_dir).as_posix()),
        'r4_sha256': file_hash(r4_file)
    })
    return r4_file


def run_retrieval_all(run_dir):
    """Sequentially retrieve R4 Top-8 evidence for all 24 claims."""
    run = Path(run_dir).resolve()
    batch = json.loads((run / 'batch.json').read_text('utf-8-sig'))
    cfg = json.loads(Path(batch['configuration_path']).read_text('utf-8-sig'))
    cfg['acquisition_plan_path'] = batch['acquisition_plan_path']

    state = json.loads((run / 'state.json').read_text('utf-8-sig'))
    done = set(state.get('retrieval_completed', []))

    total = len(batch['sample_ids'])
    for i, sid in enumerate(batch['sample_ids']):
        if sid in done:
            print(f'[{i+1}/{total}] Retrieval already done for {sid}')
            continue
        print(f'[{i+1}/{total}] Starting retrieval for {sid}...')
        t0 = time.time()
        r4 = retrieve_one_claim(run, sid, cfg)
        elapsed = time.time() - t0
        print(f'[{i+1}/{total}] Completed retrieval for {sid} in {elapsed:.1f}s (R4 sha256: {file_hash(r4)[:12]}...)')
        done.add(sid)
        state['retrieval_completed'] = sorted(done, key=lambda s: int(s.split(':')[1]))
        save(run / 'state.json', state)
    print('All target evidence retrieval completed successfully.')


def format_evidence_lines(top8_evidence):
    """Format Top-8 evidence passages into prompt lines."""
    lines = []
    for i, e in enumerate(top8_evidence):
        text = e.get('rendered_text') or e.get('quote') or e.get('text', '')
        lines.append(f'[{i+1}] {e.get("source_url", "")}\n{text}')
    return lines


def format_exemplar_lines(arm, exemplar, filler_text):
    """Format historical exemplar block for each arm."""
    lines = [
        'Historical reference (independent of the target):',
        f'Claim: {exemplar["claim"]}'
    ]
    # Add decisive quotes from the judge assessment
    quotes = exemplar.get('llm_judge', {}).get('correction_validation', {}).get('grounding_quotes', [])
    if not quotes and exemplar.get('evidence'):
        quotes = [exemplar['evidence'][0].get('quote', '')]
    for q in quotes:
        lines.append(f'Evidence: "{q}"')

    if arm == 'C':
        lines.append(f'Base verdict (wrong): {exemplar["base_verdict"]}')
    lines.append(f'Gold verdict: {exemplar["gold_verdict"]}')

    if arm in ('B', 'C'):
        lines.append(f'Verified corrective explanation: {exemplar["correction"]}')
    elif arm == 'D':
        lines.append(f'Reference note: {filler_text}')
    return lines


def render_m17_prompt(arm, exemplar, target_claim, evidence_lines):
    """Render full prompt for a specific arm."""
    filler = m162e.neutral_filler(exemplar['correction']) if arm == 'D' else None
    lines = [
        'Task: verify the target claim using only the displayed evidence.',
        m162e.LABEL_MENU
    ]
    lines.extend(format_exemplar_lines(arm, exemplar, filler))
    lines.append(f'Target claim: {target_claim}')
    lines.append('Target evidence (frozen R4 Top-8):')
    lines.extend(evidence_lines)
    return '\n'.join(lines)


def run_prediction_all(run_dir):
    """Run Qwen3-8B verification inference across 4 arms for all targets."""
    run = Path(run_dir).resolve()
    batch = json.loads((run / 'batch.json').read_text('utf-8-sig'))
    cfg = json.loads(Path(batch['configuration_path']).read_text('utf-8-sig'))
    mem = json.loads(Path(batch['memory_path']).read_text('utf-8-sig'))
    exemplar_map = {r['sample_id']: r for r in mem['records']}
    all_claims = {r['sample_id']: r for r in json.loads(CLAIMS_MANIFEST.read_text('utf-8-sig'))}

    from tffv.verification.backbone import Qwen3FrozenBackbone

    print('Loading frozen Qwen3-8B verifier backbone on CUDA...')
    backbone = Qwen3FrozenBackbone(cfg['backbone'], device='cuda', dtype='bfloat16')
    print('Qwen3-8B loaded successfully.')

    state = json.loads((run / 'state.json').read_text('utf-8-sig'))
    done = set(state.get('prediction_completed', []))

    predictions_out = run / 'predictions'
    predictions_out.mkdir(parents=True, exist_ok=True)

    total = len(batch['sample_ids'])
    for i, sid in enumerate(batch['sample_ids']):
        if sid in done:
            print(f'[{i+1}/{total}] Predictions already done for {sid}')
            continue

        target_claim = all_claims[sid]['claim']
        ex = exemplar_map[batch['exemplar_assignments'][sid]]

        # Load target Top-8 R4 evidence
        r4_file = run / sid.replace(':', '-') / 'retrieval' / 'R4.jsonl'
        r4_data = [json.loads(line) for line in r4_file.read_text('utf-8-sig').splitlines() if line.strip()][0]
        ev_lines = format_evidence_lines(r4_data['evidence'])

        claim_preds = {}
        for arm in ['A', 'B', 'C', 'D']:
            prompt_text = render_m17_prompt(arm, ex, target_claim, ev_lines)
            messages = [{'role': 'user', 'content': prompt_text}]

            t0 = time.time()
            gen = backbone.generate(messages, max_new_tokens=512, enable_thinking=False)
            latency = time.time() - t0
            raw_text = gen['generated_text']
            verdict = normalize_verdict(raw_text)

            claim_preds[arm] = {
                'sample_id': sid, 'arm': arm,
                'arm_name': batch['arm_names'][arm],
                'exemplar_id': ex['sample_id'],
                'predicted_verdict': verdict,
                'raw_generation': raw_text,
                'prompt_tokens': gen['prompt_tokens'],
                'output_tokens': gen['generated_tokens'],
                'latency_seconds': latency
            }
            print(f'  [{sid}] Arm {arm} -> {verdict} ({latency:.2f}s)')

        save(predictions_out / f'{sid.replace(":", "-")}.json', claim_preds)
        done.add(sid)
        state['prediction_completed'] = sorted(done, key=lambda s: int(s.split(':')[1]))
        save(run / 'state.json', state)

    print('All predictions completed.')


def evaluate_run(run_dir, gold_path='/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json'):
    """Score predictions against target gold and compute paired McNemar / bootstrap metrics."""
    run = Path(run_dir).resolve()
    batch = json.loads((run / 'batch.json').read_text('utf-8-sig'))
    predictions_out = run / 'predictions'

    # Load target gold labels
    gold_data = json.loads(Path(gold_path).read_text('utf-8-sig'))
    gold_labels = [gold_data[int(sid.split(':')[1])]['label'] for sid in batch['sample_ids']]

    # Load predictions per arm
    arms = ['A', 'B', 'C', 'D']
    arm_preds = {arm: [] for arm in arms}
    for sid in batch['sample_ids']:
        pred_file = predictions_out / f'{sid.replace(":", "-")}.json'
        preds = json.loads(pred_file.read_text('utf-8-sig'))
        for arm in arms:
            arm_preds[arm].append(preds[arm]['predicted_verdict'])

    results = {
        'schema_version': 'm17-evaluation-report-v1',
        'target_claims': len(batch['sample_ids']),
        'sample_ids': batch['sample_ids'],
        'metrics': {},
        'comparisons': {}
    }

    for arm in arms:
        results['metrics'][arm] = hp.metric(gold_labels, arm_preds[arm])
        results['metrics'][arm]['arm_name'] = batch['arm_names'][arm]

    # Paired comparisons
    contrasts = [('A', 'B'), ('A', 'C'), ('B', 'D')]
    for base_arm, target_arm in contrasts:
        key = f'{target_arm}_vs_{base_arm}'
        comp = hp.compare(gold_labels, arm_preds[base_arm], arm_preds[target_arm])
        comp['base_arm'] = base_arm
        comp['target_arm'] = target_arm
        comp['base_arm_name'] = batch['arm_names'][base_arm]
        comp['target_arm_name'] = batch['arm_names'][target_arm]
        results['comparisons'][key] = comp

    save(run / 'results.json', results)

    # Generate Markdown Report
    report_lines = [
        '# M1.7 Exploratory Feasibility Evaluation Report',
        '',
        f'- **Target Population:** {len(batch["sample_ids"])} Development claims (zero DEV50 leakage).',
        f'- **Memory Bank:** {batch["memory_path"]} (`LLM_VALIDATED` status).',
        '- **Verifier Backbone:** Frozen Qwen3-8B (bfloat16, greedy decoding).',
        '',
        '## 1. Arm Metrics Summary',
        '',
        '| Arm | Name | Accuracy | Macro-F1 | Valid Rate |',
        '| :--- | :--- | :--- | :--- | :--- |'
    ]
    for arm in arms:
        m = results['metrics'][arm]
        report_lines.append(f'| **{arm}** | {m["arm_name"]} | {m["accuracy"]*100:.2f}% | {m["macro_f1"]:.4f} | {m["valid_rate"]*100:.1f}% |')

    report_lines.extend([
        '',
        '## 2. Paired Contrasts',
        '',
        '| Contrast | Comparison | $\\Delta$ Accuracy | McNemar p-value | $\\Delta$ Macro-F1 | 95% Bootstrap CI ($\\Delta$ Acc) |',
        '| :--- | :--- | :--- | :--- | :--- | :--- |'
    ])
    for key, c in results['comparisons'].items():
        ci = f"[{c['accuracy_bootstrap_ci95'][0]*100:+.1f}%, {c['accuracy_bootstrap_ci95'][1]*100:+.1f}%]"
        report_lines.append(f'| **{key}** | {c["target_arm_name"]} vs {c["base_arm_name"]} | {c["accuracy_delta"]*100:+.2f}% | {c["exact_mcnemar_p"]:.4f} | {c["macro_f1_delta"]:+.4f} | {ci} |')

    report_text = '\n'.join(report_lines)
    (run / 'REPORT.md').write_text(report_text, encoding='utf-8')
    print(f'Evaluation report written to {run / "REPORT.md"}')
    print(report_text)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_prep = sub.add_parser('prepare')
    p_prep.add_argument('--output', required=True)
    p_prep.add_argument('--memory', default=str(MEMORY_PATH))
    p_prep.add_argument('--targets', type=int, default=TARGET_COUNT)

    p_ret = sub.add_parser('retrieve')
    p_ret.add_argument('--run', required=True)

    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--run', required=True)

    p_eval = sub.add_parser('evaluate')
    p_eval.add_argument('--run', required=True)
    p_eval.add_argument('--gold', default='/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json')

    args = parser.parse_args()
    if args.cmd == 'prepare':
        prepare(args.output, args.memory, args.targets)
    elif args.cmd == 'retrieve':
        run_retrieval_all(args.run)
    elif args.cmd == 'predict':
        run_prediction_all(args.run)
    elif args.cmd == 'evaluate':
        evaluate_run(args.run, args.gold)


if __name__ == '__main__':
    main()
