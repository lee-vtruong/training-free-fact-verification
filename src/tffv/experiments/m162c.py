"""M1.6.2c: cache-only four-class gold-only coverage arms on DEVELOPMENT50.

Three preregistered arms, each k=3 with the same canonical Mixed pair plus one
added Base-correct gold-only exemplar: a same-gold placebo (token/composition
control), a Not-Enough-Evidence exemplar, and a Conflicting exemplar. Reuses
immutable canonical R4 Top-8 evidence and baseline predictions; contains no
retrieval command and never regenerates P0/P1/P2/P3/N0/Mixed. GPU runs require
explicit --execute. History-revealing arms are parked: assistant drafts are not
human-verified rationale, so Base correct/wrong history is not displayed in
gold-only mode and cannot be separated without reviewed corrections.
"""
from pathlib import Path
import argparse, json, time

from tffv.provenance import digest, file_hash
from tffv.memory.heal import read, save, update, jsonl, indexed
from tffv.experiments import m16_cache, m161
from tffv.experiments.m162b import validate_pair, out
from tffv.verification import heal_pilot as hp

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('COV_PLACEBO_G3', 'COV_NEI_G3', 'COV_CONF_G3')
ORDER = ('pair_correct', 'added', 'pair_wrong')
ADDED_CLASS = {'COV_PLACEBO_G3': None,
               'COV_NEI_G3': 'Not Enough Evidence',
               'COV_CONF_G3': 'Conflicting Evidence/Cherrypicking'}
ADDED_HEADER = 'Reference 2 — additional historical example, gold-only:\n'


def load_config():
    config = read(ROOT / 'configs/m162c_coverage_gold_only.json')
    if (list(config['arms']) != list(ARMS) or config['k'] != 3
            or tuple(config['order']) != ORDER
            or config['generation'] != {'max_new_tokens': 512, 'enable_thinking': False,
                                        'do_sample': False}
            or config['added_context_excerpts'] != 1
            or config['added_selection_rule'] != SELECT_RULE):
        raise ValueError('Frozen coverage config changed')
    return config


SELECT_RULE = ('usable SOURCE Base-correct exemplar of the arm gold class; '
               'highest evidence-item count, ties ascending sample_id')


def usable(record):
    items = record.get('evidence') or []
    return bool(items) and all(i.get('source_url') and i.get('rendered_text') for i in items)


def select_added(records, gold_class, exclude, config):
    """Preregistered deterministic rule; no dev gold, no seeds, no tuning."""
    pool = [r for r in records.values()
            if r['gold_verdict'] == gold_class and not r['is_base_error']
            and r['sample_id'] not in exclude and usable(r)]
    if not pool:
        raise ValueError('No usable Base-correct exemplar for class: ' + str(gold_class))
    return sorted(pool, key=lambda r: (-len(r['evidence']), r['sample_id']))[0]


def coverage_messages(claim, evidence, slate, memory, added):
    records = indexed(memory['records'])
    good, bad = records[slate['correct_id']], records[slate['error_id']]
    native = hp.build_d1_matched_prompt(claim, evidence)['messages']
    added_context = list(hp.source_context(added))[:config_excerpts()]
    blocks = [hp.block(good, 'P2', slate['correct_context']),
              ADDED_HEADER + hp.block(added, 'P2', added_context),
              hp.block(bad, 'P2', slate['error_context'])]
    messages = [dict(native[0]), dict(native[1])]
    messages[1]['content'] = ('Historical task references (independent of the target; decide '
                              'the target from its own evidence).\n'
                              'Reference 1 — Base-correct source example, gold-only:\n'
                              + blocks[0] + '\n' + blocks[1] + '\n' + blocks[2]
                              + '\n--- Target ---\n' + messages[1]['content'])
    return messages


def config_excerpts():
    return load_config()['added_context_excerpts']


def prepare(canonical_run, cache_index, memory_path, tokenizer_path, arm, destination):
    if arm not in ARMS:
        raise SystemExit('Unknown coverage arm: ' + str(arm))
    config = load_config()
    run = Path(canonical_run).resolve()
    b = read(run / 'batch.json')
    index = m16_cache.load_index(cache_index)
    if file_hash(run / 'batch.json') != index['batch_sha256'] or \
            index['active_conditions'] != b['status']['active_conditions']:
        raise ValueError('Canonical cache identity/condition mismatch')
    if index['source_memory_sha256'] != file_hash(memory_path) or \
            b['source_memory_sha256'] != file_hash(memory_path):
        raise ValueError('Source memory cache mismatch')
    from tffv.memory.heal import SourceContract
    contract = SourceContract(ROOT)
    mem = read(memory_path)
    contract.check_memory(mem)
    records = indexed(mem['records'])
    tok = hp.tokenizer(Path(tokenizer_path))
    r4 = {x['sample_id']: x for x in index['records']}
    claims = indexed(b['claims'])
    slates = {s['sample_id']: s for s in b['slates']}
    added_by_target = {}
    for sid, slate in slates.items():
        validate_pair(slate, records, contract)
        target_group = contract.rows[sid]['group_id']
        exclude = {slate['correct_id'], slate['error_id']}
        gold_class = ADDED_CLASS[arm] or records[slate['correct_id']]['gold_verdict']
        added = select_added(records, gold_class, exclude, config)
        overlap = {target_group, records[slate['correct_id']]['group_id'],
                   records[slate['error_id']]['group_id']}
        if added['sample_id'] in exclude or added['group_id'] in overlap:
            raise ValueError('Added exemplar collides with pair/target: ' + sid)
        added_by_target[sid] = added
    prompts = {}
    for sid, slate in slates.items():
        added = added_by_target[sid]
        entry = r4.get(sid)
        if entry is None or entry['r4_sha256'] != file_hash(entry['r4_path']):
            raise ValueError('Missing/mismatched immutable R4 cache: ' + sid)
        evidence = jsonl(entry['r4_path'])
        if len(evidence) != 1 or evidence[0]['sample_id'] != sid or \
                len(evidence[0]['evidence']) != 8:
            raise ValueError('Incomplete Top-8 cache: ' + sid)
        messages = coverage_messages(claims[sid]['claim'], evidence[0]['evidence'],
                                     slate, mem, added)
        ids = hp.tokens(tok, messages)
        if len(ids) + 512 > 8192:
            raise ValueError('Coverage prompt context overflow: ' + sid)
        prompts[sid] = {'messages': messages, 'messages_sha256': digest(messages),
                        'input_ids_sha256': digest(ids), 'prompt_tokens': len(ids),
                        'evidence_cache_key': entry['evidence_cache_key'],
                        'exemplar_ids': [slate['correct_id'], added['sample_id'],
                                         slate['error_id']],
                        'exemplar_roles': list(ORDER),
                        'added_gold_verdict': added['gold_verdict'],
                        'gold_accessed': False}
    obj = {'schema_version': 'm162c-coverage-v1', 'scope': 'EXPLORATORY_CACHE_ONLY',
           'condition': arm, 'arms': list(ARMS), 'k': 3, 'order': list(ORDER),
           'canonical_run': str(run),
           'canonical_batch_sha256': file_hash(run / 'batch.json'),
           'cache_index_sha256': file_hash(cache_index),
           'source_memory_sha256': file_hash(memory_path),
           'cache_index_path': str(Path(cache_index).resolve()),
           'sample_ids': b['sample_ids'], 'prompts': prompts,
           'condition_config_sha256':
               file_hash(ROOT / 'configs/m162c_coverage_gold_only.json'),
           'generation': {'max_new_tokens': 512, 'enable_thinking': False,
                          'do_sample': False},
           'backbone': b['configuration']['backbone'],
           'selection_label': ('added Base-correct exemplar of gold class '
                               + ADDED_CLASS[arm]) if ADDED_CLASS[arm]
           else 'added Base-correct exemplar of the pair gold class (placebo control)',
           'history_displayed': False, 'gold_accessed': False}
    dest = out(destination)
    dest.mkdir(parents=True, exist_ok=False)
    save(dest / 'coverage_batch.json', obj)
    save(dest / 'state.json', {'completed_ids': [], 'spent_seconds': 0.,
                               'active_child_pid': None})
    print(json.dumps({'prepared': str(dest), 'arm': arm, 'claims': len(prompts),
                      'new_inference_calls': len(prompts), 'k': 3, 'retrieval_calls': 0,
                      'added_class': ADDED_CLASS[arm] or 'pair_gold',
                      'added_ids': sorted({added_by_target[s]['sample_id']
                                           for s in slates}),
                      'gold_accessed': False}, indent=2))
    return obj


def check(run):
    run = out(run)
    b = read(run / 'coverage_batch.json')
    if b['condition'] not in ARMS or list(b['arms']) != list(ARMS) or \
            b['k'] != 3 or tuple(b['order']) != ORDER or b['gold_accessed'] is not False:
        raise ValueError('Coverage protocol identity mismatch')
    if len(set(b['sample_ids'])) != 50 or set(b['prompts']) != set(b['sample_ids']):
        raise ValueError('Coverage prompt coverage mismatch')
    if b['generation'] != {'max_new_tokens': 512, 'enable_thinking': False, 'do_sample': False}:
        raise ValueError('Coverage decoding changed')
    if b['condition_config_sha256'] != file_hash(ROOT / 'configs/m162c_coverage_gold_only.json'):
        raise ValueError('Coverage condition config changed')
    if file_hash(b['cache_index_path']) != b['cache_index_sha256']:
        raise ValueError('Cache index changed after coverage preparation')
    cache = m16_cache.load_index(b['cache_index_path'])
    byid = {r['sample_id']: r for r in cache['records']}
    for sid, p in b['prompts'].items():
        if p['exemplar_roles'] != list(ORDER) or len(p['exemplar_ids']) != 3 or \
                p['prompt_tokens'] + 512 > 8192:
            raise ValueError('Coverage prompt contract mismatch: ' + sid)
        content = p['messages'][1]['content']
        if not (content.index('Reference 1') < content.index(str(ADDED_HEADER).splitlines()[0])
                < content.index('--- Target ---')):
            raise ValueError('Coverage block ordering violated: ' + sid)
        if 'Previous Base verdict' in content and '[not provided]' not in content:
            raise ValueError('Coverage gold-only mask violated: ' + sid)
        if 'Verified correction: Verified' in content or \
                'incorrect for this historical example' in content:
            raise ValueError('Coverage history display leaked: ' + sid)
        e = byid.get(sid)
        if e is None or e['evidence_cache_key'] != p['evidence_cache_key'] or \
                file_hash(e['r4_path']) != e['r4_sha256']:
            raise ValueError('Missing/mismatched immutable R4 cache: ' + sid)
    return b


def receipt(run, sid):
    path = out(run) / sid.replace(':', '-') / 'prediction.receipt.json'
    if not path.exists():
        return None
    r = read(path)
    q = (out(run) / r['output']).resolve()
    if r['sample_id'] != sid or r['batch_sha256'] != file_hash(out(run) / 'coverage_batch.json') \
            or not q.is_file() or file_hash(q) != r['output_sha256']:
        raise ValueError('Coverage prediction receipt mismatch')
    return r, q


def worker(run, sid):
    from tffv.verification.backbone import Qwen3FrozenBackbone
    from tffv.verification.ontology import normalize_verdict
    run = out(run)
    b = check(run)
    p = b['prompts'][sid]
    model = Qwen3FrozenBackbone(b['backbone'], device='cuda', dtype='bfloat16')
    try:
        ids = hp.tokens(model.tokenizer, p['messages'])
        if digest(ids) != p['input_ids_sha256']:
            raise ValueError('Effective coverage prompt changed')
        model.reset_peak()
        model.torch.cuda.synchronize()
        started = time.perf_counter()
        g = model.generate(p['messages'], max_new_tokens=512, enable_thinking=False)
        model.torch.cuda.synchronize()
        peak = model.peak_bytes()
    finally:
        model.close()
    d = run / sid.replace(':', '-')
    attempt = 1
    while (d / f'prediction-attempt-{attempt:03d}').exists():
        attempt += 1
    target = d / f'prediction-attempt-{attempt:03d}'
    target.mkdir(parents=True)
    row = {'sample_id': sid, 'condition': b['condition'],
           'raw_generation': g['generated_text'],
           'predicted_verdict': normalize_verdict(g['generated_text']),
           'prompt_tokens': g['prompt_tokens'], 'output_tokens': g['generated_tokens'],
           'latency_seconds': time.perf_counter() - started, 'peak_gpu_memory_bytes': peak,
           'input_ids_sha256': p['input_ids_sha256'], 'exemplar_ids': p['exemplar_ids'],
           'gold_accessed': False, 'max_new_tokens': 512, 'enable_thinking': False}
    save(target / 'prediction.json', row)
    save(d / 'prediction.receipt.json',
         {'sample_id': sid, 'batch_sha256': file_hash(run / 'coverage_batch.json'),
          'output': (target / 'prediction.json').relative_to(run).as_posix(),
          'output_sha256': file_hash(target / 'prediction.json')})
    return row


def execute(run, allowed=False):
    if not allowed:
        raise ValueError('New GPU inference requires explicit --execute')
    run = out(run)
    with m161.run_lock(run, False):
        b = check(run)
        import torch
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 24 * 1024**3:
            raise ValueError('Need Linux CUDA with >=24GiB free VRAM')
        state = read(run / 'state.json')
        for sid in b['sample_ids']:
            if receipt(run, sid):
                if sid not in state['completed_ids']:
                    state['completed_ids'].append(sid)
                continue
            started = time.time()
            worker(run, sid)
            state['spent_seconds'] += time.time() - started
            state['completed_ids'].append(sid)
            update(run / 'state.json', state)
            print(f'CHECKPOINT {b["condition"]}={len(state["completed_ids"])}/50 sid={sid}',
                  flush=True)
        if len(state['completed_ids']) != 50:
            raise ValueError('Incomplete coverage prediction population')
        save(run / 'completed.json',
             {'batch_sha256': file_hash(run / 'coverage_batch.json'),
              'sample_ids': b['sample_ids'], 'condition': b['condition'],
              'gold_accessed': False})


def evaluate(run, canonical_run, mixed_run, gold_source, destination):
    run = out(run)
    b = check(run)
    done = read(run / 'completed.json')
    if done['batch_sha256'] != file_hash(run / 'coverage_batch.json') or \
            done['sample_ids'] != b['sample_ids']:
        raise ValueError('Coverage run incomplete')
    raw = read(gold_source)
    pred = []
    rows = []
    baselines = {'P2': [], 'N0': [], 'MIXED_GOLD_ONLY_K2': []}
    for sid in b['sample_ids']:
        _, p = receipt(run, sid)
        x = read(p)
        pred.append(x['predicted_verdict'])
        rows.append(x)
        prior = {r['condition']: r for r in jsonl(Path(canonical_run) / sid.replace(':', '-')
                                                  / 'predictions-attempt-001' / 'predictions.jsonl')}
        mixed = sorted((Path(mixed_run) / sid.replace(':', '-'))
                       .glob('prediction-attempt-*/prediction.json'))
        if not {'P2', 'N0'}.issubset(prior) or len(mixed) != 1:
            raise ValueError('Baseline rows missing for: ' + sid)
        baselines['P2'].append(prior['P2']['predicted_verdict'])
        baselines['N0'].append(prior['N0']['predicted_verdict'])
        baselines['MIXED_GOLD_ONLY_K2'].append(read(mixed[-1])['predicted_verdict'])
    gold = [raw[int(s.split(':')[1])]['label'] for s in b['sample_ids']]
    result = {'scope': 'EXPLORATORY_NOT_CONFIRMATORY', 'condition': b['condition'],
              'claims': 50, 'metrics': hp.metric(gold, pred),
              'prompt_tokens': sum(x['prompt_tokens'] for x in rows),
              'output_tokens': sum(x['output_tokens'] for x in rows),
              'generation_wall_seconds': sum(x['latency_seconds'] for x in rows),
              'gpu_active_seconds': None}
    result['comparisons'] = {}
    for name, base in baselines.items():
        z = hp.compare(gold, base, pred, seed=42, n_boot=10000, n_perm=19999)
        z['correction_ids'] = [sid for sid, g, a, y in zip(b['sample_ids'], gold, base, pred)
                               if a != g and y == g]
        z['regression_ids'] = [sid for sid, g, a, y in zip(b['sample_ids'], gold, base, pred)
                               if a == g and y != g]
        result['comparisons'][b['condition'] + '_vs_' + name] = z
    dest = out(destination)
    dest.mkdir(parents=True, exist_ok=False)
    save(dest / 'results.json', result)
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('prepare')
    q.add_argument('--canonical-run', type=Path, required=True)
    q.add_argument('--cache-index', type=Path, required=True)
    q.add_argument('--memory', type=Path, required=True)
    q.add_argument('--tokenizer', type=Path, required=True)
    q.add_argument('--arm', required=True, choices=list(ARMS))
    q.add_argument('--output', type=Path, required=True)
    q = s.add_parser('check')
    q.add_argument('--run', type=Path, required=True)
    q = s.add_parser('run')
    q.add_argument('--run', type=Path, required=True)
    q.add_argument('--execute', action='store_true')
    q = s.add_parser('evaluate')
    q.add_argument('--run', type=Path, required=True)
    q.add_argument('--canonical-run', type=Path, required=True)
    q.add_argument('--mixed-run', type=Path, required=True)
    q.add_argument('--gold-source', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.cmd == 'prepare':
        prepare(a.canonical_run, a.cache_index, a.memory, a.tokenizer, a.arm, a.output)
    elif a.cmd == 'check':
        print(json.dumps({k: v for k, v in check(a.run).items()
                          if k not in ('prompts',)}, indent=2))
    elif a.cmd == 'run':
        execute(a.run, a.execute)
    else:
        evaluate(a.run, a.canonical_run, a.mixed_run, a.gold_source, a.output)


if __name__ == '__main__':
    main()
