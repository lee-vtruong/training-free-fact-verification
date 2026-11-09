"""Deterministic label-free CPU grouping/partition audit; no model calls.

Read only the allowlisted metadata projection, exposure inventory and frozen config.
All conservative lexical edges are retained, including transitive bridges.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text or '').casefold()))


def canonical_url(url):
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        u = urlsplit(url.strip())
    except ValueError:
        return None
    if u.scheme not in ('http', 'https') or not u.hostname:
        return None
    host = u.hostname.lower().removeprefix('www.')
    path = re.sub('/+', '/', u.path).rstrip('/')
    query = sorted((k, v) for k, v in parse_qsl(u.query, keep_blank_values=True)
                   if not k.lower().startswith('utm_') and k.lower() not in ('fbclid', 'gclid'))
    if not path and not query:
        return None  # Generic homepages are not event/article identities.
    return urlunsplit(('https', host, path, urlencode(query), ''))


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def build_groups(rows, config):
    uf = UnionFind(len(rows))
    words = [normalize(r['claim']).split() for r in rows]
    tokens = [set(w) for w in words]
    grams = [set(zip(w, w[1:], w[2:])) for w in words]
    edges = []
    for i in range(len(rows)):
        if not tokens[i]:
            continue
        for j in range(i):
            if not tokens[j]:
                continue
            common = len(tokens[i] & tokens[j])
            jac = common / len(tokens[i] | tokens[j]) if common else 0
            if grams[i] and grams[j]:
                gc = len(grams[i] & grams[j])
                containment = gc / min(len(grams[i]), len(grams[j]))
            else:
                containment = jac
            if jac >= config['token_jaccard_threshold'] or containment >= config['symmetric_token_trigram_containment_threshold']:
                uf.union(i, j)
                edges.append({'a': rows[j]['sample_id'], 'b': rows[i]['sample_id'],
                              'kind': 'lexical', 'jaccard': round(jac, 8), 'max_containment': round(containment, 8)})
    lexical_roots = [uf.find(i) for i in range(len(rows))]
    lexical_ids = {}
    for i, r in enumerate(rows):
        lexical_ids.setdefault(lexical_roots[i], []).append(r['sample_id'])
    dup_map = {sid: 'dup-' + sha(canonical(sorted(v)))[:16] for v in lexical_ids.values() for sid in v}
    keys = defaultdict(list)
    anchors = config['event_rules']['anchors']
    generic_speakers = {'unknown', 'none', 'n a', 'na', 'social media', 'facebook', 'twitter', 'multiple', 'various'}
    for i, r in enumerate(rows):
        if not tokens[i]:
            continue
        for field in ('fact_checking_article', 'original_claim_url'):
            u = canonical_url(r.get(field))
            if u:
                keys[(field, u)].append(i)
        speaker = normalize(r.get('speaker'))
        date = r.get('claim_date') or ''
        month = None
        for fmt in config['event_rules']['claim_date_formats']:
            try:
                month = datetime.strptime(date, fmt).strftime('%Y-%m')
                break
            except ValueError:
                pass
        if speaker and speaker not in generic_speakers and month:
            text = ' ' + normalize(r['claim']) + ' '
            for anchor, terms in anchors.items():
                if any(' ' + normalize(term) + ' ' in text for term in terms):
                    keys[('speaker_month_anchor', speaker + ':' + month + ':' + anchor)].append(i)
    for (kind, key), ix in sorted(keys.items()):
        for j in ix[1:]:
            uf.union(ix[0], j)
            edges.append({'a': rows[ix[0]]['sample_id'], 'b': rows[j]['sample_id'], 'kind': kind, 'key_sha256': sha(key.encode())})
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[uf.find(i)].append(r['sample_id'])
    return list(groups.values()), edges, dup_map


def assign(rows, groups, dup_map, inventory, config, external_exposed=()):
    outcomes = set(f'train:{i}' for i in range(*config['legacy_exploratory_ordinals']))
    source_legacy = set(f'train:{i}' for i in range(*config['legacy_source_ordinals']))
    origin = defaultdict(list)
    inputs = set()
    for art in inventory['artifact_exposure']:
        if art['errors']:
            raise ValueError('Exposure scan has errors: ' + art['path'])
        if art['kind'] == 'claim_metadata_or_retrieval_input':
            inputs.update(art['sample_ids'])
        elif not any(s in art['path'].lower() for s in ('synthetic', 'smoke')):
            for sid in art['sample_ids']:
                # A's historical predictions are permitted source, not novel holdout.
                if sid not in source_legacy:
                    outcomes.add(sid)
                origin[sid].append(art['path'])
    external = set(external_exposed)
    by_id = {r['sample_id']: r for r in rows}
    records, blocks = [], []
    for ids in groups:
        ids = sorted(ids, key=lambda s: int(s.split(':')[1]))
        gid = 'grp-' + sha(canonical(ids))[:16]
        bad = [s for s in ids if not normalize(by_id[s]['claim'])]
        has_a = bool(set(ids) & source_legacy)
        has_exposure = bool(set(ids) & outcomes)
        has_external = bool(set(ids) & external)
        rank = int(sha((config['seed'] + ':' + gid).encode()), 16) / 2**256
        if bad:
            partition = 'QUARANTINE_INVALID'
        elif has_a:
            partition = 'SOURCE'
        elif has_exposure:
            partition = 'LEGACY_EXPLORATORY'
        else:
            partition = next(k for k, (lo, hi) in config['assignment_intervals'].items() if lo <= rank < hi)
        if has_external and partition == 'TEST_SEALED':
            partition = 'TEST_QUARANTINE_EXPOSED'
        block = {'group_id': gid, 'sample_ids': ids, 'partition': partition,
                 'legacy_A_present': has_a, 'outcome_exposed_component': has_exposure,
                 'external_execution_exposed_component': has_external,
                 'hash_rank': rank, 'claims': len(ids)}
        blocks.append(block)
        for sid in ids:
            is_a = sid in source_legacy
            exposed = sid in outcomes
            propagated = has_exposure or has_a
            memory_eligible = partition == 'SOURCE' and (is_a or not propagated)
            records.append({'sample_id': sid, 'official_split': 'train', 'group_id': gid,
                            'duplicate_group_id': dup_map[sid], 'partition': partition,
                            'claim_sha256': sha(by_id[sid]['claim'].encode()),
                            'claim_text_previously_exported': sid in inputs,
                            'artifact_outcome_exposed_directly': exposed or is_a,
                            'artifact_outcome_exposure_propagated': propagated,
                            'legacy_source_A': is_a, 'source_memory_eligible': memory_eligible,
                            'outcome_artifact_paths': sorted(origin[sid]),
                            'external_exposure_status': 'KNOWN_EXTERNAL_PREDICTION_EXPOSURE' if has_external else 'UNKNOWN',
                            'confirmatory_eligible': False,
                            'test_certification': 'NOT_TEST' if partition != 'TEST_SEALED' else
                              'SEALED_CANDIDATE_UNKNOWN_EXTERNAL_HISTORY'})
    return sorted(records, key=lambda r: int(r['sample_id'].split(':')[1])), sorted(blocks, key=lambda g: g['group_id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--external-exposure', type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output must be a new directory; never overwrite a sealed partition')
    inventory = json.loads((a.input / 'inventory.json').read_text(encoding='utf-8-sig'))
    config = json.loads(a.config.read_text(encoding='utf-8-sig'))
    assert inventory['dataset']['sha256'] == config['expected_train_sha256']
    rows = [json.loads(line) for line in (a.input / 'claims_metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    forbidden = {'label', 'gold_label', 'questions', 'answers', 'justification', 'claim_types', 'fact_checking_strategies'}
    assert all(not forbidden.intersection(r) for r in rows)
    assert len({r['sample_id'] for r in rows}) == len(rows) == inventory['dataset']['claims']
    groups, edges, dup_map = build_groups(rows, config)
    external = json.loads(a.external_exposure.read_text(encoding='utf-8-sig')) if a.external_exposure else None
    records, blocks = assign(rows, groups, dup_map, inventory, config, external['known_external_train_ids'] if external else ())
    by_id = {r['sample_id']: r for r in records}
    assert all(by_id[e['a']]['partition'] == by_id[e['b']]['partition'] for e in edges)
    assert all(not r['artifact_outcome_exposure_propagated'] for r in records if r['partition'] in ('DEVELOPMENT', 'TEST_SEALED'))
    assert all(not r['source_memory_eligible'] for r in records if r['partition'] != 'SOURCE')
    a.output.mkdir(parents=True)
    def write(name, data):
        (a.output / name).write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    write('claims_manifest.json', records)
    write('groups_manifest.json', blocks)
    write('group_edges.json', edges)
    write('protocol.json', config)
    for part in sorted({r['partition'] for r in records} | {'SOURCE', 'DEVELOPMENT', 'TEST_SEALED'}):
        write(part.lower() + '_manifest.json', [r for r in records if r['partition'] == part])
    counts = {part: {'claims': sum(r['partition'] == part for r in records),
                     'groups': sum(g['partition'] == part for g in blocks),
                     'source_memory_eligible': sum(r['partition'] == part and r['source_memory_eligible'] for r in records)}
              for part in sorted({r['partition'] for r in records} | {'SOURCE', 'DEVELOPMENT', 'TEST_SEALED'})}
    hashes = {f.name: sha(f.read_bytes()) for f in sorted(a.output.iterdir()) if f.is_file()}
    summary = {'protocol': config['protocol'], 'dataset_sha256': inventory['dataset']['sha256'],
               'inventory_sha256': sha((a.input / 'inventory.json').read_bytes()),
               'projection_sha256': sha((a.input / 'claims_metadata.jsonl').read_bytes()),
               'config_sha256': sha(a.config.read_bytes()), 'script_sha256': sha(Path(__file__).read_bytes()),
               'external_exposure_sha256': sha(a.external_exposure.read_bytes()) if a.external_exposure else None,
               'claims': len(records), 'duplicate_groups': len(set(dup_map.values())), 'leakage_blocks': len(blocks),
               'edge_counts': dict(Counter(e['kind'] for e in edges)), 'largest_block_claims': max(map(len, groups)),
               'partitions': counts, 'leakage_checks': {'cross_partition_edges': 0, 'test_or_dev_local_method_exposed_components': 0,
                                                       'test_or_dev_memory_eligible': 0, 'label_fields_in_projection': 0},
               'files_sha256': hashes, 'external_exposure_attestation': config['external_exposure_attestation'],
               'confirmation_ready': False, 'gpu_calls': 0,
               'certified_confirmatory_claims': 0, 'certified_confirmatory_groups': 0,
               'known_external_prediction_claims': len(external['known_external_train_ids']) if external else 0,
               'limits': ['Leakage blocks are operational groups, not proof of statistical independence.',
                          'All prior claim-text exports remain explicitly disclosed.',
                          'No gold labels accessed for grouping/assignment; no test class counts computed.',
                          'Candidate TEST reservoir sealed, confirmatory size and power decision deferred.']}
    write('seal.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
