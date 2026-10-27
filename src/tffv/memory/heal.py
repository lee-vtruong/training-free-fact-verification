"""Source-only HEAL memory and independent review contracts. No inference."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from tffv.provenance import digest, file_hash
from tffv.verification.ontology import VERDICTS, normalize_verdict
from tffv.verification.prompting import build_verification_prompt

PARTITION_SEAL = '53e1409c29a545d6d32fb40e2bd3e3f2ed814856a94607c106efc5a5f5a96be0'
SCHEDULE_HASH = '4c90f0ed1768f2d628ac23d0d7a3e0d3f73eb89ed333749e40d07d1a4a367b92'
MODEL_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
LEGACY_PREDICTIONS = '83d5b6a9f7828b51796691afcbc2c79211dc966a228e532cc5ffa11257e02af6'
LEGACY_EVIDENCE = 'f5e47540925d0b7e635d5c906aa12723d2fab04eb704449e83686375b5c11cb1'
STATUSES = ('PENDING_REVIEW', 'VERIFIED', 'REJECTED', 'INSUFFICIENT_EVIDENCE', 'PROVENANCE_UNKNOWN')
MECHANISMS = {
    'scope_quantifier': r'\b(percent|percentage|per capita|average|total|budget|allocated|allocation|less than|more than)\b|%',
    'evidence_sufficiency': r'\b(insufficient|not enough|lack|missing|does not provide|cannot determine)\b',
    'attribution': r'\b(said|says|stated|claimed|according to|quote)\b',
    'negation_polarity': r'\b(not|never|no|denied|without)\b',
    'reported_conflict': r'\b(disagree|disagreement|debate|dispute|conflicting)\b',
    'partial_truth_context': r'\b(context|partial|misleading|cherrypicking|cherry.picking)\b',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')


def indexed(rows):
    result = {}
    for r in rows:
        if r['sample_id'] in result:
            raise ValueError('Duplicate sample ID: ' + r['sample_id'])
        result[r['sample_id']] = r
    return result


class SourceContract:
    def __init__(self, project):
        self.project = Path(project).resolve()
        folder = self.project / 'docs/partitions/m16-v2'
        if file_hash(folder / 'seal.json') != PARTITION_SEAL:
            raise ValueError('Partition v2 seal identity changed')
        self.seal = read(folder / 'seal.json')
        for name, h in self.seal['files_sha256'].items():
            if file_hash(folder / name) != h:
                raise ValueError('Partition artifact changed: ' + name)
        self.rows = indexed(read(folder / 'claims_manifest.json'))
        schedule = self.project / 'docs/partitions/m16-execution-plan-v1/phase_schedule.json'
        if file_hash(schedule) != SCHEDULE_HASH:
            raise ValueError('Fixed schedule changed')
        self.schedule = read(schedule)
        if self.schedule['parent_seal_sha256'] != PARTITION_SEAL:
            raise ValueError('Schedule parent changed')
        self.legacy_ids = [f'train:{i}' for i in range(50)]
        self.batch_ids = self.schedule['source_expansion_order'][:25]
        self.require(self.legacy_ids + self.batch_ids)
        if len(set(self.batch_ids)) != 25:
            raise ValueError('Batch must be exactly 25 unique fixed source IDs')

    def require(self, ids):
        if len(ids) != len(set(ids)):
            raise ValueError('Repeated source IDs')
        for sid in ids:
            r = self.rows.get(sid, {})
            if r.get('partition') != 'SOURCE' or r.get('source_memory_eligible') is not True:
                raise ValueError('Not eligible SOURCE memory: ' + sid)

    def check_memory(self, memory):
        if memory['partition_seal_sha256'] != PARTITION_SEAL:
            raise ValueError('Memory partition mismatch')
        self.require([r['sample_id'] for r in memory['records']])
        for r in memory['records']:
            if r['group_id'] != self.rows[r['sample_id']]['group_id']:
                raise ValueError('Memory group mismatch')
            if hashlib.sha256(r['claim'].encode()).hexdigest() != self.rows[r['sample_id']]['claim_sha256']:
                raise ValueError('Memory claim differs from sealed source')
            validate_record(r)


def source_gold(path, ids, contract):
    """Offline source-only join. Never copy QA, justifications or other labels."""
    contract.require(ids)
    if file_hash(path) != contract.seal['dataset_sha256']:
        raise ValueError('Gold source release/hash mismatch')
    data = read(path)
    result = {}
    for sid in ids:
        row = data[int(sid.split(':')[1])]
        result[sid] = {'claim': row['claim'].strip(), 'label': row['label']}
        if result[sid]['label'] not in VERDICTS:
            raise ValueError('Unexpected source gold ontology')
    return result


def refresh(r):
    r['content_sha256'] = digest({k:v for k,v in r.items() if k != 'content_sha256'})
    return r


def candidate_hash(r):
    obj={k:r[k] for k in ('record_id','corrected_explanation','citations','candidate_mechanisms')}
    if 'draft_assessment' in r: obj['draft_assessment']=r['draft_assessment']
    return digest(obj)


def check_citations(r):
    ev = {e['passage_id']: e for e in r['evidence']}
    if not r['corrected_explanation'] or not r['citations']:
        raise ValueError('Correction requires nonempty explanation and exact evidence citations')
    for c in r['citations']:
        text = ev[c['passage_id']]['rendered_text']
        a, b = c['start'], c['end']
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text) or text[a:b] != c['quote']:
            raise ValueError('Citation span/quote not in exact displayed evidence')


def review_status(r):
    if not r['provenance']['artifact_checks_passed']:
        return 'PROVENANCE_UNKNOWN'
    if not r['evidence']:
        return 'INSUFFICIENT_EVIDENCE'
    reviews = r['reviews']
    if len(reviews) < 2:
        if r.get('draft_assessment',{}).get('status')=='INSUFFICIENT_EVIDENCE':
            return 'INSUFFICIENT_EVIDENCE'
        return 'PENDING_REVIEW'
    if len(reviews) != 2 or len({v['reviewer_id'].casefold() for v in reviews}) != 2:
        raise ValueError('Exactly two distinct human reviewers required')
    for v in reviews:
        if v['reviewer_type'] != 'human' or v['independent_review'] is not True:
            raise ValueError('Human/independence attestation required')
        if not isinstance(v['reviewer_id'],str) or not v['reviewer_id'].strip() or not isinstance(v['reason'],str) or not v['reason'].strip():
            raise ValueError('Review identity/reason required')
        if v.get('failure_category') not in ('REASONING_FAILURE_SUPPORTED_BY_EVIDENCE','INCOMPLETE_RETRIEVAL_EVIDENCE','PARSER_ERROR','BASE_CORRECT','OTHER','UNKNOWN'):
            raise ValueError('Completed review requires an explicit failure category')
        stamp = datetime.fromisoformat(v['reviewed_at'].replace('Z','+00:00'))
        if stamp.tzinfo is None:
            raise ValueError('Review timestamp must include timezone')
        if v['candidate_sha256'] != candidate_hash(r) or v['record_sha256'] != r['review_record_sha256']:
            raise ValueError('Stale/different candidate review')
        if v['decision'] not in ('VERIFIED','REJECTED','INSUFFICIENT_EVIDENCE'):
            raise ValueError('Invalid completed review decision')
        if not set(v['confirmed_mechanisms']).issubset(r['candidate_mechanisms']):
            raise ValueError('Review confirms a mechanism outside the candidate revision')
    if any(v['decision'] == 'INSUFFICIENT_EVIDENCE' for v in reviews):
        return 'INSUFFICIENT_EVIDENCE'
    if any(v['decision'] == 'REJECTED' for v in reviews):
        return 'REJECTED'
    if r['failure_classification']=='PARSER_ERROR':
        raise ValueError('Parser failures are engineering cases, not verified rationale exemplars')
    check_citations(r)
    if not all(v.get(k) is True for v in reviews for k in ('evidence_sufficient','correction_grounded','gold_relation_supported')):
        raise ValueError('VERIFIED requires both reviewers to affirm grounding and gold relation')
    return 'VERIFIED'


def validate_record(r):
    def reject_annotations(value):
        if isinstance(value,dict):
            if set(value)&{'questions','answers','gold_justification','justification','gold_qa'}:
                raise ValueError('Gold QA/justification fields are prohibited in memory')
            for v in value.values(): reject_annotations(v)
        elif isinstance(value,list):
            for v in value: reject_annotations(v)
    reject_annotations(r)
    if r['schema_version'] != 'heal-memory-v1' or r['review_status'] not in STATUSES:
        raise ValueError('Memory schema/status mismatch')
    if r['content_sha256'] != digest({k:v for k,v in r.items() if k != 'content_sha256'}):
        raise ValueError('Memory record hash mismatch')
    identity=digest({'sid':r['sample_id'],'claim':r['claim'],'gold':r['gold_verdict'],
                     'base':r['base_prediction'],'evidence':r['evidence'],'provenance':r['provenance']})
    if r['record_id']!=identity:
        raise ValueError('Immutable source record identity changed')
    if r['gold_verdict'] not in VERDICTS:
        raise ValueError('Invalid gold ontology')
    if normalize_verdict(r['base_prediction']['raw_generation']) != r['base_prediction']['reparsed_verdict']:
        raise ValueError('Raw parser mismatch')
    base=r['base_prediction']
    if digest(base['raw_generation'])!=base['raw_generation_sha256']:
        raise ValueError('Raw generation hash mismatch')
    parser_error=base['reparsed_verdict']!=base['stored_verdict'] or base['reparsed_verdict'] not in VERDICTS
    expected_error=not parser_error and base['reparsed_verdict']!=r['gold_verdict']
    if r['is_base_error']!=expected_error:
        raise ValueError('Incorrect/error flag inconsistent with raw parse and gold')
    expected_class='PARSER_ERROR' if parser_error else ('BASE_INCORRECT_VERDICT_UNDIAGNOSED' if expected_error else 'BASE_CORRECT')
    if r['failure_classification']!=expected_class or r['confusion_type']!=(f'{base["reparsed_verdict"]} -> {r["gold_verdict"]}' if base['reparsed_verdict'] in VERDICTS and base['reparsed_verdict']!=r['gold_verdict'] else None):
        raise ValueError('Failure/confusion classification inconsistent with raw verdict')
    if r['review_status'] != review_status(r):
        raise ValueError('Review status cannot be assigned without required evidence/reviews')
    if not set(r['candidate_mechanisms']).issubset(MECHANISMS):
        raise ValueError('Unknown candidate mechanism')
    if 'draft_assessment' in r:
        d=r['draft_assessment']
        if d['author_type']!='assistant' or d['status'] not in ('PENDING_REVIEW','INSUFFICIENT_EVIDENCE') or not d['limitations'] or not d['reason']:
            raise ValueError('Invalid non-human source drafting assessment')
        if set(d['evidence_inventory_passage_ids'])!={e['passage_id'] for e in r['evidence']}:
            raise ValueError('Draft evidence inventory changed')
        if d['status']=='INSUFFICIENT_EVIDENCE' and r['corrected_explanation'] is not None:
            raise ValueError('Do not fabricate a gold correction for insufficient evidence')
    if len({e['passage_id'] for e in r['evidence']})!=len(r['evidence']):
        raise ValueError('Duplicate evidence passage IDs')
    for e in r['evidence']:
        if digest(e['text']) != e['text_sha256'] or digest(e['rendered_text']) != e['rendered_text_sha256']:
            raise ValueError('Evidence text hash mismatch')
        if r['provenance']['artifact_checks_passed'] and e.get('original_text_sha256')!=digest(e['text']):
            raise ValueError('Original evidence fingerprint is not verified')


def make_record(sid, claim, gold, prediction, evidence, provenance, contract):
    raw = prediction.get('raw_generation', prediction.get('raw_text', ''))
    parsed = normalize_verdict(raw)
    stored = prediction.get('predicted_verdict')
    parser_error = parsed != stored or parsed not in VERDICTS
    wrong = parsed in VERDICTS and parsed != gold
    evs = []
    for e in evidence:
        rendered = e['text'][:1200] + ('...' if len(e['text']) > 1200 else '')
        evs.append({**e, 'text_sha256': digest(e['text']), 'rendered_text': rendered,
                    'rendered_text_sha256': digest(rendered), 'original_text_sha256': e.get('text_sha256')})
    candidate_text = claim + '\n' + raw
    r = {'schema_version':'heal-memory-v1', 'sample_id':sid, 'claim':claim,
         'group_id':contract.rows[sid]['group_id'], 'evidence':evs,
         'base_prediction':{'stored_verdict':stored, 'reparsed_verdict':parsed, 'raw_generation':raw,
                            'raw_generation_sha256':digest(raw), 'prompt_sha256':prediction.get('prompt_sha256'),
                            'prompt_tokens':prediction.get('prompt_tokens'), 'generated_tokens':prediction.get('generated_tokens'),
                            'latency_seconds':prediction.get('latency_seconds'), 'max_new_tokens':prediction.get('max_new_tokens'),
                            'enable_thinking':prediction.get('enable_thinking')},
         'gold_verdict':gold, 'is_base_error':wrong and not parser_error,
         'confusion_type':f'{parsed} -> {gold}' if wrong else None,
         'failure_classification':'PARSER_ERROR' if parser_error else ('BASE_INCORRECT_VERDICT_UNDIAGNOSED' if wrong else 'BASE_CORRECT'),
         'candidate_mechanisms':[k for k,pattern in MECHANISMS.items() if re.search(pattern,candidate_text,re.I)] if wrong else [],
         'corrected_explanation':None, 'citations':[], 'reviews':[], 'review_record_sha256':None,
         'provenance':provenance, 'review_status':'PENDING_REVIEW'}
    r['record_id'] = digest({'sid':sid,'claim':claim,'gold':gold,'base':r['base_prediction'],'evidence':evs,'provenance':provenance})
    r['review_status'] = review_status(r)
    return refresh(r)


def summary(memory):
    rr = memory['records']
    verified = [r for r in rr if r['review_status']=='VERIFIED']
    wrong = [r for r in rr if r['is_base_error']]
    support = Counter()
    reasoning = 0
    for r in verified:
        if r['is_base_error']:
            support.update(set.intersection(*(set(v['confirmed_mechanisms']) for v in r['reviews'])))
            reasoning += int(all(v.get('failure_category')=='REASONING_FAILURE_SUPPORTED_BY_EVIDENCE' for v in r['reviews']))
    costs={}
    for key in ('prompt_tokens','generated_tokens','latency_seconds'):
        values=[r['base_prediction'].get(key) for r in rr]
        costs[key]=sum(values) if all(v is not None for v in values) else None
    return {'source_records':len(rr), 'base_incorrect_verdicts':len(wrong),
            'base_correct_verdicts':sum(r['failure_classification']=='BASE_CORRECT' for r in rr),
            'parser_errors':sum(r['failure_classification']=='PARSER_ERROR' for r in rr),
            'statuses':dict(Counter(r['review_status'] for r in rr)),
            'artifact_chain_checked_records':sum(r['provenance']['artifact_checks_passed'] for r in rr),
            'verified_corrective_rationales':sum(r['is_base_error'] for r in verified),
            'verified_correct_control_rationales':sum(r['failure_classification']=='BASE_CORRECT' for r in verified),
            'verified_reasoning_failures':reasoning, 'verified_mechanism_support':dict(support),
            'candidate_mechanism_flags_not_verified':dict(Counter(k for r in wrong for k in r['candidate_mechanisms'])),
            'incomplete_evidence_cases':sum(r['review_status']=='INSUFFICIENT_EVIDENCE' for r in rr),
            'evidence_sufficiency_unreviewed':sum(len(r['reviews'])<2 for r in rr),
            'assistant_insufficient_screenings':sum(r.get('draft_assessment',{}).get('status')=='INSUFFICIENT_EVIDENCE' for r in rr),
            'human_reviews':sum(len(r['reviews']) for r in rr),
            'new_gpu_calls_during_memory_build':0,
            'completed_new_source_predictions':sum(r['provenance']['origin']=='OWN_PROJECT_M161_BATCH25' for r in rr),
            'retained_prediction_costs_not_new_build_costs':costs,
            'source_gpu_active_seconds':None if any(r['provenance']['origin']=='OWN_PROJECT_M161_BATCH25' for r in rr) else 0}


def build_legacy(contract, predictions_path, prediction_manifest, evidence_path, retrieval_manifest, gold_path):
    pm, rm = read(prediction_manifest), read(retrieval_manifest)
    hashes = {str(Path(p)):file_hash(p) for p in (predictions_path,prediction_manifest,evidence_path,retrieval_manifest,gold_path)}
    if file_hash(predictions_path) != LEGACY_PREDICTIONS or pm['predictions_sha256'] != LEGACY_PREDICTIONS:
        raise ValueError('Canonical own-project legacy prediction hash mismatch')
    if file_hash(evidence_path) != LEGACY_EVIDENCE or pm['r4_evidence_sha256'] != LEGACY_EVIDENCE or rm['outputs_sha256']['R4.jsonl'] != LEGACY_EVIDENCE:
        raise ValueError('Legacy R4 evidence chain mismatch')
    model = pm['model_spec']
    cfg = pm['generation_config']
    if model['model_id']!='Qwen/Qwen3-8B' or model['revision']!=MODEL_REVISION or pm['synthetic'] is not False or pm['gold_labels_used'] is not False:
        raise ValueError('Legacy frozen model/protocol mismatch')
    if cfg['max_new_tokens']!=512 or cfg['enable_thinking'] is not False or cfg['do_sample'] is not False:
        raise ValueError('Legacy D0/B1 decoder mismatch')
    if digest(rm['config']) != rm['config_sha256'] or rm['gold_labels_used'] is not False:
        raise ValueError('Retrieval configuration provenance mismatch')
    ids = contract.legacy_ids
    gold = source_gold(gold_path,ids,contract)
    preds, evidence = indexed(jsonl(predictions_path)), indexed(jsonl(evidence_path))
    records, failures = [], {}
    for sid in ids:
        p, e, g = preds[sid], evidence[sid], gold[sid]
        checks = {'claim':p['claim_sha256']==digest(g['claim'])==e['claim_sha256'],
                  'prompt':build_verification_prompt(g['claim'],e['evidence'])['prompt_sha256']==p['prompt_sha256'],
                  'passage_ids':p['evidence_passage_ids']==[v['passage_id'] for v in e['evidence']],
                  'source_urls':p['evidence_source_urls']==[v['source_url'] for v in e['evidence']],
                  'evidence_hashes':all(v['text_sha256']==digest(v['text']) for v in e['evidence']),
                  'decoder':p['max_new_tokens']==512 and p['enable_thinking'] is False and p['gold_accessed'] is False}
        if not all(checks.values()):
            failures[sid] = [k for k,v in checks.items() if not v]
        prov = {'origin':'OWN_PROJECT_M121_B1', 'artifact_checks_passed':all(checks.values()),
                'checks':checks, 'source_files_sha256':hashes, 'declared_model':model,
                'declared_generation_config':cfg, 'retrieval_config':rm['config'],
                'runtime_weight_fingerprint':None, 'effective_input_token_ids_sha256':None,
                'raw_store_bytes_reaudited':False,'original_document_offsets_reaudited':False,
                'limitation':'Serialized historical artifacts checked; original runtime weight/effective-token fingerprints absent. Raw stores/document offsets not re-audited here. No reconstructed data claimed as original.'}
        records.append(make_record(sid,g['claim'],g['label'],p,e['evidence'],prov,contract))
    memory = {'schema_version':'heal-bank-v1', 'partition_seal_sha256':PARTITION_SEAL,
              'origin':'OWN_PROJECT_LEGACY_FOLD_A', 'records':records,
              'source_file_hashes':hashes, 'legacy_check_failures':failures}
    contract.check_memory(memory)
    return memory


def apply_candidates(memory, annotations, contract):
    result = deepcopy(memory)
    result['parent_bank_sha256']=[digest(memory)]
    lookup = indexed(result['records'])
    for a in annotations:
        r = lookup[a['sample_id']]
        if a['record_sha256'] != r['content_sha256'] or r['reviews']:
            raise ValueError('Stale candidate update or reviewed revision: create a new revision explicitly')
        r['corrected_explanation'] = a['corrected_explanation']
        r['citations'] = a['citations']
        r['candidate_mechanisms'] = a.get('candidate_mechanisms',r['candidate_mechanisms'])
        if 'draft_assessment' in a: r['draft_assessment']=a['draft_assessment']
        check_citations(r)
        r['review_record_sha256'] = None
        r['review_status']=review_status(r)
        refresh(r)
    contract.check_memory(result)
    return result


def export_reviews(memory, destination):
    dest = Path(destination)
    dest.mkdir(parents=True,exist_ok=False)
    for slot in (1,2):
        sheets = []
        for r in memory['records']:
            sheets.append({'sample_id':r['sample_id'], 'record_sha256':r['content_sha256'],
                           'candidate_sha256':candidate_hash(r), 'claim':r['claim'],
                           'base_prediction':r['base_prediction'], 'gold_verdict':r['gold_verdict'],
                           'evidence':r['evidence'], 'corrected_explanation':r['corrected_explanation'],
                           'citations':r['citations'], 'candidate_mechanisms':r['candidate_mechanisms'],
                           'reviewer_id':None, 'reviewer_type':None, 'independent_review':None,
                           'reviewed_at':None,'decision':None,'reason':None,
                           'evidence_sufficient':None,'correction_grounded':None,'gold_relation_supported':None,
                           'failure_category':None,'confirmed_mechanisms':[]})
            if 'draft_assessment' in r: sheets[-1]['draft_assessment']=r['draft_assessment']
        save(dest / f'reviewer_{slot}.json',sheets)
    save(dest / 'candidate_annotations_template.json',[
        {'sample_id':r['sample_id'],'record_sha256':r['content_sha256'],
         'corrected_explanation':None,'citations':[],'candidate_mechanisms':r['candidate_mechanisms']}
        for r in memory['records']])


def import_reviews(memory, first, second, contract):
    result = deepcopy(memory)
    result['parent_bank_sha256']=[digest(memory)]
    a, b = indexed(first), indexed(second)
    expected = {r['sample_id'] for r in memory['records']}
    if set(a)!=expected or set(b)!=expected:
        raise ValueError('Review files must cover the exact same sealed record IDs; no drops/additions')
    for sheet in (first,second):
        completed={v.get('reviewer_id') for v in sheet if v.get('decision') is not None}
        if len(completed)>1:
            raise ValueError('One human identity per independent review sheet')
    fields = ('reviewer_id','reviewer_type','independent_review','reviewed_at','decision','reason',
              'evidence_sufficient','correction_grounded','gold_relation_supported','failure_category','confirmed_mechanisms',
              'record_sha256','candidate_sha256')
    for r in result['records']:
        pair = (a[r['sample_id']],b[r['sample_id']])
        for v in pair:
            for key in ('claim','base_prediction','gold_verdict','evidence','corrected_explanation','citations','candidate_mechanisms'):
                if v[key]!=r[key]:
                    raise ValueError('Review sheet source/candidate context was edited: '+key)
            if v.get('draft_assessment')!=r.get('draft_assessment'):
                raise ValueError('Review sheet drafting limitations/status were edited')
        if any(v['record_sha256']!=r['content_sha256'] or v['candidate_sha256']!=candidate_hash(r) for v in pair):
            raise ValueError('Review sheet pins a stale or different source/candidate')
        if all(v.get('decision') is None for v in pair):
            continue
        if any(v.get('decision') is None for v in pair):
            raise ValueError('Incomplete two-reviewer decision; retain pending bank')
        r['review_record_sha256'] = r['content_sha256']
        r['reviews'] = [{k:v[k] for k in fields} for v in pair]
        r['review_status'] = review_status(r)
        refresh(r)
    result['review_input_sha256'] = [digest(first),digest(second)]
    contract.check_memory(result)
    return result


def example_view(memory, ids, mode, contract, mechanism=None):
    """Explicit-ID future ICL interface; no target query, retrieval or generation."""
    contract.check_memory(memory)
    records = indexed(memory['records'])
    contract.require(ids)
    if mode not in ('gold_only','wrong_gold','wrong_gold_verified','contrastive','mechanism'):
        raise ValueError('Unknown memory view')
    selected = [records[sid] for sid in ids]
    if mode=='contrastive' and (len(selected)!=2 or {r['is_base_error'] for r in selected}!={True,False}):
        raise ValueError('Contrastive view needs exactly one correct and one incorrect source record')
    out = []
    for r in selected:
        if r['failure_classification']=='PARSER_ERROR' or not r['provenance']['artifact_checks_passed'] or r['review_status'] in ('REJECTED','INSUFFICIENT_EVIDENCE','PROVENANCE_UNKNOWN'):
            raise ValueError('Parser/provenance failures cannot become demonstrations')
        if mode in ('wrong_gold','wrong_gold_verified') and not r['is_base_error']:
            raise ValueError('Wrong view requires an actual incorrect verdict')
        x = {'sample_id':r['sample_id'],'record_id':r['record_id'],'claim':r['claim'],
             'evidence':r['evidence'],'gold_verdict':r['gold_verdict']}
        if mode != 'gold_only':
            x['base_verdict'] = r['base_prediction']['reparsed_verdict']
        if mode in ('wrong_gold_verified','mechanism'):
            if r['review_status']!='VERIFIED':
                raise ValueError('Verified-rationale view requires two completed human reviews')
            if mode=='mechanism' and (mechanism is None or not all(mechanism in v['confirmed_mechanisms'] for v in r['reviews'])):
                raise ValueError('Mechanism lacks agreement from both reviewers')
            x.update(corrected_explanation=r['corrected_explanation'],citations=r['citations'])
        out.append(x)
    return out


def merge_memories(memories,contract):
    records=[]
    for m in memories:
        contract.check_memory(m)
        records.extend(deepcopy(m['records']))
    indexed(records)  # No silent replacement of repeated source IDs/review revisions.
    result={'schema_version':'heal-bank-v1','partition_seal_sha256':PARTITION_SEAL,
            'origin':'OWN_PROJECT_COMBINED_SOURCE','parent_bank_sha256':[digest(m) for m in memories],
            'records':records}
    contract.check_memory(result)
    return result
