"""Deterministic replay of source-only assistant drafting decisions, never human review."""
from copy import deepcopy
from pathlib import Path
import json

from tffv.memory.heal import (SourceContract, read, save, indexed, refresh, check_citations,
                            review_status, export_reviews, summary)
from tffv.provenance import digest,file_hash


def complete(memory,decisions,contract):
    contract.check_memory(memory)
    expected=set(contract.legacy_ids+contract.batch_ids)
    if {r['sample_id'] for r in memory['records']}!=expected:
        raise ValueError('Correction audit requires the exact own-project source75 cohort')
    wrong={r['sample_id'] for r in memory['records'] if r['is_base_error']}
    if set(decisions['errors'])!=wrong:
        raise ValueError('Draft decisions must audit every incorrect source ID exactly once')
    result=deepcopy(memory)
    result['parent_bank_sha256']=[digest(memory)]
    result['source_drafting_decisions_sha256']=digest(decisions)
    for r in result['records']:
        if r['reviews']:
            raise ValueError('Do not replace an already human-reviewed revision')
        if not r['is_base_error']: continue
        d=decisions['errors'][r['sample_id']]
        if d['record_id']!=r['record_id']:
            raise ValueError('Source draft belongs to different evidence/prediction provenance')
        r['draft_assessment']={'author_type':'assistant','status':d['status'],'reason':d['reason'],
            'limitations':d['limitations'],'human_reviews':0,
            'evidence_inventory_passage_ids':[e['passage_id'] for e in r['evidence']]}
        r['corrected_explanation']=d.get('explanation')
        r['candidate_mechanisms']=d.get('mechanisms',r['candidate_mechanisms'])
        r['citations']=[]
        for selector in d.get('citations',[]):
            e=r['evidence'][selector['passage_index']]
            text=e['rendered_text'];quote=selector['quote']
            if text.count(quote)!=1: raise ValueError('Citation selector is absent or ambiguous')
            start=text.index(quote)
            r['citations'].append({'passage_id':e['passage_id'],'start':start,'end':start+len(quote),'quote':quote})
        if d['status']=='PENDING_REVIEW': check_citations(r)
        elif d['status']!='INSUFFICIENT_EVIDENCE': raise ValueError('Assistant cannot assign VERIFIED/REJECTED human decisions')
        r['review_status']=review_status(r)
        r['review_record_sha256']=None
        refresh(r)
    contract.check_memory(result)
    return result


def run(memory_path,decisions_path,output,project):
    output=Path(output).resolve()
    if not output.is_relative_to(Path(project).resolve()/'outputs'):
        raise ValueError('Use new project outputs/')
    contract=SourceContract(project)
    result=complete(read(memory_path),read(decisions_path),contract)
    output.mkdir(parents=True,exist_ok=False)
    save(output/'memory75.json',result)
    export_reviews(result,output/'reviews75')
    packets=output/'review_packets';packets.mkdir()
    for r in result['records']:
        lines=[f"# Source review packet: {r['sample_id']}",
            'ASSISTANT PREPARATION ONLY. No human decision is represented here.',
            f"Claim: {r['claim']}",f"Base verdict: {r['base_prediction']['reparsed_verdict']}",
            f"Source gold: {r['gold_verdict']}",f"Current disposition: {r['review_status']}",
            f"Immutable record ID: {r['record_id']}",f"Candidate/content hash: {r['content_sha256']}",
            '## Exact Base generation',r['base_prediction']['raw_generation'],
            '## Candidate explanation (UNVERIFIED)',r['corrected_explanation'] or '(none)',
            '## Assistant limitations / disposition',str(r.get('draft_assessment',{'status':'Control evidence sufficiency remains human-unreviewed'})),
            '## Exact cited spans (Unicode offsets, end-exclusive)',json.dumps(r['citations'],ensure_ascii=False,indent=2),
            '## All original rendered Top-8 evidence']
        for e in r['evidence']:
            lines.extend([f"### {e['passage_id']}",f"URL: {e['source_url']}",
                          f"Rendered SHA-256: {e['rendered_text_sha256']}",e['rendered_text']])
        lines.extend(['## Retained provenance (limitations included)',json.dumps(r['provenance'],ensure_ascii=False,indent=2),
            'Review independently using reviewer_1.json or reviewer_2.json. Inspect all evidence, not only cited spans. '
            'Do not label VERIFIED if the retrieved context cannot establish the source gold relation. '
            'Record actual identity, timestamp, grounding judgments and reason; never copy another reviewer.'])
        (packets/(r['sample_id'].replace(':','-')+'.md')).write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    report={'summary':summary(result),'input_memory_sha256':file_hash(memory_path),
            'decisions_sha256':file_hash(decisions_path),'output_memory_sha256':file_hash(output/'memory75.json'),
            'eligible_pending_error_ids':[r['sample_id'] for r in result['records'] if r['is_base_error'] and r['review_status']=='PENDING_REVIEW'],
            'insufficient_error_ids':[r['sample_id'] for r in result['records'] if r['is_base_error'] and r['review_status']=='INSUFFICIENT_EVIDENCE'],
            'correct_controls_audited':sum(not r['is_base_error'] for r in result['records']),
            'control_evidence_sufficiency':'HUMAN_REVIEW_PENDING; structural/raw/provenance checks only',
            'human_reviews_created':0}
    save(output/'audit.json',report)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory',type=Path,required=True);p.add_argument('--decisions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[3])
    a=p.parse_args();print(json.dumps(run(a.memory,a.decisions,a.output,a.project),indent=2))
