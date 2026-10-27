"""Prospective stage costs from historical project timing, not new GPU evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    prov = json.loads((ROOT / 'docs/audits/M1_6_RESOURCE_PROVENANCE.json').read_text(encoding='utf-8-sig'))
    timed = {r['path']: r for r in prov['timings']}
    cold_paths = [p for p in timed if any(s in p for s in ('pilot100','foldc','foldd'))]
    cold = [timed[p]['elapsed_seconds'] / timed[p]['methods']['R4']['n_claims'] for p in cold_paths]
    low, high = min(cold), max(cold)
    scenarios = []
    for stage, n, calls in [('source_initial',25,25), ('source_example_50',50,50),
                           ('source_example_100',100,100), ('pilot_initial_nine_arms',50,450),
                           ('development_upper_example',200,2800), ('confirmation_per_100_claims',100,400)]:
        # 1.5-5.2 seconds/call reflects historical concise/D0 means only;
        # larger inputs/CoT can be slower. Loading and variability not measured here.
        scenarios.append({'stage':stage, 'new_retrieval_claims_assumed':n, 'verifier_calls_assumed':calls,
                          'retrieval_wall_hours_range':[n*low/3600,n*high/3600],
                          'verifier_wall_hours_range':[calls*1.5/3600,calls*5.2/3600],
                          'planning_total_hours_range':[n*low/3600+calls*1.5/3600,n*high/3600+calls*5.2/3600],
                          'conservative_reservation_hours':2*(n*high+calls*5.2)/3600+0.25,
                          'input_tokens_assumed_range':[calls*2500,calls*3500],
                          'output_tokens_assumed_range':[calls*64,calls*200],
                          'output_tokens_hard_ceiling_at_512':calls*512})
    print(json.dumps({'status':'ESTIMATES_ONLY_NO_NEW_GPU_RUN',
                      'cold_seconds_per_claim_range':[low,high], 'cold_reference_paths':cold_paths,
                      'cached_reference_seconds_per_claim':timed['outputs/m15-folde-r4/summary.json']['elapsed_seconds']/100,
                      'cache_reuse_verified_for_new_groups':False,
                      'assumptions':['Historical end-to-end time includes CPU/I/O and GPU; not measured active GPU-hours.',
                                     'Input/output token counts are planning assumptions, not new measurements.',
                                     'No 64-token cap; every core arm retains 512-token cap. E5 may take longer.',
                                     'Confirmation scenario is per-100 scaling only; eligible current train test N=0.'],
                      'budgets_proposed_hours':{'source_gpu_session':12,'development_gpu_session':24,'confirmation':None},
                      'money_cost':None, 'new_measured_gpu_hours':0,'scenarios':scenarios},indent=2))


if __name__ == '__main__':
    main()
