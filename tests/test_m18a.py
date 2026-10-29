"""CPU-only unit test suite for M1.8a Full-Train Historical Memory Construction."""
import json
from pathlib import Path
import pytest

from tffv.experiments.m18a import (
    audit_full_train_integrity,
    scan_cache_inventory,
    build_outcome_bank,
    generate_candidate_corrections,
    build_expanded_memory,
    verify_official_dev_readiness,
)
from tffv.verification.ontology import VERDICTS


def test_full_train_integrity_manifest(tmp_path):
    manifest = audit_full_train_integrity(output_dir=tmp_path)
    assert manifest['schema_version'] == 'm18a-train-manifest-v1'
    assert manifest['total_claims'] == 3068
    assert manifest['valid_claims'] == 3067
    assert manifest['invalid_claims'] == 1
    assert manifest['invalid_sample_ids'] == ['train:1948']

    # Label distribution & ontology
    labels = manifest['gold_label_counts']
    assert set(labels.keys()) == set(VERDICTS)
    assert sum(labels.values()) == 3068

    # Partition preservation
    partitions = manifest['partition_counts']
    assert partitions['DEVELOPMENT'] == 541
    assert partitions['SOURCE'] == 1013
    assert partitions['TEST_SEALED'] == 1230

    # Cross-split overlap
    overlap = manifest['train_dev_overlap']
    assert overlap['exact_normalized_text_matches'] == 4
    assert overlap['fact_checking_url_matches'] == 3

    # Check written file
    saved = json.loads((tmp_path / 'train_manifest.json').read_text(encoding='utf-8'))
    assert saved['total_claims'] == 3068


def test_cache_inventory_invariants(tmp_path):
    inv = scan_cache_inventory(output_dir=tmp_path)
    assert inv['schema_version'] == 'm18a-cache-inventory-v1'
    assert inv['total_valid_train_claims'] == 3067
    assert inv['reusable_r4_evidence_count'] + inv['needs_retrieval_count'] == 3067
    assert inv['reusable_base_predictions_count'] + inv['needs_base_inference_count'] == 3067

    r4_set = set(inv['reusable_r4_sample_ids'])
    need_r4_set = set(inv['needs_retrieval_sample_ids'])
    assert r4_set.isdisjoint(need_r4_set)
    assert len(r4_set | need_r4_set) == 3067

    base_set = set(inv['reusable_base_sample_ids'])
    need_base_set = set(inv['needs_base_sample_ids'])
    assert base_set.isdisjoint(need_base_set)
    assert len(base_set | need_base_set) == 3067


def test_outcome_bank_metrics_and_separation(tmp_path):
    bank = build_outcome_bank(output_dir=tmp_path)
    assert bank['schema_version'] == 'm18a-outcome-bank-v1'
    total = bank['evaluated_claims_count']
    assert total > 0
    assert bank['correct_count'] + bank['wrong_count'] == total

    # Check disjointness of pools
    correct_sids = {r['sample_id'] for r in bank['correct_records']}
    wrong_sids = {r['sample_id'] for r in bank['wrong_records']}
    assert correct_sids.isdisjoint(wrong_sids)

    # Accuracy consistency
    expected_acc = round(bank['correct_count'] / total, 6)
    assert abs(bank['base_accuracy'] - expected_acc) < 1e-4

    # All wrong claims must have base != gold
    for r in bank['wrong_records']:
        assert r['base_verdict'] != r['gold_verdict']
        assert r['is_correct'] is False

    # All correct claims must have base == gold
    for r in bank['correct_records']:
        assert r['base_verdict'] == r['gold_verdict']
        assert r['is_correct'] is True


def test_candidate_corrections_schema(tmp_path):
    cands = generate_candidate_corrections(output_dir=tmp_path)
    assert len(cands) > 0
    for c in cands:
        assert 'sample_id' in c
        assert 'claim' in c and c['claim']
        assert 'gold_verdict' in c and c['gold_verdict'] in VERDICTS
        assert 'base_verdict' in c and c['base_verdict'] in VERDICTS
        assert c['gold_verdict'] != c['base_verdict']
        assert 'draft_text' in c and len(c['draft_text']) > 10
        assert 'mechanism' in c


def test_expanded_memory_invariants(tmp_path):
    mem = build_expanded_memory(output_dir=tmp_path)
    assert mem['schema_version'] == 'm18a-expanded-memory-v1'
    assert mem['scope'] == 'FULL_TRAIN_HISTORICAL_ERROR_MEMORY'
    assert mem['human_review_status'] == 'UNCHANGED_NOT_HUMAN_VERIFIED'
    assert mem['memory_status'] == 'LLM_VALIDATED'
    assert mem['total_records'] > 0

    cfg = mem['heal_b_locked_configuration']
    assert cfg['k'] == 1
    assert cfg['selection_rule'] == 'deterministic_within_class_rotation'
    assert cfg['prompt_template'] == 'm162e_standard_prompt'
    assert cfg['token_budget_overhead_max'] == 768
    assert cfg['arms'] == ['D0', 'D1', 'N0', 'A', 'B', 'C', 'D']

    # Verify strictly NO dev claims are in the memory
    for r in mem['records']:
        assert not r['sample_id'].startswith('dev:')


def test_dev_readiness_preflight(tmp_path):
    dev_rep = verify_official_dev_readiness(output_dir=tmp_path)
    assert dev_rep['schema_version'] == 'm18a-dev-readiness-v1'
    assert dev_rep['official_dev_claims'] == 500
    assert dev_rep['status'] == 'READY_FOR_SEALED_EVALUATION'
    assert dev_rep['train_dev_separation']['exact_text_overlap_count'] == 4
    assert dev_rep['train_dev_separation']['dev_claims_in_memory_count'] == 0
    assert dev_rep['arms_configured'] == ['D0', 'D1', 'N0', 'A', 'B', 'C', 'D']
