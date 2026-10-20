"""All examples here are synthetic; no research results are represented."""
import json
import shutil
from dataclasses import replace
from pathlib import Path
import pytest
from tffv.backbones import VerificationInput
from tffv.data import load_audit_inputs
from tffv.evaluation import classification_metrics
from tffv.experiments.audit import run
from tffv.inference import prediction_key
from tffv.memory import assert_isolated, grouped_folds, source_partition
from tffv.prompting import BASELINES, BaselineSpec, correction_target, require_comparable
from tffv.provenance import digest, file_hash


@pytest.fixture
def inputs(tmp_path):
    rows = [dict(sample_id=str(i), dataset="SYNTHETIC", split="source", claim=f"Synthetic claim {i}",
                 evidence="Synthetic evidence.", evidence_reference=f"fixture:{i}",
                 evidence_protocol="synthetic-text-v1", gold_label=g, group_id=f"g{i}")
            for i, g in enumerate(["S", "R", "N", "S", "R", "N"])]
    m = dict(dataset="SYNTHETIC", split="source", role="source", frozen=True,
             population_status="exploratory", grouping_verified=True, grouping_provenance="fixture unique claims",
             evidence_protocol="synthetic-text-v1", checkpoint_id="fixture", checkpoint_sha256="0" * 64,
             checkpoint_hash_type="snapshot_content_sha256", prompt_id="fixture", prompt_sha256="1" * 64,
             labels=["S", "R", "N"], scoring_protocol="fixture", gold_accessed_during_inference=False,
             expected_records=6, confidence_definition="fixture uncalibrated proxy")
    pred = [dict(sample_id=r["sample_id"], prediction=p, gold_accessed=False,
                 input_sha256=digest({k: r[k] for k in ("claim", "evidence", "evidence_protocol")}),
                 **{k: m[k] for k in ("checkpoint_sha256", "prompt_sha256", "scoring_protocol")})
            for r, p in zip(rows, ["S", "S", "N", "R", "R", "N"])]
    data, predictions, manifest = (tmp_path / n for n in ("data.jsonl", "predictions.jsonl", "manifest.json"))
    def save():
        data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        predictions.write_text("\n".join(json.dumps(r) for r in pred), encoding="utf-8")
        m.update(source_sha256=file_hash(data), predictions_sha256=file_hash(predictions))
        manifest.write_text(json.dumps(m), encoding="utf-8")
    save()
    return rows, pred, m, (data, predictions, manifest), save


def test_metrics_hand_calculation():
    r = classification_metrics(["S", "R", "N", "S", "R", "N"], ["S", "S", "N", "R", "R", "N"], ["S", "R", "N"])
    assert r["accuracy"] == pytest.approx(2 / 3)
    assert r["macro_f1"] == pytest.approx(2 / 3)
    assert r["confusion_matrix"]["R"]["S"] == 1
    assert classification_metrics(["S"], ["S"], ["S", "R"])["macro_f1"] == .5


def test_load_and_optional_scores(inputs):
    _, _, _, paths, _ = inputs
    records, _ = load_audit_inputs(*paths)
    assert len(records) == 6
    assert records[1]["correct"] is False
    assert records[1]["label_scores"] is None
    assert correction_target(records[1])["correct_verdict"] == "R"


@pytest.mark.parametrize("failure", ["missing", "extra", "duplicate", "unknown_label", "evidence", "prompt", "scores", "nan", "gold_access", "checkpoint", "grouping", "target", "path_hash"])
def test_reject_incompatible_cache(inputs, failure):
    rows, pred, manifest, paths, save = inputs
    if failure == "missing": pred.pop()
    elif failure == "extra": pred.append(dict(pred[0], sample_id="extra"))
    elif failure == "duplicate": pred.append(pred[0])
    elif failure == "unknown_label": pred[0]["prediction"] = "NEW"
    elif failure == "evidence": rows[0]["evidence"] = "different"
    elif failure == "prompt": pred[0]["prompt_sha256"] = "2" * 64
    elif failure == "scores": pred[0]["label_scores"] = {"S": 1}
    elif failure == "nan": pred[0]["confidence"] = float("nan")
    elif failure == "gold_access": pred[0]["gold_accessed"] = True
    elif failure == "checkpoint": pred[0]["checkpoint_sha256"] = "2" * 64
    elif failure == "grouping": manifest["grouping_verified"] = False
    elif failure == "target": manifest["role"] = "target"
    elif failure == "path_hash": manifest["checkpoint_hash_type"] = "sha256_model_identifier"
    save()
    with pytest.raises(ValueError): load_audit_inputs(*paths)


def test_changed_file_rejected(inputs):
    _, _, _, paths, _ = inputs
    paths[0].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"): load_audit_inputs(*paths)


def test_group_safe_folds_and_exclusion(inputs):
    rows, _, _, _, _ = inputs
    rows[1]["group_id"] = rows[0]["group_id"]
    assignment = grouped_folds(rows, 3)
    assert assignment == grouped_folds(list(reversed(rows)), 3)
    assert assignment["0"] == assignment["1"]
    for fold in range(3):
        memory, evaluation = source_partition(rows, assignment, fold)
        assert len(memory) + len(evaluation) == len(rows)
    with pytest.raises(ValueError): assert_isolated(rows[:1], rows[:1])
    with pytest.raises(ValueError): assert_isolated(rows[:1], [dict(rows[0], sample_id="different", group_id="different")])
    with pytest.raises(ValueError): grouped_folds(rows[:1], 5)


def test_protocol_and_cache_interface():
    assert set(BASELINES) == {f"B{i}" for i in range(9)}
    spec = BaselineSpec("B0", "0" * 64, ("S", "R"), "score", "evidence", "dataset", "1" * 64)
    require_comparable(spec, replace(spec, baseline_id="B1"))
    with pytest.raises(ValueError): require_comparable(spec, replace(spec, evidence_protocol="oracle"))
    example = VerificationInput("1", "dataset", "claim", "evidence", "protocol")
    key = prediction_key(example, spec, "knowledge-v1")
    assert key != prediction_key(replace(example, evidence="changed"), spec, "knowledge-v1")
    assert key != prediction_key(example, spec, "knowledge-v2")
    assert "gold_label" not in vars(example)


def test_end_to_end_replay(inputs):
    _, _, _, paths, _ = inputs
    import uuid
    destination = Path(__file__).resolve().parents[1] / "outputs" / ("pytest-" + uuid.uuid4().hex)
    try:
        result = run(*paths, destination, folds=3)
        records = [json.loads(line) for line in (destination / "audit_records.jsonl").read_text(encoding="utf-8").splitlines()]
        replay = classification_metrics([r["gold_label"] for r in records], [r["prediction"] for r in records], ["S", "R", "N"])
        assert result["accuracy"] == replay["accuracy"]
        assert result["macro_f1"] == replay["macro_f1"]
        assert result["distributions"]["confidence"]["missing"]["n"] == 6
        with pytest.raises(FileExistsError): run(*paths, destination, folds=3)
    finally:
        if destination.exists(): shutil.rmtree(destination)


def test_output_escape_rejected(inputs, tmp_path):
    with pytest.raises(ValueError, match="Output must"): run(*inputs[3], tmp_path / "outside", folds=3)
