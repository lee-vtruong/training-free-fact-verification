import json
import math
from pathlib import Path
from tffv.provenance import digest, file_hash


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} must be an object")
            rows.append(row)
    return rows


def require(row, fields):
    for key in fields:
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Missing/non-string field: {key}")


def index(rows):
    result = {}
    for row in rows:
        require(row, ["sample_id"])
        if row["sample_id"] in result:
            raise ValueError("Duplicate sample_id")
        result[row["sample_id"]] = row
    return result


def load_audit_inputs(data_path, predictions_path, manifest_path):
    """Exact alignment only. No label projection, missing-row filling, or guessing."""
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    require(m, ["dataset", "split", "population_status", "evidence_protocol", "checkpoint_id",
                "checkpoint_sha256", "prompt_id", "prompt_sha256", "scoring_protocol",
                "grouping_provenance", "source_sha256", "predictions_sha256"])
    if m.get("role") != "source" or m.get("frozen") is not True:
        raise ValueError("Audit requires a frozen source manifest")
    if m["population_status"] not in {"exploratory", "previously_inspected", "source_development"}:
        raise ValueError("Declare the source population status; holdouts are not source audit inputs")
    if m.get("grouping_verified") is not True:
        raise ValueError("Verified near-duplicate grouping is required")
    if m.get("gold_accessed_during_inference") is not False:
        raise ValueError("Inference must declare no gold access")
    if m.get("checkpoint_hash_type") != "snapshot_content_sha256":
        raise ValueError("A model path hash is not a checkpoint content fingerprint")
    for key in ("checkpoint_sha256", "prompt_sha256", "source_sha256", "predictions_sha256"):
        if len(m[key]) != 64 or any(c not in "0123456789abcdef" for c in m[key]):
            raise ValueError(f"Invalid SHA256: {key}")
    labels = m.get("labels")
    if not isinstance(labels, list) or len(labels) < 2 or any(not isinstance(x, str) or not x for x in labels) or len(set(labels)) != len(labels):
        raise ValueError("Explicit unique label ontology required")
    for path, key in ((data_path, "source_sha256"), (predictions_path, "predictions_sha256")):
        if file_hash(path) != m[key]:
            raise ValueError(f"Input hash mismatch: {key}")
    source, pred = index(read_jsonl(data_path)), index(read_jsonl(predictions_path))
    if not source or set(source) != set(pred) or m.get("expected_records") != len(source):
        raise ValueError("Empty/incomplete cache or ID alignment failure")
    records = []
    for sid, s in source.items():
        p = pred[sid]
        require(s, ["dataset", "split", "claim", "evidence_protocol", "evidence_reference", "gold_label", "group_id"])
        if not isinstance(s.get("evidence"), str):
            raise ValueError("Evidence must be explicit text (empty string allowed)")
        if s["gold_label"] not in labels or p.get("prediction") not in labels:
            raise ValueError("Unknown label; explicit reviewed mapping is required")
        for key in ("dataset", "split", "evidence_protocol"):
            if s[key] != m[key]:
                raise ValueError(f"Source protocol mismatch: {key}")
        identity = {"claim": s["claim"], "evidence": s["evidence"], "evidence_protocol": s["evidence_protocol"]}
        if p.get("input_sha256") != digest(identity):
            raise ValueError("Prediction evidence/claim hash mismatch")
        for key in ("checkpoint_sha256", "prompt_sha256", "scoring_protocol"):
            if p.get(key) != m[key]:
                raise ValueError(f"Prediction protocol mismatch: {key}")
        if p.get("gold_accessed") is not False:
            raise ValueError("Prediction gold access unknown or true")
        scores = p.get("label_scores")
        if scores is not None and (not isinstance(scores, dict) or set(scores) != set(labels) or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in scores.values())):
            raise ValueError("Scores must contain one finite number per declared label")
        confidence = p.get("confidence")
        if confidence is not None:
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Confidence must be a finite [0,1] proxy")
            require(m, ["confidence_definition"])
        record = dict(s, prediction=p["prediction"], correct=s["gold_label"] == p["prediction"],
                      confusion_pair={"gold": s["gold_label"], "predicted": p["prediction"]},
                      label_scores=scores, confidence=confidence,
                      checkpoint_id=m["checkpoint_id"], checkpoint_sha256=m["checkpoint_sha256"],
                      prompt_id=m["prompt_id"], prompt_sha256=m["prompt_sha256"],
                      scoring_protocol=m["scoring_protocol"], evidence_words=len(s["evidence"].split()))
        records.append(record)
    return records, m
