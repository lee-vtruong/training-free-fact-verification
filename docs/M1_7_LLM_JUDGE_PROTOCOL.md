# M1.7 — LLM-Judged Corrective Memory Feasibility

Status: implementation prepared; **no judge call, retrieval, GPU inference, SSH, or development evaluation has run**.

M1.7 is an exploratory branch beside M1.6 human review. `LLM_VALIDATED` means a frozen judge accepted an evidence-grounded correction; it never means `HUMAN_VERIFIED`, and it does not alter M1.6 review sheets, gates, or records.

## Frozen scope and two stages

The input is exactly 21 SOURCE candidates: five original and 16 E3 drafts from `M1_6_2E_CANDIDATE_DRAFTS.json`. E1 is unsupported and E2 is quarantined because its evidence availability changed. No source expansion is authorized.

1. **Gold-blind evidence assessment** receives claim, stored Base verdict, and asserted rendered Top-8 excerpts. It receives neither source gold nor candidate correction.
2. **Correction validation** receives the same evidence, source gold verdict, candidate correction and stage-one output. It judges grounding and gold relation.

Both require structured JSON. Acceptance requires exact quote matches, source-gold verdict match, `evidence_supported=true`, at least one grounding quote, and no declared unsupported facts. Every raw output stores model ID/revision, temperature, generation settings, and prompt hash. Any mismatch fails closed.

## Four-arm feasibility protocol

Once a sealed LLM-validated bank exists, exemplar selection is frozen before target inference and source groups remain disjoint from target groups. Arms use identical exemplar IDs, source evidence, target R4 Top-8, verifier model, decoding, schema, and ordering:

| Arm | Historical content |
|---|---|
| A | Gold only |
| B | Gold + LLM-validated correction |
| C | Base wrong verdict + Gold + same LLM-validated correction |
| D | Gold + generic rationale matched to B's correction length |

The next development population must use sealed unused DEVELOPMENT groups only, never DEVELOPMENT50, and must not read target gold when selecting targets/examples. One R4 retrieval pass is then reused by all four arms; later conditions are inference-only and fail closed on evidence-cache mismatch. TEST_QUARANTINE remains inaccessible.

## User-run commands

The judge adapter is implemented but needs a user-supplied executable that accepts one request JSON on stdin and emits one judge JSON object on stdout. It must not access development/test data. Substitute your actual frozen stronger-model adapter path and its immutable model revision.

```bash
cd /home/stackops/whale/training-free-fact-verification
git switch codex/m16-heal-design-integrity && git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m tffv.experiments.m17 make-requests \
  --evidence-bank "$PWD/outputs/m162a-inputs-v1/memory75.json" \
  --output outputs/m17-stage1-requests

python scripts/m17_judge_adapter.py \
  --requests outputs/m17-stage1-requests/evidence_assessment_requests.jsonl \
  --output outputs/m17-stage1-raw.jsonl \
  --command 'python /absolute/path/to/your_frozen_judge_adapter.py' \
  --model-id 'YOUR_FROZEN_STRONGER_JUDGE' --model-revision 'IMMUTABLE_REVISION' \
  --temperature 0.0 --max-new-tokens 768

python -m tffv.experiments.m17 validate \
  --requests outputs/m17-stage1-requests/evidence_assessment_requests.jsonl \
  --outputs outputs/m17-stage1-raw.jsonl --output outputs/m17-stage1-validated

python -m tffv.experiments.m17 make-requests \
  --evidence-bank "$PWD/outputs/m162a-inputs-v1/memory75.json" \
  --stage-one-outputs outputs/m17-stage1-validated/validated_outputs.jsonl \
  --output outputs/m17-stage2-requests

python scripts/m17_judge_adapter.py \
  --requests outputs/m17-stage2-requests/correction_validation_requests.jsonl \
  --output outputs/m17-stage2-raw.jsonl \
  --command 'python /absolute/path/to/your_frozen_judge_adapter.py' \
  --model-id 'YOUR_FROZEN_STRONGER_JUDGE' --model-revision 'IMMUTABLE_REVISION' \
  --temperature 0.0 --max-new-tokens 768

python -m tffv.experiments.m17 validate \
  --requests outputs/m17-stage2-requests/correction_validation_requests.jsonl \
  --outputs outputs/m17-stage2-raw.jsonl --output outputs/m17-stage2-validated

python -m tffv.experiments.m17 build-memory \
  --evidence-bank "$PWD/outputs/m162a-inputs-v1/memory75.json" \
  --stage-one outputs/m17-stage1-validated/validated_outputs.jsonl \
  --stage-two outputs/m17-stage2-validated/validated_outputs.jsonl \
  --output outputs/m17-llm-validated-memory-v1
```

The final development-batch retrieval/inference runner is deliberately not enabled by this protocol-only milestone: it requires a separately locked new target manifest after the observed LLM-valid bank is sealed. That prevents post-hoc target/exemplar selection. The exact four-arm runner will be implemented only after that lock, not improvised at server execution time.

## Resource envelope

Known workload: 21 stage-one + 21 stage-two judge calls (42 maximum), 768 output tokens/call maximum. Prompt tokens, latency, GPU-hours, and monetary cost are **not measured** and depend on the user-chosen frozen judge adapter. New development retrieval/inference cost is not estimated because target count is intentionally not selected before memory sealing.
