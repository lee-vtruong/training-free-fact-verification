# M1.6.1 — User-run fixed batch 25

Date: 2026-10-27. **Prepared, not executed by the agent.** Exploratory source expansion is authorized as a research direction independently of clean-holdout resolution; the user chooses when to run this Bash block. It starts no HEAL ablation or confirmation and never expands beyond the first fixed 25.

## Copy-paste on stackops@hvtham-server

```bash
set -Eeuo pipefail
cd /home/stackops/whale/training-free-fact-verification
git fetch origin
git switch codex/m16-heal-design-integrity
git pull --ff-only origin codex/m16-heal-design-integrity
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python --version
python scripts/m16_validate_seal.py docs/partitions/m16-v2
bash scripts/run_m161_batch25.sh
```

An existing mode-only change to `scripts/run_m15_eval.sh` is preserved; no stash/reset/clean/force checkout is used. If Git reports an actual conflict, stop instead of discarding local work. Use the printed revision for the run; if a later checkout changes source code, existing run resume will refuse it.

The [wrapper](../scripts/run_m161_batch25.sh) prints HEAD/branch, Python/GPU metadata, fixed source count, paths and a UTC-stamped log. It audits own-project legacy A on CPU, exports blank reviews, prepares batch inputs, runs full environment/model-file preflight, and invokes GPU/network execution **only after preflight succeeds**. A separate process joins source gold after all predictions are sealed. It does not load ERRORFlow predictions or read development/test gold for selection/inference.

Paths:

- Logs: `outputs/m161-logs/batch25-<UTC>.log` (tee to console and file).
- Legacy own-source bank/review sheets: `outputs/m161-legacy-source-v1/`.
- Fixed new-source batch: `outputs/m161-source-batch25-v1/`; `batch.json`, `environment.json`, `state.json`, per-claim receipts, raw outputs, `completed.json`.
- New-source bank/reviews after completion: `outputs/m161-source-batch25-v1/memory/`.

The wrapper creates no background job or scheduler. Use an existing interactive terminal/session the user controls. No such job was started by the agent.

## Preflight and data contract

The fixed IDs are taken byte-for-byte from the sealed schedule, not from a command-line range or a new random sample:

```text
1873 1644 1863 426 1288 467 1429 694 2224 1236 1231 3012 2055
1402 978 2811 828 979 1817 2449 903 1881 1824 718 1847
```

Shards: 8 stores from `train_0_999.zip`, 12 from `train_1000_1999.zip`, 5 from `train_2000_3067.zip`. Acquisition uses existing `archive_for`, not the first-shard-only fold wrapper. The unchanged historical plan in [config](../configs/m161_acquisition_plan.json) has SHA-256 `833f46c5f8ebefa28d9a171e909a8f9500319eaf539ce2bf290e262245604f23`; its official URLs/revision/member ranges are validated. Whole-ZIP hashes are expected metadata; consumed members are CRC/SHA checked, not falsely described as a complete ZIP download/hash verification.

CPU prepare checks v2/schedule hashes, exactly 25 eligible SOURCE IDs, claim-text hashes, scrubbed fields, frozen retrieval config and generation caps. It materializes one claim-only manifest per ID. No remote ZIP directory, store, model or tokenizer is loaded during preparation/structural preflight.

Full user-run preflight requires Linux, installed torch/transformers/numpy, CUDA, at least 24 GiB free VRAM, ≥100 GiB free disk, ≤30 GiB task footprint and local pinned model files. It verifies embedding/reranker content fingerprints, records the current Qwen3-8B content fingerprint and checks it on resume. This records new execution provenance; it does not invent the missing old runtime weight fingerprint. `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` prevent model downloads; explicit official knowledge-store HTTP Range acquisition still occurs when execution begins.

Protocol remains frozen R1/R2/R3/R4, Top-8, 180-word/30-overlap segmentation, depths 200, RRF 60, dedup .95, seed 2026; exact model revisions/dtypes live in [config](../configs/m161_source_batch25.json). Source D0 uses greedy non-thinking Qwen3-8B, bfloat16, 512 output tokens. Prompt/context overflow or incomplete Top-8 stops the job rather than truncating target evidence or changing the protocol.

Legacy originals, model/cache references, manifests and input exports must exist; preflight failures are blockers, not a reason to substitute other conditions. The source claim export defaults to `outputs/m11-source-claims-20261021-055413/claims_manifest.json`. Do not point the worker at raw gold or an ERRORFlow memory file.

## Execution, checkpoint and resume

The wrapper uses these implemented commands:

```bash
python -m tffv.experiments.m161 prepare-batch \
  --output outputs/m161-source-batch25-v1 \
  --plan configs/m161_acquisition_plan.json \
  --claims-manifest outputs/m11-source-claims-20261021-055413/claims_manifest.json
python -m tffv.experiments.m161 preflight --run outputs/m161-source-batch25-v1
python -m tffv.experiments.m161 run-batch \
  --run outputs/m161-source-batch25-v1 --execute --resume
```

Each ID completes acquisition → source audit → quarantine-policy candidate preparation → frozen retrieval → D0 generation. Models run sequentially. Every completed stage gets a receipt with batch identity and all output hashes, including cached vectors. Resume checks **all upstream receipts**, not just a prediction count. Tampered completed artifacts stop; they are not recomputed silently. Unsealed interrupted stages use fresh numbered attempt directories. This avoids the old acquisition function's size-only reuse branch being mistaken for a verified CRC. Incomplete attempts remain preserved and count toward footprint.

No existing cache candidate matched these 25 IDs. The old bare `.npy` loader is not pointed at an unverified shared cache. A complete sealed retrieval is reused as a whole with input/code/config/model-environment checks; an unsealed embedding from a failed attempt has insufficient provenance and is not guessed valid.

The parent has an exclusive host/PID lock and launches one child worker per claim. It monitors wall/disk limits and terminates its own worker process group on timeout/resource failure. `state.json` accumulates session wall time across normal retries; a crashed parent's still-active interval is conservatively charged through resume, including downtime. The fixed **12-hour** envelope is not reset by `--resume`. A completed invalid verdict is retained as an engineering case, not rerun because it was unfavorable.

Normally rerun the same wrapper in the same checkout/config/input paths. If a crash leaves an execution lock, first ensure the old run has stopped. The implemented recovery command checks same host and refuses live parent/worker PIDs:

```bash
python -m tffv.experiments.m161 run-batch \
  --run outputs/m161-source-batch25-v1 --execute --resume --recover-stale-lock
```

Do not delete a lock, edit elapsed time, move to a fresh output name to bypass the budget, or change frozen config to get past a blocker. Code/model/dependency/input/hash changes require an explicit new protocol/run decision; preserve the failed run. A hard-killed CPU prepare may also require a reviewed new preparation directory rather than blind overwriting. No automated cleanup touches old project/ERRORFlow artifacts.

After all 25 sealed predictions, the wrapper runs `build-batch-memory --run ... --gold-source /home/stackops/whale/ERRORFlow/data/raw/averitec/train.json --output .../memory` and exports reviews. This is a source-only postprocessing join, not an inference input. All new corrections begin null/pending. CPU `merge-memory` may later combine an explicitly chosen legacy and batch bank; duplicate IDs/review versions fail closed. No next batch or M1.6.2 pilot starts automatically.

## Budget and readiness

Estimated cold batch cost 1.88–2.66 wall hours, conservative reservation 5.57 h; actual new GPU cost remains **0 before user execution**. Network/raw/record caps remain enforced. Parent monitoring bounds runtime footprint/free space; large single stages can overshoot between monitoring polls, so limits are guarded with substantial free-space reserve. If a stage exceeds resource/time/corpus/context constraints, report INCOMPLETE rather than reduce retrieval depth or skip IDs.

Software/structural checks were tested on CPU. New server GPU preflight, model execution, real timeout behavior under CUDA and source expansion outcomes remain unmeasured. Source gold/validity/latency/token costs will be reported from retained new artifacts only after execution. Clean confirmation remains 0; this does not block authorized exploratory source construction. M1.6.2 is MODIFY pending real review/coverage and later explicit pilot instructions.
