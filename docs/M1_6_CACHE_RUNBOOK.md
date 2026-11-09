# M1.6 cache-first runbook

Run these commands only after the feature branch is checked out. They are CPU/read-only with respect to the canonical pilot and do not start retrieval or inference.

```bash
cd /home/stackops/whale/training-free-fact-verification
git fetch origin
git switch codex/m16-heal-design-integrity
git pull --ff-only origin codex/m16-heal-design-integrity
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m tffv.experiments.m16_cache index \
  --canonical-run outputs/m162a-dev50-v1 \
  --output outputs/m16-cache-audit-v1/canonical-index.json

python -m tffv.experiments.m16_cache prepare-evidence \
  --index outputs/m16-cache-audit-v1/canonical-index.json

python -m tffv.experiments.m16_cache inference-only \
  --index outputs/m16-cache-audit-v1/canonical-index.json \
  --condition P3

python -m tffv.experiments.m16_cache resume \
  --index outputs/m16-cache-audit-v1/canonical-index.json \
  --condition P3

python -m tffv.experiments.m16_cache evaluate-only \
  --index outputs/m16-cache-audit-v1/canonical-index.json \
  --gold-source /home/stackops/whale/ERRORFlow/data/raw/averitec/train.json \
  --output outputs/m16-cache-audit-v1/evaluation-only.json
```

`index` verifies every receipt-covered file and emits `INDEX n/50` progress to the terminal. The canonical run occupies about 24 GiB, so this CPU/I/O hash audit can take minutes. It writes `canonical-index.json` only after all 50 claims pass; `Ctrl+C` leaves no partial index. If it is interrupted, rerun the same `index` command and wait for `INDEX complete: 50/50` before running any later command. A missing index makes `prepare-evidence`, `inference-only`, `resume`, and `evaluate-only` fail by design.

`evaluate-only` is the only command above that reads gold, and it does so after every record/receipt/prediction has been admitted. Compare its metrics with the canonical `evaluation-v1/results.json`; it must not be used to change a prompt or select a follow-up condition.

For a new condition on the same 50 claims, first construct and seal its exact prompts against `canonical-index.json`. Its evidence key must be a hit for all 50 claims. Run only the 50 missing prediction keys with the frozen model; do not invoke retrieval. This repository version provides the admission/planning commands but intentionally does not infer a new prompt or launch GPU work from a cache hit.

For the next 50 claims, `prepare-evidence` must report them missing. A separately authorized, sealed DEVELOPMENT protocol is required before the existing retrieval worker is used. Once all Top-8 receipts are sealed, index them as a new cache generation and run shared evidence across all approved conditions. Never point it at `TEST_QUARANTINE_EXPOSED` or ERRORFlow prediction files.
