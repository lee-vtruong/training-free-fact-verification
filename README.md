# Training-Free Error-Informed Fact Verification

Improve frozen fact-verification Accuracy and Macro-F1 using historical source
errors and successes, with **no parameter updates**. This is an independent
project; ERRORFlow is read-only. No routing architecture is selected.

Milestone one provides source-cache validation, error profiles, fixed-ontology
metrics, grouped partitions, provenance, immutable prediction caching, and
B0–B8 specifications. It does not run a model or implement advanced methods.
Real Base results are pending access to historical server caches.

M1.1 adds R1–R4 retrieval code and offline synthetic tests. Actual AVeriTeC
benchmark results await user-run server commands and verified inputs. Start with
[server commands](docs/M11_SERVER_COMMANDS.md),
[retrieval protocol](docs/M11_RETRIEVAL_PROTOCOL.md), and
[current report](reports/M11_RETRIEVAL_REPORT.md).

## Run lightweight checks

Python 3.10+; the library uses only the standard library. Tests require pytest.
From this repository in PowerShell:

```powershell
python scripts/run_tests.py
$env:PYTHONPATH = "$PWD/src"
python -m tffv.experiments.audit --help
python scripts/inventory.py --root $env:ERRORFLOW_ROOT
```

The test runner uses a fresh temporary directory and disables pytest's disk
cache, avoiding cross-account permissions on Windows. It forwards pytest
arguments, for example `python scripts/run_tests.py -q tests/test_retrieval.py`.

On a prepared server, from the NEW repository:

```bash
PYTHONPATH=src python -m tffv.experiments.audit \
  --data "$TFFV_SOURCE_DATA" --predictions "$TFFV_BASE_PREDICTIONS" \
  --manifest "$TFFV_BASE_MANIFEST" --output outputs/base-audit-001 \
  --folds 5 --seed 2026
```

Use a fresh output directory on every run. Inputs are never written. Do not
point outputs or writable cache links at the old project. Environment variables
are read directly; `.env` is a local reference, not automatically executed.
Optional installation: `python -m pip install -e '.[dev]'` in a dedicated environment.

See [configuration and contracts](docs/CONFIGURATION.md),
[resource inventory](docs/RESOURCE_INVENTORY.md),
[leakage audit](docs/DATA_LEAKAGE_AUDIT.md),
[Base profile](reports/BASE_ERROR_PROFILE.md),
[scope](docs/RESEARCH_SCOPE.md), and [roadmap](docs/EXPERIMENT_ROADMAP.md).
