# Milestone validation

Date: 2026-10-20, Asia/Saigon. Local Python 3.13.11, pytest 9.1.1.

Commands executed from the new project:

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
$env:PYTHONPATH = "$PWD/src"
python -m tffv.experiments.audit --help
python scripts/inventory.py --root C:/Users/ASUS/Downloads/ERRORFlow
```

Test result: **22 passed**. Compilation and CLI help succeeded.
Tests cover hand-computed Accuracy/Macro-F1 including absent classes,
exact joins, input hashes, model/prompt/evidence protocol rejection,
score/confidence validation, gold-access declarations, group exclusion,
normalized duplicate exclusion, deterministic folds, artifact envelopes,
B0–B8 comparability, correction targets, immutable caching, output containment,
and complete audit-record metric replay.

All validation examples are synthetic. Tests create temporary inputs and fresh
ignored outputs, then remove their own output directories. They do not import
or execute ERRORFlow code. No scientific dataset, checkpoint, GPU job, fitted
component, package installation or external-label evaluation was used.

The new Git repository was initialized without staging or committing files.
The old checkout retains its pre-existing dirty state.

## Pending empirical work

No research Base metrics, real duplicate counts, checkpoint compatibility,
server GPU/storage or cache completeness have been validated. SSH to
`hvtham@10.44.101.230` timed out; the documented hostname did not resolve.
Restore reachability, then run the resource inventory from the NEW server
checkout with `--root ~/whale/ERRORFlow`. Save output in the new project's
`outputs/`, inspect actual cache schemas and original manifests, and implement
a reviewed canonical conversion before running the audit.
