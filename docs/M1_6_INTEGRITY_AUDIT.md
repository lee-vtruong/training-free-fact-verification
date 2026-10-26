# M1.6.0 — Cross-Milestone Integrity Audit

Date: 2026-10-26 (Asia/Saigon). Audited revision: `08c6ba77f8d148ca77311ae465d0c349a402d0df`.
Status: requested numerical audit COMPLETE; historical execution provenance has the limitations below. M1.6 GPU experiments NOT RUN.

## Evidence and precedence

CPU recomputation ran directly on `hvtham-server`, project `/home/stackops/whale/training-free-fact-verification`. Gold was read only for offline scoring from the read-only ERRORFlow `data/raw/averitec/train.json`; its SHA-256 is recorded, and no gold was sent to retrieval/inference. No ERRORFlow files were written.

Evidence: [machine-readable scores and 196-file frozen inventory](audits/M1_6_INTEGRITY_EVIDENCE.json), [failed/recovered run provenance](audits/M1_6_SERVER_PROVENANCE.json), and [auditor](../scripts/audit_m16_integrity.py). All ten prediction files match their existing manifest SHA-256. All scored populations have unique IDs. M1.3 canonical C0 CSV agrees with raw C0 predictions. This audit takes precedence over conflicting narrative summaries for M1.6; original reports/artifacts remain unchanged.

## A1. D0 Fold D: 49/91 is canonical

Run: `outputs/m142-confirmatory-20261025-233407/D0/predictions.jsonl`.
SHA-256: `27836c3ff4fad006f9fd9c707171e831e403f9ae6089f765511dd79aca8f87e1`.

| Fold D condition | Correct / N | Accuracy | Macro-F1 | Generation cap |
|---|---:|---:|---:|---:|
| D0 | 49/91 | 0.538462 | 0.416119 | 512 |
| D1_matched | 58/91 | 0.637363 | 0.391883 | 512 |
| D2 | 54/91 | 0.593407 | 0.371839 | 512 |
| D3 | 52/91 | 0.571429 | 0.360260 | 512 |
| D4 | 52/91 | 0.571429 | 0.383559 | 512 |

The M1.5 synthesis's D0 57.1% is unsupported by this canonical run. That percentage corresponds to D3/D4 here; copying their value is a plausible transcription explanation, not an established cause. No alternative D0 run supporting 52/91 was identified in the inspected project artifacts. D1_matched gains **9**, not 6, correct claims over D0, but loses **0.024236 Macro-F1**. Paired accounting: CC=44, CW=5, WC=14, WW=28. Therefore higher Accuracy does not establish improvement on both primary metrics.

The synthesis also calls D1_matched a 64-token condition on Fold D. Its manifest actually specifies **512 max_new_tokens**. Concise instructions shortened realized outputs; the hard 64-token cap belongs to Fold E. Do not attribute the Fold D effect to an experimentally isolated cap reduction.

## A2. D0 Fold B: 0.4883 is supported in both runs

| Identity | Raw prediction SHA-256 | Correct / N | Macro-F1 |
|---|---|---:|---:|
| M1.3 C0, `m13-error-informed-20261025-000523` | `88593066618526bec92aff3f4b17e4ab8e184c8daaa9f007c1457bebea2b018c` | 28/46 | 0.488290 |
| M1.4 D0, `m14-pilot-20261025-014111` | `f85331f2db9062be7eb650cb68cac29f4ce9b84c81583620cc334b6e9963a36c` | 28/46 | 0.488290 |

Their admitted ID sets and predicted verdict mappings are identical (CC=28, WW=18, no changed labels). They remain separate executions: raw hashes and timing differ. The M1.3 repaired canonical CSV independently gives 0.488290. **0.4079 is a narrative inconsistency, not a verified alternative condition/version.** No inspected canonical run supports it. Do not average the two runs or pool them as independent samples.

The M1.4 pilot protocol's prose quarantine IDs also disagree with its actual manifest. Use the manifest/actual admitted IDs, not the prose list, for any reproduction. The audit joins on IDs and asserts population equality rather than trusting index ranges.

## A3. D2: two deployed directives versus four historical checklist rules

- M1.3 **C6** is the four-point checklist condition: media debate, evidence sufficiency, cherrypicking precision, polarity. Source: M1.3 protocol and historical source-memory checklist. This is not D2.
- M1.4 **D2** uses `CONCISE_SOURCE_ERROR_RULES`, introduced in `d3996ee`: two numbered directives and a separate one-sentence response instruction. M1.4.1 retains that family.
- M1.4.2 D2 saved `prompts.jsonl` explicitly contains two directives. `D2_minus_r1` keeps only Rule 2; `D2_minus_r2` keeps only Rule 1; `D2_para1` changes the wording of both rules. The dispatcher also supports `D2_para2`, which is not evidence of an executed condition.
- `65f3e3f` adds matched controls/ablations. Four historical rule ideas were distilled into two; the numbering is not a one-to-one mapping. The M1.5 synthesis's statement that D2 *injected four rules* is incorrect for the saved Fold D prompts.

Exact M1.6 Rule 1: “Media debate is not factual contradiction: An article describing a political disagreement does not constitute Refutation of a verifiable fact.”

Exact M1.6 Rule 2: “Missing context is not Cherrypicking: Absence of secondary details does not warrant Cherrypicking if the primary assertion is substantiated.”

Both are hypotheses to ablate. D2 vs D1_matched on Fold D has CC=54, CW=4, WC=0, WW=33. This is evidence about this condition/population; it does not prove that all error rules fail universally. Historical D2 replaces generic directives; proposed B4 adds rules to D1_matched and must receive a new identity.

## A4. Fold E validity and recovery

Two directories remain separate and are hashed in the inventory:

| Run | M0 | M1 | M4 | Interpretation |
|---|---:|---:|---:|---|
| `m15-evaluation-20261026-142010` | 0/99 valid | 0/99 | 0/99 | Failed capture/execution condition |
| `m15-evaluation-20261026-151554` | 99/99 | 99/99 | 99/99 | Recovered non-thinking condition |

The failed M0/M1 rows store empty `raw_text` and zero token accounting; all 297 failed M4 permutation texts are empty. An adapter returned `generated_text`, `prompt_tokens`, `generated_tokens` while the runner requested different keys. Commit `fc5ab9b` adds aliases and fallbacks. Commit `5f5e7a2` explicitly passes `enable_thinking=False` for single-pass and permutation inference; previously the backbone default `None` did not override Qwen's native chat-template behavior.

The WORK_LOG diagnoses native thinking exhausting 64 tokens as a second failure. Code supports that risk, but the failed serialized generations were lost through the capture bug. **The exact token-level thinking behavior cannot be independently verified from those empty records.** A separately preserved nonempty intermediate failing trace was not found in the inspected inventory. Do not invent it or describe recovery as reparsing the original generations: it was a **new inference run** after code fixes.

Partition manifests are byte-identical between failed/recovered runs; M0/M1 per-ID prompt hashes match. The decoding/chat-template invocation changed. Identical prompt hashes therefore do not prove identical effective model inputs. Failed outputs remain an engineering failure record, not evidence that the verifier's true quality was 0%.

Recovered M0/M1 raw texts reparse to 99/99 valid and agree with stored labels. M4 aggregate records have no raw generation by design; all **297/297 constituent raw generations** parse valid, their stored aggregates match canonical predictions, and majority/tie-to-canonical aggregation is checked. Aggregate validity is 99/99. A missing aggregate raw field is not 0% validity.

| Recovered Fold E | Correct/N | Accuracy | Macro-F1 | CC/CW/WC/WW vs M0 |
|---|---:|---:|---:|---|
| M0 | 67/99 | 0.676768 | 0.428964 | baseline |
| M1 | 69/99 | 0.696970 | 0.440743 | 67/0/2/30 |
| M4 | 67/99 | 0.676768 | 0.423312 | 65/2/2/30 |

M1 repairs `train:306,345`; M4 repairs `train:343,345` and regresses `train:262,301`. Additional wrong-to-wrong label changes are retained in the JSON. Zero observed M1 regressions on 99 claims does not establish universal safety. Neither comparison met the reported adjusted significance criteria. M4's result does not prove inference aggregation can never help.

Measured token totals M0: 260,971 input / 5,792 output; M1: 260,971 / 5,830; M4: 782,913 / 17,400 (sum across three calls). Manifest mean latency is approximately 1.5063 / 1.4974 / 4.4744 seconds respectively. Raw single-pass latency recomputation differs slightly due to rounding. Older missing exact token fields remain null in this audit; no estimates are presented as measurements.

## A5. Condition identity and provenance limits

An experiment identity must include dataset release/hash, exact admitted IDs/groups, model/tokenizer/chat-template revision, code revision and dirty diff, full messages/effective chat input hash, retrieved evidence IDs/order/text hashes, exemplar IDs/order/fields, decoding, hard output cap, parser, retries, aggregation, device/dtype and timing method. A label such as D0 or M0 alone is insufficient.

Fold E M0 shares generic directive text with D1_matched, but has a different population, hard cap and runner/template. No Fold E D0 was run in that suite. B/C/D/E have all informed research and are exploratory for M1.6. No pooled cross-fold improvement or untouched-confirmation claim is made here.

Old manifests incompletely capture effective inputs, runtime versions and run-time commit provenance. Current HEAD plus code history is not proof of the exact process revision used in every old run. Do not retrofit metadata as if recorded at execution. File timestamps/run-name times are identifiers, not certified execution times.

Server's existing `scripts/run_m15_eval.sh` dirty diff is solely executable mode `100644 -> 100755`; preserved. Final read-only inspection verified **all 196 inventoried files unchanged**, including failed and recovered Fold E raw artifacts. Historical M1.5 artifacts and reports are unchanged. No GPU jobs, memory construction, or paper/novelty claims are part of this audit.

## Reproduce

```bash
cd /home/stackops/whale/training-free-fact-verification
PYTHONPATH=src python3 /tmp/audit_m16_integrity.py \
  --root "$PWD" \
  --gold /home/stackops/whale/ERRORFlow/data/raw/averitec/train.json \
  --output outputs/m16-integrity-audit/audit.json
python3 /tmp/inspect_m16_server.py
```

The `/tmp` scripts are copies of the committed scripts; copy them there before reproducing from an older server checkout. The auditor writes only its new output. The source raw artifacts stay in their original directories; provenance JSON supplies hashes and paths, not fabricated replacement raw data.
