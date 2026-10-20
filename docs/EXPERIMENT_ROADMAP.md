# Experiment roadmap

## Gate 0: finish empirical Base audit

Restore server access. Inventory datasets, raw cache rows, sidecars, checkpoint
snapshots, score profiles, evidence builders and group manifests read-only.
Admit only protocol-compatible caches. Implement a specific canonical adapter
after inspecting actual fields; preserve raw input and conversion hashes.
Run `tffv-audit` once per homogeneous source protocol, review error examples
and group overlaps, and publish replayable Base metrics. Resolve conventional
versus audit-style Base first. No advanced method is selected yet.

## Pilot 1: demonstration composition

Compare B0/B1 with B3 random, B4 correct-only, B5 error-only and B6 mixed source
examples on group-safe source folds. Match demonstration count, template and
context budget. Display explicit gold corrections for historical wrong outputs.
Use deterministic seeds; report when a partition lacks enough eligible examples
rather than silently changing the count. Add a correction ablation: corrected
gold versus historical-prediction context with a neutral instruction, keeping
the wrong answer clearly marked as historical and never as the correct target.

## Pilot 2: fold-local source rules

Generate a small checklist from only allowed source errors with a frozen LLM;
compare B7 to generic B1/B2 instructions, length-matched generic checklists and
shuffled/irrelevant source rules. Bind generator prompt/checkpoint, allowed IDs,
artifact hashes and fold assignment. Generate no global summary for OOF use.
Any source prompt selection happens in inner folds.

## Pilot 3: relevant contrastive examples

Compare B8 source correct/wrong contrasts with random contrasts and relevant
correct-only examples. Retrieve using frozen representations or deterministic
lexical similarity; no fitted embeddings or reranker. Match counts and token
budgets; distinguish retrieval relevance from the value of error correction.
Do not start this pilot until a leakage-safe group-aware retriever exists.

## Controls and reporting for every pilot

Keep checkpoint, label space/order, evidence text, scoring protocol and dataset
IDs identical across compared methods. Track prompt hashes, seeds, exemplar IDs,
generated knowledge and exact token lengths/truncation. If a method adds multiple
inference steps, include an ordinary structured-reasoning control with comparable
forward passes/tokens. Missing cost measurements remain null. Report Accuracy
and Macro-F1 first, per-class/confusion effects and harmful flips second; compute
paired group-aware confidence intervals and source fold/seed stability before
making improvement claims. Cost is descriptive, not the optimization objective.

Freeze the selected source configuration before any new external evaluation.
Previously inspected AVeriTeC dev, CLIMATE-FEVER and SciFact results are
exploratory; seek an untouched population for confirmatory claims. No target
labels enter adaptation. Stop after source pilot review before choosing a final
architecture or drafting a paper.
