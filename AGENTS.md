# Project working rules

- Codex works locally: implement code, run lightweight local tests, document
  actual results and provide commands for the user to execute on their server.
  Do not initiate SSH, server jobs, remote push/pull or checkpoint downloads.
  The user executes server commands and sends logs/artifacts for analysis.
- Every code handoff must include the Windows commands for the user to
  review/commit/push to GitHub and the Linux commands to pull that code onto
  the server before execution. Provide first-clone instructions if needed.
  Check local Git state before drafting commands; never force-push or overwrite
  remote history. Codex does not execute those synchronization commands.
- Accuracy and Macro-F1 are primary. All models/features remain frozen; no
  parameter fitting, optimizers, neural routers or learned rerankers.
- ERRORFlow and its environments/data/caches are read-only references. New
  artifacts belong to this project. Verify provenance before reusing caches.
- Never use evaluation gold, gold QA or justifications during retrieval or
  inference. Preserve official splits and the AVeriTeC four-class ontology.
  Historical source knowledge must exclude evaluation groups/folds.
- Log every implementation/research step in `docs/WORK_LOG.md` with commands,
  tests, failures, Git state, decisions, limitations and next steps.
- Unknown metrics are null/not measured. Synthetic validation is never a
  substitute for real experimental evidence. Report unfinished milestones as
  INCOMPLETE, and do not silently change a protocol to improve outcomes.
- Current scope is M1.1 retrieval only: R1 BM25, R2 frozen dense, R3 RRF hybrid,
  R4 frozen reranking. Do not run Qwen3-8B verification, build Historical Error
  Memory, or automatically select R4 as the M1.2 baseline.
