# Research scope

Primary objective: improve Accuracy and fixed-ontology Macro-F1 of a frozen
fact verifier through reusable knowledge from historical source successes and
errors. Source gold labels are allowed. Evaluation labels cannot enter prompts,
memory, rule generation, retrieval selection or hyperparameter selection.

No model parameters, adapters, embeddings, continuous prompts, classifiers,
routers or rerankers are fitted. No optimizer steps, gradients, SFT, LoRA,
PEFT, DPO or reinforcement learning. Frozen inference, deterministic transforms,
non-parametric memory, frozen feature retrieval, source-only rule generation,
and in-context demonstrations are candidate operations.

ERRORFlow is complete and independent. Its data may be reused after audit;
its routing architecture and reported gains are not inherited. The conventional
Base and historical audit-style Base must be distinguished explicitly.

Milestone one implements contracts, strict source-cache alignment, descriptive
error metrics, group-safe partitions, provenance and baseline specifications.
No candidate method is selected, no target tuning is authorized, no expensive
inference is dispatched, and no paper is written. Token usage, forward passes,
memory and runtime are secondary metrics, never the quality objective.

Historical confusion pairs are observed outcomes. Candidate categories such as
negation, temporal mismatch, numeric scope, incomplete evidence, multi-part
claims and conflicting sources remain hypotheses until claim/evidence review.
Root-cause annotations must distinguish observed evidence from interpretation,
record annotator provenance and be constructed within permitted source folds.

Milestone status: local infrastructure and synthetic validation can be completed;
the empirical Base Error Audit requires currently inaccessible server artifacts.
No missing scientific results may be replaced with synthetic or reported values.
