# AI Usage Declaration

AI coding assistants were used throughout this project, as permitted by the assignment. This document declares which tools were used, for what, and — per the assignment's request — specifically where they changed a decision.

## Tools used

### Claude (Anthropic) — review and verification support
Used as a second pair of eyes on the assignment requirements and on code/documentation produced during the build — reviewing the assignment PDF against the milestone plan, and spot-checking specific claims in generated documentation against the real CSV files and the trained model rather than accepting written summaries at face value.

This review surfaced a few issues that were then fixed directly:
- Two lines in an early decision log described the duplicate-`employee_id` handling and a "missing salary" cleaning step inaccurately relative to the raw data (salary is actually identical across duplicate pairs, with `enrolled` conflicting; and `salary` has zero missing values) — corrected in the documentation.
- The model's initial 1.0000 ROC-AUC was checked via cross-validation and a simplified decision tree to confirm it reflected a genuine property of the dataset rather than a pipeline bug.
- The saved model was found to include `age`, `gender`, and `marital_status` as input features, which didn't match the project's intended fairness policy of excluding them. `train_test_split.py` was updated to drop these features and the model was retrained, bringing ROC-AUC to a more representative 0.9682; the fairness audit was also extended to cover `marital_status` alongside gender and age group.
- `predict_tool.py` was missing the leakage-refusal check that `explain_tool.py` already had; a matching check was added.
- A printed summary line in `model_training_and_evaluation.py` stated an outdated AUC value inconsistent with the number printed just above it; corrected to compute the value dynamically.
- The Streamlit dashboard's ranking table was displaying the raw, uncleaned `plan_tier_requested` column instead of the cleaned version; corrected to reference `plan_tier_requested_clean`.

### Gemini (Google)
Used for early-stage brainstorming of the system architecture diagram and layout (regions/layers of the data → model → agent flow) before this was finalized and rendered separately, and for occasional secondary opinions on the fairness-checkpoint reasoning while deciding on the demographic-exclusion policy.

### ChatGPT (OpenAI)
Used for auxiliary drafting and quick sanity-checks during early exploration of the dataset schema, and for phrasing/wording help on parts of the categorical-cleaning mapping dictionaries (e.g. brainstorming the full set of canonical plan-tier labels to map free-text variants onto).

### Antigravity (Google, VS Code extension)
Used occasionally as coding assistance within the IDE while implementing parts of the pipeline (`data_processing.py` / `feature_engineering.py`, `train_test_split.py`, `model_training_and_evaluation.py`, the agent tools, `cli.py`, `app.py`). All code it produced was subject to the same independent verification described above before being accepted — several of the fixes listed under Claude were corrections to logic it had generated.

## Where AI assistance changed a decision vs. where it did not

- **Changed a decision:** the demographic feature exclusion was intended from the start (per the initial project plan), but the generated code did not actually implement it — verification against the real model artifact caught this gap, and the fix (retraining without demographics) is what's reflected in the current `models/enrollment_model.joblib` and the metrics in `report.md`.
- **Changed a decision:** the two documentation inaccuracies in the original decision log (duplicate-row cause, missing-salary cause) were corrected to match verified reality rather than left as originally written.
- **Did not change:** the core leakage call on `legacy_propensity_score` — this was correctly identified as leakage from the start (0.976 correlation, 1.0000 standalone AUC) and confirmed independently; no correction was needed here.
- **Did not change:** the sentinel-handling logic for `prior_year_enrolled` (`-1` = no prior record) was correctly implemented from the start and confirmed against the raw data as-is.

## What was NOT AI-generated without review

Every cleaning decision, leakage classification, fairness policy, and refusal rule documented in `report.md` was independently re-derived against the actual raw and processed CSV files and the actual saved model artifact before being accepted as final — not taken on the word of any single AI tool's generated summary.
