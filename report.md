# Insurance Enrollment Prediction & Outreach Assistant — Project Report

## 1. Data Issues Found

**Source files:** `employees_raw.csv` (10,008 rows × 18 columns), `region_benefit_profiles.csv` (4 regions × 9 columns).

### Duplicate employee_id records
8 `employee_id` values appear twice each (16 rows total). Investigation showed:
- `salary` and all other attributes are **identical** across each pair.
- `enrolled` **conflicts** (0 vs 1) in every one of the 8 pairs.
- `legacy_propensity_score` differs (one row populated, one `NaN`) in 7 of the 8 pairs — a secondary hint that this field tracks outcome-adjacent information (see §2).

**Policy:** dropped both rows for all 8 conflicting pairs (`keep=False`). A majority vote is impossible with exactly 2 conflicting labels, and keeping either row arbitrarily would inject label noise. Result: 10,008 → 9,992 rows.

### Missing values
| Column | Missing | % |
|---|---|---|
| `application_date` | 719 | 7.18% |
| `last_contact_channel` | 1,185 | 11.84% |
| `plan_tier_requested` | 526 | 5.26% |
| `broker_channel` | 470 | 4.70% |
| `legacy_propensity_score` | 908 | 9.07% |
| `outreach_notes` | 1,453 | 14.52% |
| `salary` | 0 | 0.00% |

Missing categoricals were mapped to an explicit `"Unknown"` category rather than imputed, to preserve missingness as a potentially informative signal without fabricating values. Missing `application_date` is retained as `has_application_date = 0` plus `NaN` in the derived date fields (left for the tree model to handle natively) rather than dropped or imputed.

### Operational field cleaning
- **`application_date`**: mixed formats (`YYYY-MM-DD` and `DD/MM/YYYY`), parsed with `pd.to_datetime(..., format='mixed', dayfirst=True)`.
- **`last_contact_date` vs `application_date`**: 61 rows show contact recorded *after* the application date — chronologically valid as late follow-ups, but flagged explicitly via a boolean `contact_after_app` rather than silently trusted, plus a `days_contact_to_app` numeric feature.
- **`last_contact_channel`**: casing/spelling variants (`Email`, `EMAIL`, `e-mail`, `Call`, `SMS`, `sms`, …) mapped to 4 canonical values: `Email`, `Phone`, `SMS`, `Unknown`.
- **`plan_tier_requested`**: free text (`basic`, `STANDARD`, `silver plan`, `Gold Plan`, …) mapped to canonical tiers: `Basic`, `Bronze`, `Silver`, `Standard`, `Premium`, `Gold`, `Unknown`.
- **`broker_channel`**: missing values mapped to `Unknown`.
- **`state_mandate_level`** (region table): dirty casing (`High`, `low`, `MED`) normalized to `High`/`Medium`/`Low`.

### Sentinel handling — `prior_year_enrolled`
Raw values are `{-1, 0, 1}`, not a clean binary. `-1` means "no prior-year record" (new hire) — a sentinel, not a literal enrollment status. Split into two independent binary flags to avoid imposing a false ordinal relationship:
- `no_prior_record` = 1 if raw value is `-1` (4,042 rows / 40.4%)
- `prior_year_enrolled_clean` = 1 if raw value is `1` (3,571 rows / 35.7%)

### Other data-quality flags
- **`salary`**: 4 implausibly low values (< $15,000) identified for review; no rows dropped for this on its own.
- **`tenure_years` vs `age`**: 385 rows (3.9%) have `tenure_years > age - 18`, i.e. imply starting employment before age 18. Flagged as `tenure_inconsistent = 1` rather than silently dropped or clipped, preserving the raw value for transparency.

## 2. Feature Choices

### Leakage screen
- **`legacy_propensity_score` — FORBIDDEN / LEAKAGE.** Correlation with `enrolled` = 0.976; used alone, this single field achieves **AUC = 1.0000** (mean 0.152 for non-enrolled vs 0.882 for enrolled). It also differs suspiciously between duplicate-ID pairs correlated with their conflicting labels. This is a rules-engine score "captured at contact time" that almost certainly encodes post-hoc knowledge of the outcome. **Dropped entirely** and explicitly refused by the agent if requested (see §4).
- **`hist_enrollment_rate_region`** — computed from the same data scrape as the target. Kept as a feature (correlation with target is negligible, ~0.01), since it's a coarse regional macro statistic rather than a per-row leak, but flagged as **analysis-caution**: a stricter implementation would compute this out-of-fold. Noted as a limitation (§6).
- **`outreach_notes`** — free text, excluded from the feature set entirely (not vectorized) to avoid inadvertently encoding post-decision commentary into the model.
- **`n_employees_region`, `avg_salary_region`** — excluded to avoid regional-aggregate discrepancies introduced by the deduplication step changing region-level counts.

### The "too-perfect" finding
Independent investigation (a simple decision tree, depth 5, on just `salary`, `age`, `has_dependents`, `employment_type`) achieved **100% accuracy on a held-out test split** — not just training data. Cross-validated 5-fold, the full 23-feature model (including demographics) also produced a stable **1.0000 ROC-AUC across every fold**, even with a heavily regularized shallow model. This rules out both a single-column leak and train/test contamination (verified: 0 overlapping records between `train.csv`/`test.csv`), pointing instead to the practice dataset's `enrolled` label being a near-deterministic function of a handful of features by construction, not classical leakage.

### Fairness / compliance checkpoint (required)
`age`, `gender`, `marital_status` are present and mildly-to-moderately predictive (`has_dependents_bin` correlation with target = 0.45; `salary` = 0.37; `age` = 0.27 alone, but a strong contributor in combination — the near-perfect 1.0000 AUC above required `age` to be included).

**Decision: demographic features (`age`, `gender`, `marital_status`) are excluded from the final model's training inputs.**

**Rationale:**
1. Citing gender, age, or marital status to justify differential outreach is ethically and legally risky, even when the correlation is real rather than spurious.
2. Excluding them removes the possibility of an outreach decision being (or appearing to be) driven by a protected attribute, without requiring post-hoc bias correction.
3. The measured cost is small and honest: ROC-AUC drops from **1.0000** (with demographics) to **0.9682** (without) — a real, quantified trade-off rather than an assumed one.
4. Demographics are **retained in the raw/processed dataset for post-modeling fairness auditing only**, never as model input (see below).

**Subgroup performance breakdown** (predicted vs. actual enrollment rate, on the held-out test set, computed from raw metadata joined back in only for this audit — not used in training):

| Gender | n | Actual rate | Predicted rate | Avg. probability |
|---|---|---|---|---|
| Female | 949 | 60.2% | 64.7% | 0.594 |
| Male | 979 | 63.5% | 67.2% | 0.624 |
| Other | 71 | 57.8% | 59.2% | 0.554 |

| Age group | n | Actual rate | Predicted rate | Avg. probability |
|---|---|---|---|---|
| 20–35 | 595 | 41.3% | 60.2% | 0.553 |
| 36–50 | 723 | 68.9% | 66.5% | 0.615 |
| 51–65 | 681 | 72.0% | 69.8% | 0.646 |

| Marital status | n | Actual rate | Predicted rate | Avg. probability |
|---|---|---|---|---|
| Divorced | 214 | 56.5% | 59.8% | 0.534 |
| Married | 885 | 61.5% | 65.9% | 0.608 |
| Single | 775 | 63.1% | 67.0% | 0.620 |
| Widowed | 125 | 64.0% | 67.2% | 0.644 |

**Comment:** gender and marital-status subgroups are closely tracked (all within ~4 points of actual rate) — no group looks meaningfully disadvantaged. The one gap worth flagging honestly: the **20–35 age group is over-predicted by ~19 points** (41.3% actual vs. 60.2% predicted) relative to the other two age bands, which stay within ~3 points. Even with `age` excluded from training, this pattern likely comes through correlated features (`tenure_years`, `salary`, `prior_year_enrolled`), which are legitimately younger-skewed. This means the model may over-prioritize younger employees for outreach relative to their true likelihood — worth monitoring, not something we can fully rule out without a stricter fairness-constrained model, which is out of scope given the timeline (see §6).

## 3. Model, Baselines & Evaluation

**Split:** stratified 80/20 (`random_state=42`) — this is cross-sectional data (no temporal ordering to respect), so a plain stratified split is appropriate. Train: 7,993 rows (61.73% enrolled). Test: 1,999 rows (61.73% enrolled).

**Model:** `LightGBMClassifier` (`n_estimators=150, learning_rate=0.05, max_depth=5, num_leaves=31`), selected over XGBoost after 5-fold stratified CV (LightGBM 0.9731 ± 0.0016 vs. XGBoost 0.9731 ± 0.0013 — effectively tied; LightGBM chosen for faster training and native categorical support, avoiding manual one-hot encoding).

**Final test-set metrics (20 features, no demographics, no leaky fields):**

| Metric | Value |
|---|---|
| ROC-AUC | 0.9682 |
| PR-AUC | 0.9771 |
| Accuracy | 92.40% |
| Precision | 0.9117 |
| Recall | 0.9708 |
| F1 | 0.9403 |
| Brier Score | 0.0544 |

**Baseline comparison:**

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Majority-class baseline | 61.73% | — |
| Naive rule (`predict 1 if has_dependents == Yes`) | 74.04% | 0.7301 |
| **Final LightGBM model** | **92.40%** | **0.9682** |

The final model clearly beats both naive baselines.

**Business-relevant ranking metric (Precision@K / Lift@K),** using each region's `hr_outreach_capacity` scaled to the test-fold size:

| Region | Test n | K | Base rate | Precision@K | Lift@K |
|---|---|---|---|---|---|
| Midwest | 502 | 94 | 64.3% | 100.0% | 1.55× |
| Northeast | 531 | 30 | 62.3% | 100.0% | 1.60× |
| South | 466 | 65 | 60.7% | 100.0% | 1.65× |
| West | 500 | 87 | 59.4% | 98.9% | 1.66× |

Within each region's outreach budget, the model's top-ranked candidates are enrolled at close to 100% — a 1.5–1.7× lift over the region's baseline enrollment rate. This is the metric that most directly answers the business question: "if the benefits team can only contact K people, how much better is this than contacting K people at random?"

## 4. Agent Design

**Tools** (`src/tools/`): `predict_tool.py` (`predict_employee_enrollment`), `rank_tool.py` (`rank_employees`, capacity-aware), `explain_tool.py` (`explain_prediction`), plus `refusal_guardrails.py` for general safety checks. Interface: `src/cli.py` (predict / rank / explain / query subcommands) and an optional Streamlit dashboard (`src/app.py`) wrapping the same tools.

**Refusal rule 1 — target leakage.** Both `predict_tool.py` and `explain_tool.py` explicitly refuse requests referencing `legacy_propensity_score`, returning a structured `{"status": "refusal", "refusal_type": "TARGET_LEAKAGE_REFUSAL", ...}` response rather than silently ignoring the field.

**Refusal rule 2 — protected-attribute-safe explanations.** `explain_prediction` generates narrative explanations using only compliant drivers (`has_dependents`, `salary`, `employment_type`, `plan_tier_requested`, etc.) and a `PROTECTED_ATTRIBUTES` filter as a second safeguard. Since the underlying model itself excludes `age`/`gender`/`marital_status` (§2), this requirement is satisfied at the model level, not just the text-generation level — the strongest form of the guarantee.

**Example transcripts** (5 demo queries, run via `src/cli.py` / `src/app.py`):
1. `python src/cli.py rank --region Northeast --capacity` — top outreach priorities in the Northeast, capped at 151 candidates.
2. `python src/cli.py predict 17825` + `python src/cli.py explain 17825` — single-employee prediction and plain-language explanation.
3. Region profile lookup (Streamlit "Region Profile" tab / `lookup_region_profile("Midwest")`) — returns capacity, historical enrollment rate, premium cost, etc.
4. Attempting to request `legacy_propensity_score` directly — explicit refusal returned.
5. "What's wrong with this raw row?" — raw-vs-cleaned side-by-side diagnostic (Streamlit "Data Quality Check" tab), surfacing sentinel/chronology/categorical flags for a selected employee.

## 5. Feature Availability at Prediction Time

Region-level fields (`hist_enrollment_rate_region`, `avg_premium_cost_usd`, `benefits_broker_rating`, `hr_outreach_capacity`, `open_enrollment_window_days`, `state_mandate_level`) are static per-window lookups, realistically available before any individual prediction is made. `n_employees_region` and `avg_salary_region` were excluded not because of timing but to avoid drift from the deduplication step changing effective regional counts.


