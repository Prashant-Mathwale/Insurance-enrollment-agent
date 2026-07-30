"""
Comprehensive Full Audit Verifier Script
Runs empirical verification checks across all 8 audit sections.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Configure sys.path so both root and src directory are searchable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for path in [PROJECT_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Set UTF-8 printing for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Package imports (src is marked as a package via src/__init__.py)
from src.predict import EXPECTED_FEATURES, predict
from src.feature_engineering import engineer_features

# Absolute paths based on project root for reliable file loading
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

raw_emp = pd.read_csv(os.path.join(DATA_DIR, "employees_raw.csv"))
raw_reg = pd.read_csv(os.path.join(DATA_DIR, "region_benefit_profiles.csv"))
processed_emp = pd.read_csv(os.path.join(DATA_DIR, "processed/employees_processed.csv"))
train_df = pd.read_csv(os.path.join(DATA_DIR, "processed/train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "processed/test.csv"))
model = joblib.load(os.path.join(MODELS_DIR, "enrollment_model.joblib"))

print("=" * 80)
print("AUDIT VERIFICATION RUNNER")
print("=" * 80)

# -------------------------------------------------------------------------
# TASK 4: DUPLICATE INVESTIGATION VERIFICATION
# -------------------------------------------------------------------------
print("\n--- TASK 4: DUPLICATE INVESTIGATION VERIFICATION ---")
dups = raw_emp[raw_emp.duplicated(subset=['employee_id'], keep=False)].sort_values('employee_id')
dup_ids = dups['employee_id'].unique()
print(f"Total duplicate rows found: {len(dups)} across {len(dup_ids)} unique employee_ids")

for emp_id in dup_ids:
    pair = dups[dups['employee_id'] == emp_id]
    differing_cols = []
    for col in pair.columns:
        if pair[col].nunique(dropna=False) > 1:
            differing_cols.append(col)
    
    salaries = pair['salary'].tolist()
    labels = pair['enrolled'].tolist()
    legacies = pair['legacy_propensity_score'].tolist()
    
    print(f"  Employee ID {emp_id}: Differing cols = {differing_cols} | Salaries = {salaries} | Enrolled = {labels} | Legacy Score = {legacies}")

# Confirm row counts after dropping keep=False
raw_dedup_none = raw_emp.drop_duplicates(subset=['employee_id'], keep=False)
print(f"Rows in raw: {len(raw_emp)} | Rows after drop_duplicates(keep=False): {len(raw_dedup_none)} | (Diff = {len(raw_emp) - len(raw_dedup_none)})")

# -------------------------------------------------------------------------
# TASK 5: MISSING VALUE VERIFICATION
# -------------------------------------------------------------------------
print("\n--- TASK 5: MISSING VALUE VERIFICATION ---")
print("Missing values in raw employees dataset:")
raw_nulls = raw_emp.isnull().sum()
print(raw_nulls[raw_nulls > 0])

# Check if salary missing rows were removed
salary_null_ids = raw_emp[raw_emp['salary'].isnull()]['employee_id'].tolist()
print(f"Employee IDs with missing salary in raw: {salary_null_ids} ({len(salary_null_ids)} rows)")

# Check if these IDs exist in processed_emp
in_processed = processed_emp['salary'].isnull().sum()
print(f"Missing salary in processed_emp: {in_processed}")
print(f"Total rows in processed_emp: {len(processed_emp)}")

# Calculate expected rows: 10008 raw - 16 duplicates = 9992
print(f"Calculation: 10008 raw - 16 duplicates = 9992.")
df_engineered, cand_feats = engineer_features()
print(f"Rows output by engineer_features(): {len(df_engineered)}")
print(f"Number of null salaries in engineer_features output: {df_engineered['salary'].isnull().sum()}")

# -------------------------------------------------------------------------
# TASK 6: TRAIN/TEST SPLIT AUDIT
# -------------------------------------------------------------------------
print("\n--- TASK 6: TRAIN/TEST SPLIT AUDIT ---")
train_ids = set(train_df['salary'].astype(str) + "_" + train_df['age'].astype(str) + "_" + train_df['region'])
test_ids = set(test_df['salary'].astype(str) + "_" + test_df['age'].astype(str) + "_" + test_df['region'])

overlap = train_ids.intersection(test_ids)
print(f"Overlap between train and test (based on salary+age+region signature): {len(overlap)}")

# -------------------------------------------------------------------------
# TASK 1 & 2: MODEL LEAKAGE & PERFECT PERFORMANCE
# -------------------------------------------------------------------------
print("\n--- TASK 1 & 2: MODEL LEAKAGE & PERFECT PERFORMANCE ---")
model_features = model.feature_name_
print(f"Model feature list ({len(model_features)} features):")
print(model_features)

# Check decision tree performance on just 4 demographic features
df_clean = raw_emp.drop_duplicates(subset=['employee_id'], keep=False).dropna(subset=['salary']).copy()
df_clean['has_dep_num'] = (df_clean['has_dependents'] == 'Yes').astype(int)
df_clean['emp_type_num'] = df_clean['employment_type'].astype('category').cat.codes

X_demo = df_clean[['salary', 'age', 'has_dep_num', 'emp_type_num']]
y_demo = df_clean['enrolled']

dt4 = DecisionTreeClassifier(max_depth=4, random_state=42)
dt4.fit(X_demo, y_demo)
print(f"DecisionTree (depth 4) on [salary, age, has_dependents, employment_type] Accuracy: {dt4.score(X_demo, y_demo):.6f}")

dt5 = DecisionTreeClassifier(max_depth=5, random_state=42)
dt5.fit(X_demo, y_demo)
print(f"DecisionTree (depth 5) on [salary, age, has_dependents, employment_type] Accuracy: {dt5.score(X_demo, y_demo):.6f}")

# -------------------------------------------------------------------------
# TASK 8: INFERENCE PIPELINE AUDIT
# -------------------------------------------------------------------------
print("\n--- TASK 8: INFERENCE PIPELINE AUDIT ---")
print(f"EXPECTED_FEATURES matches model.feature_name_: {EXPECTED_FEATURES == model_features}")

print("\nAudit verification script execution complete.")
