"""
Comprehensive Audit Script for Insurance Enrollment Agent
Inspects data, features, performance, documentation, duplicates, missing values, split, and pipeline.
"""
import os
import sys
import pandas as pd
import numpy as np

# Load raw datasets
emp_raw = pd.read_csv("data/employees_raw.csv")
reg_raw = pd.read_csv("data/region_benefit_profiles.csv")

print("=== RAW DATA SHAPES ===")
print("employees_raw:", emp_raw.shape)
print("region_benefit_profiles:", reg_raw.shape)

print("\n=== COLUMN NAMES ===")
print("employees_raw columns:", list(emp_raw.columns))
print("region_benefit_profiles columns:", list(reg_raw.columns))

# --- 1. Target & Feature Relationships ---
print("\n=== TARGET DISTRIBUTION ===")
print(emp_raw['enrolled'].value_counts(dropna=False, normalize=True))

# Check has_application_date vs enrolled
app_date_notnull = emp_raw['application_date'].notnull()
print("\n=== application_date presence vs enrolled ===")
ct = pd.crosstab(app_date_notnull, emp_raw['enrolled'], margins=True)
print(ct)

# Check application_date missingness enrollment rate
print("\nEnrollment rate by application_date presence:")
print(emp_raw.groupby(app_date_notnull)['enrolled'].mean())

# Check plan_tier_requested vs enrolled
print("\n=== plan_tier_requested vs enrolled ===")
print(pd.crosstab(emp_raw['plan_tier_requested'].fillna('Missing'), emp_raw['enrolled'], normalize='index'))

# Check last_contact_channel vs enrolled
print("\n=== last_contact_channel vs enrolled ===")
print(pd.crosstab(emp_raw['last_contact_channel'].fillna('Missing'), emp_raw['enrolled'], normalize='index'))

# Check outreach_notes vs enrolled
print("\n=== outreach_notes vs enrolled ===")
print(pd.crosstab(emp_raw['outreach_notes'].fillna('Missing'), emp_raw['enrolled'], normalize='index'))

# Check legacy_propensity_score vs enrolled
print("\n=== legacy_propensity_score vs enrolled ===")
print(emp_raw.groupby('enrolled')['legacy_propensity_score'].describe())

# Check deterministic synthetic patterns in age, salary, has_dependents, employment_type
print("\n=== SYNTHETIC / DETERMINISTIC PATTERN CHECK ===")
# Let's inspect tree decision boundary on demographic features alone
from sklearn.tree import DecisionTreeClassifier, export_text

df_clean = emp_raw.drop_duplicates(subset=['employee_id'], keep=False).dropna(subset=['salary']).copy()
df_clean['has_dep_num'] = (df_clean['has_dependents'] == 'Yes').astype(int)
df_clean['emp_type_num'] = df_clean['employment_type'].astype('category').cat.codes

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
X_sub = df_clean[['salary', 'age', 'has_dep_num', 'emp_type_num']]
y_sub = df_clean['enrolled']
dt.fit(X_sub, y_sub)
acc = dt.score(X_sub, y_sub)
print(f"DecisionTree (depth 5) on [salary, age, has_dependents, employment_type] Accuracy: {acc:.6f}")

dt_full = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_full.fit(X_sub, y_sub)
print(f"DecisionTree (depth 10) on [salary, age, has_dependents, employment_type] Accuracy: {dt_full.score(X_sub, y_sub):.6f}")

print("\nTree text (depth 5):")
print(export_text(dt, feature_names=['salary', 'age', 'has_dep_num', 'emp_type_num']))
