"""
Find exact deterministic formula for enrolled.
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv("data/employees_raw.csv")

# Let's inspect tree on all rows (including duplicates or missing salary if any)
df['has_dep_bin'] = (df['has_dependents'] == 'Yes').astype(int)
df['emp_type_code'] = df['employment_type'].astype('category').cat.codes
df_valid = df.dropna(subset=['salary']).copy()

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
X = df_valid[['salary', 'age', 'has_dep_bin', 'emp_type_code']]
y = df_valid['enrolled']
dt.fit(X, y)

print("Decision Tree depth 5 accuracy on entire dataset (excluding 8 missing salary rows):", dt.score(X, y))
print(export_text(dt, feature_names=['salary', 'age', 'has_dep_bin', 'emp_type_code']))

# Check if there are any errors if we test depth 6, 7...
for d in range(1, 8):
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X, y)
    print(f"Depth {d} accuracy: {clf.score(X, y):.6f}")
