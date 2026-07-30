"""
Inspect exact synthetic deterministic formula for `enrolled` label.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("data/employees_raw.csv")
df = df.drop_duplicates(subset=['employee_id'], keep=False).dropna(subset=['salary']).copy()

# Let's inspect the rules of the decision tree depth 4 or 5
from sklearn.tree import DecisionTreeClassifier, export_text

df['has_dep_bin'] = (df['has_dependents'] == 'Yes').astype(int)

# Employment type encoding:
# Contract = 0, Full-time = 1, Part-time = 2 ? Let's check cat codes
df['emp_type_cat'] = df['employment_type'].astype('category')
print("Cat categories:", df['emp_type_cat'].cat.categories)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
X = df[['salary', 'age', 'has_dep_bin', 'emp_type_cat'].copy()]
X['emp_type_cat'] = X['emp_type_cat'].cat.codes

dt.fit(X, df['enrolled'])
print("Accuracy:", dt.score(X, df['enrolled']))

# Check if salary > 60000 or salary ~ 60000 is a boundary
print("Salary min/max/quantile around 60000:")
print(df[df['salary'].between(59990, 60050)][['salary', 'age', 'has_dependents', 'employment_type', 'enrolled']])

# Check how many samples are misclassified by a simple depth 4 tree
print("\nMisclassified by depth 5 tree:", (dt.predict(X) != df['enrolled']).sum())
