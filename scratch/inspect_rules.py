import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from baseline_model import prepare_data

X_train, y_train, X_test, y_test = prepare_data()

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
# Encode employment_type as ordinal for quick tree
X_tr = X_train.copy()
X_tr['employment_type_code'] = X_tr['employment_type'].astype('category').cat.codes

features = ['has_dependents_bin', 'age', 'salary', 'employment_type_code']
dt.fit(X_tr[features], y_train)

print(f"Decision Tree (depth 4) Accuracy on Train: {dt.score(X_tr[features], y_train):.4f}")
print("\nTree Structure:")
print(export_text(dt, feature_names=features))
