"""
Find sample IDs from both train split and held-out test split.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

raw = pd.read_csv("data/employees_raw.csv")
dedup = raw.drop_duplicates(subset=['employee_id'], keep=False)

train, test = train_test_split(dedup, test_size=0.20, random_state=42, stratify=dedup['enrolled'])

print("=== SAMPLE IDs FROM HELD-OUT TEST SPLIT (20%) ===")
print(test[['employee_id', 'age', 'salary', 'region', 'enrolled']].head(10))

print("\n=== SAMPLE IDs FROM TRAIN SPLIT (80%) ===")
print(train[['employee_id', 'age', 'salary', 'region', 'enrolled']].head(10))
