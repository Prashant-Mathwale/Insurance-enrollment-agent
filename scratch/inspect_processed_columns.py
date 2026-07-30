"""
Inspect columns of data/processed/employees_processed.csv
"""
import pandas as pd
df = pd.read_csv("data/processed/employees_processed.csv")
print("Columns in employees_processed.csv:")
print(list(df.columns))
print("\nSample row:")
print(df.iloc[0].to_dict())
