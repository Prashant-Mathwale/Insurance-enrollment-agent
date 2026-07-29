import pandas as pd
df = pd.read_csv("employees_raw.csv")
dup_mask = df.duplicated(subset=['employee_id'], keep=False)
duplicates = df[dup_mask].sort_values(by='employee_id')
print(duplicates[['employee_id', 'region']])
print(duplicates.groupby('region')['employee_id'].count() // 2)
