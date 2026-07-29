import pandas as pd
df = pd.read_csv("employees_raw.csv")
region_df = pd.read_csv("region_benefit_profiles.csv")

# Drop all duplicates of employee_id (or keep first, let's see both)
print("--- Average salary comparison (Drop both duplicate rows) ---")
df_drop_both = df.drop_duplicates(subset=['employee_id'], keep=False)
print(df_drop_both.groupby('region')['salary'].mean())

print("\n--- Average salary comparison (Keep first duplicate row) ---")
df_keep_first = df.drop_duplicates(subset=['employee_id'], keep='first')
print(df_keep_first.groupby('region')['salary'].mean())

print("\nRegion profiles average salary:")
print(region_df.set_index('region')['avg_salary_region'])
