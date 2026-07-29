import pandas as pd
import numpy as np

df = pd.read_csv("employees_raw.csv")

# Let's inspect tenure vs age further
print("--- Tenure vs Age details ---")
df['start_age'] = df['age'] - df['tenure_years']
print("Descriptive stats for start_age (age - tenure_years):")
print(df['start_age'].describe())

print("\nRows where start_age < 18:")
start_age_under_18 = df[df['start_age'] < 18]
print(f"Count: {len(start_age_under_18)}")
print("Let's look at a few where start_age is extremely low or negative:")
print(start_age_under_18[['employee_id', 'age', 'tenure_years', 'start_age']].sort_values(by='start_age').head(15))

print("\n--- Let's inspect low salaries ---")
print("Salaries distribution (percentiles):")
print(df['salary'].quantile([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]))

print("\nRows with salary < 15000:")
low_sal = df[df['salary'] < 15000]
print(low_sal[['employee_id', 'age', 'salary', 'employment_type', 'tenure_years']])

# Let's join the region table
region_df = pd.read_csv("region_benefit_profiles.csv")
print("\n--- Join check ---")
print(f"Employees region unique values: {df['region'].unique()}")
print(f"Region profiles unique values: {region_df['region'].unique()}")
# Check if there are any mismatch in region names
mismatches = set(df['region'].unique()) - set(region_df['region'].unique())
print(f"Mismatches: {mismatches}")

# Let's check state_mandate_level in region_benefit_profiles
print("\n--- state_mandate_level values in region profiles ---")
print(region_df['state_mandate_level'].value_counts())
