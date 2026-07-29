import pandas as pd
import numpy as np

df = pd.read_csv("employees_raw.csv")

print("--- Check Salary ---")
print("Salaries below $10,000:")
low_salaries = df[df['salary'] < 10000]
print(f"Count: {len(low_salaries)}")
print(low_salaries[['employee_id', 'age', 'salary', 'employment_type']].head(10))

print("\n--- Check Tenure vs Age ---")
inconsistent_tenure = df[df['tenure_years'] > (df['age'] - 18)]
print(f"Count of tenure > (age - 18): {len(inconsistent_tenure)}")
print(inconsistent_tenure[['employee_id', 'age', 'tenure_years', 'employment_type']].head(10))
print("Max tenure years:")
print(df['tenure_years'].max())

print("\n--- Check Date Relationship (last_contact_date vs application_date) ---")
# Parse dates
df['app_date_parsed'] = pd.to_datetime(df['application_date'], errors='coerce', format='mixed')
df['contact_date_parsed'] = pd.to_datetime(df['last_contact_date'], errors='coerce', format='mixed')

contact_after_app = df[df['contact_date_parsed'] > df['app_date_parsed']]
print(f"Count of contact_date > application_date: {len(contact_after_app)}")
print(contact_after_app[['employee_id', 'application_date', 'last_contact_date', 'enrolled']].head(10))

print("\n--- Correlation of numeric features with enrolled ---")
numeric_cols = ['age', 'salary', 'tenure_years', 'prior_year_enrolled', 'legacy_propensity_score', 'enrolled']
print(df[numeric_cols].corr()['enrolled'])

print("\n--- Check outreach_notes ---")
print("A few outreach notes:")
print(df['outreach_notes'].dropna().unique()[:20])

print("\nValue counts of outreach_notes for enrolled = 1 vs 0:")
for val in df['enrolled'].unique():
    print(f"\nEnrolled = {val}")
    print(df[df['enrolled'] == val]['outreach_notes'].value_counts(dropna=False).head(10))
