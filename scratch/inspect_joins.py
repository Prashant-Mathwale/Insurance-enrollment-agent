import pandas as pd

df = pd.read_csv("employees_raw.csv")
region_df = pd.read_csv("region_benefit_profiles.csv")

print("--- Compare salaries ---")
emp_avg_salary = df.groupby('region')['salary'].mean()
print("Computed average salary by region from employees_raw:")
print(emp_avg_salary)
print("Region profiles average salary:")
print(region_df.set_index('region')['avg_salary_region'])

print("\n--- Compare employee counts ---")
emp_counts = df.groupby('region')['employee_id'].count()
print("Computed employee count by region from employees_raw:")
print(emp_counts)
print("Region profiles employee counts:")
print(region_df.set_index('region')['n_employees_region'])
