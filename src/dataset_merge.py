"""
Milestone 8: Dataset Merge
Joins employees_raw.csv and region_benefit_profiles.csv on 'region'.
Verifies row count, column count, no duplicates introduced,
and that merged statistics match the source tables.
"""
import os
import pandas as pd

def merge_datasets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    employees_path = os.path.join(current_dir, "../data/employees_raw.csv")
    regions_path   = os.path.join(current_dir, "../data/region_benefit_profiles.csv")

    df_emp = pd.read_csv(employees_path)
    df_reg = pd.read_csv(regions_path)

    # --- Apply our approved cleaning before merge ---
    # Drop duplicate employee_ids
    df_emp = df_emp.drop_duplicates(subset=['employee_id'], keep=False)

    # Normalize state_mandate_level in region table
    mandate_map = {'High': 'High', 'MED': 'Medium', 'low': 'Low', 'Low': 'Low'}
    df_reg['state_mandate_level'] = df_reg['state_mandate_level'].map(mandate_map)

    print("=== Pre-merge shapes ===")
    print(f"Employees: {df_emp.shape}")
    print(f"Regions:   {df_reg.shape}")
    print(f"Employee region keys: {sorted(df_emp['region'].unique())}")
    print(f"Region profile keys:  {sorted(df_reg['region'].unique())}")

    # Check for any region keys in employees not in region table
    emp_regions  = set(df_emp['region'].unique())
    reg_regions  = set(df_reg['region'].unique())
    unmatched    = emp_regions - reg_regions
    print(f"\nUnmatched employee regions (will produce NaN after join): {unmatched}")

    # --- Perform left join ---
    merged = df_emp.merge(df_reg, on='region', how='left')

    print("\n=== Post-merge shape ===")
    print(f"Merged: {merged.shape}")
    print(f"Expected rows: {len(df_emp)} (same as employees after duplicate removal)")
    assert len(merged) == len(df_emp), "Row count mismatch after merge!"
    assert merged['employee_id'].nunique() == len(merged), "Duplicate employee_ids introduced by merge!"
    print("Assertions passed: row count and uniqueness OK.\n")

    # --- Verify NaN count from join ---
    new_cols = [c for c in df_reg.columns if c != 'region']
    print("=== Missing values in region-derived columns after merge ===")
    print(merged[new_cols].isnull().sum())

    # --- Verify salary and employee count match source tables ---
    print("\n=== Sanity check: avg salary by region (merged vs region profile) ===")
    merged_avg  = merged.groupby('region')['salary'].mean().round(2)
    profile_avg = df_reg.set_index('region')['avg_salary_region']
    comparison  = pd.DataFrame({'merged_avg_salary': merged_avg, 'profile_avg_salary': profile_avg})
    comparison['diff'] = (comparison['merged_avg_salary'] - comparison['profile_avg_salary']).abs()
    print(comparison)

    print("\n=== Sanity check: employee count by region (merged vs region profile) ===")
    merged_count  = merged.groupby('region')['employee_id'].count()
    profile_count = df_reg.set_index('region')['n_employees_region']
    count_comp    = pd.DataFrame({'merged_count': merged_count, 'profile_count': profile_count})
    count_comp['diff'] = (count_comp['merged_count'] - count_comp['profile_count']).abs()
    print(count_comp)

    print("\n=== Final merged columns ===")
    print(list(merged.columns))

if __name__ == "__main__":
    merge_datasets()
