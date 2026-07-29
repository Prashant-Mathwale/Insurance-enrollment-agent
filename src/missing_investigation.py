import os
import pandas as pd

def investigate_missing_values():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    employees_path = os.path.join(current_dir, "../data/employees_raw.csv")
    df = pd.read_csv(employees_path)
    
    # Apply our approved duplicate resolution policy: drop the duplicates
    df = df.drop_duplicates(subset=['employee_id'], keep=False)
    
    print("=== Missing Values Profile (Excluding Duplicates) ===")
    missing_counts = df.isnull().sum()
    missing_pcts = (df.isnull().sum() / len(df)) * 100
    missing_info = pd.DataFrame({'Counts': missing_counts, 'Percentage': missing_pcts})
    print(missing_info[missing_info['Counts'] > 0].sort_values(by='Counts', ascending=False))
    print("")

    # Investigate relationships
    print("=== Correlation of missingness with enrollment target ===")
    for col in ['application_date', 'last_contact_channel', 'plan_tier_requested', 'broker_channel', 'legacy_propensity_score', 'outreach_notes']:
        is_missing = df[col].isnull().astype(int)
        corr = is_missing.corr(df['enrolled'])
        print(f"Missingness in '{col}' correlation with enrolled: {corr:.4f}")
    print("")

    # Check cross-tabulation of missing values
    print("=== Co-occurrence of missing values ===")
    missing_df = df.isnull()
    print("Count of rows where multiple columns are missing:")
    for col1 in ['application_date', 'last_contact_channel', 'plan_tier_requested', 'broker_channel']:
        for col2 in ['application_date', 'last_contact_channel', 'plan_tier_requested', 'broker_channel']:
            if col1 < col2:
                co_occur = (missing_df[col1] & missing_df[col2]).sum()
                print(f"  {col1} AND {col2} are both missing in: {co_occur} rows")

if __name__ == "__main__":
    investigate_missing_values()
