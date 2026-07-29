import os
import pandas as pd

def investigate_sentinels():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    employees_path = os.path.join(current_dir, "../data/employees_raw.csv")
    df = pd.read_csv(employees_path)
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['employee_id'], keep=False)
    
    print("=== prior_year_enrolled distribution ===")
    counts = df['prior_year_enrolled'].value_counts(dropna=False)
    pcts = df['prior_year_enrolled'].value_counts(dropna=False, normalize=True) * 100
    print(pd.DataFrame({'Count': counts, 'Percentage': pcts}))
    print("")

    # Verify if -1 means "new hire" by checking tenure_years
    print("=== Average tenure_years by prior_year_enrolled ===")
    print(df.groupby('prior_year_enrolled')['tenure_years'].describe())
    print("")

    # Let's check if there are any new hires (-1) with tenure > 1 year
    new_hire_long_tenure = df[(df['prior_year_enrolled'] == -1) & (df['tenure_years'] >= 1.0)]
    print(f"Number of new hires (-1) with tenure >= 1.0 years: {len(new_hire_long_tenure)}")
    print("Tenure stats for new hires (-1):")
    print(df[df['prior_year_enrolled'] == -1]['tenure_years'].describe())
    print("")

    # Let's check correlation with target enrolled
    print("=== Target enrollment rate by prior_year_enrolled ===")
    print(df.groupby('prior_year_enrolled')['enrolled'].mean())
    print("")

if __name__ == "__main__":
    investigate_sentinels()
