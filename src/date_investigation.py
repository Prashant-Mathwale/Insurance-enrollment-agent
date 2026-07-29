import os
import pandas as pd

def investigate_dates():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    employees_path = os.path.join(current_dir, "../data/employees_raw.csv")
    df = pd.read_csv(employees_path)
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['employee_id'], keep=False)
    
    # Parse dates robustly
    df['app_date_clean'] = pd.to_datetime(df['application_date'], format='mixed', dayfirst=True, errors='coerce')
    df['contact_date_clean'] = pd.to_datetime(df['last_contact_date'], format='mixed', errors='coerce')
    
    print(f"Total rows: {len(df)}")
    print(f"Null clean application_date: {df['app_date_clean'].isnull().sum()}")
    print(f"Null clean last_contact_date: {df['contact_date_clean'].isnull().sum()}\n")
    
    # Check chronological relation
    has_both_dates = df['app_date_clean'].notnull() & df['contact_date_clean'].notnull()
    df_both = df[has_both_dates].copy()
    
    df_both['days_contact_to_app'] = (df_both['app_date_clean'] - df_both['contact_date_clean']).dt.days
    
    contact_after_app = df_both[df_both['days_contact_to_app'] < 0]
    print(f"Number of rows where last_contact_date > application_date: {len(contact_after_app)}")
    print(f"Percentage of rows with both dates where this happens: {len(contact_after_app)/len(df_both)*100:.2f}%")
    print(f"Range of negative days (contact after app): {contact_after_app['days_contact_to_app'].min()} to {contact_after_app['days_contact_to_app'].max()}")
    print("")
    
    print("=== Target enrollment rate comparison ===")
    print("Rows where last_contact_date <= application_date:")
    print(df_both[df_both['days_contact_to_app'] >= 0]['enrolled'].value_counts(normalize=True))
    print("\nRows where last_contact_date > application_date:")
    print(contact_after_app['enrolled'].value_counts(normalize=True))
    print("")
    
    # Let's inspect a few records to see if a date swap makes sense
    # E.g. last_contact_date is 2024-07-17, application_date was 07/08/2024 (parsed as August 7th or July 8th?)
    # Wait, 07/08/2024 with dayfirst=True parses as 7th of August 2024.
    # If dayfirst=False, it would parse as July 8th, which is before July 17th.
    # Let's check a few cases of contact after application
    print("=== Sample rows with contact after application ===")
    print(contact_after_app[['employee_id', 'application_date', 'last_contact_date', 'app_date_clean', 'contact_date_clean', 'days_contact_to_app', 'enrolled']].head(10))

if __name__ == "__main__":
    investigate_dates()
