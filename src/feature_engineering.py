"""
Milestone 9: Feature Engineering
Applies all cleaning decisions and engineers the final feature set.
Produces a clean, analysis-ready DataFrame and prints a summary.
"""
import os
import pandas as pd
import numpy as np

def engineer_features():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    employees_path = os.path.join(current_dir, "../data/employees_raw.csv")
    regions_path   = os.path.join(current_dir, "../data/region_benefit_profiles.csv")

    df_emp = pd.read_csv(employees_path)
    df_reg = pd.read_csv(regions_path)

    # ------------------------------------------------------------------ #
    # STEP 1 — Duplicate removal (Decision 1)
    # ------------------------------------------------------------------ #
    df_emp = df_emp.drop_duplicates(subset=['employee_id'], keep=False)

    # ------------------------------------------------------------------ #
    # STEP 2 — Region table cleaning + merge (Decisions 5, 6)
    # ------------------------------------------------------------------ #
    mandate_map = {'High': 'High', 'MED': 'Medium', 'low': 'Low', 'Low': 'Low'}
    df_reg['state_mandate_level'] = df_reg['state_mandate_level'].map(mandate_map)

    df = df_emp.merge(df_reg, on='region', how='left')
    assert len(df) == len(df_emp), "Row count changed after merge!"

    # ------------------------------------------------------------------ #
    # STEP 3 — Date parsing and date-derived features (Decision 4)
    # ------------------------------------------------------------------ #
    df['app_date']     = pd.to_datetime(df['application_date'], format='mixed', dayfirst=True, errors='coerce')
    df['contact_date'] = pd.to_datetime(df['last_contact_date'], format='mixed', errors='coerce')

    df['has_application_date'] = df['app_date'].notnull().astype(int)
    df['days_contact_to_app']  = (df['app_date'] - df['contact_date']).dt.days   # NaN where app_date missing
    df['contact_after_app']    = ((df['contact_date'] > df['app_date']) & df['app_date'].notnull()).astype(int)

    print(f"has_application_date == 1: {df['has_application_date'].sum()}")
    print(f"contact_after_app == 1:    {df['contact_after_app'].sum()}")
    print(f"days_contact_to_app NaN:   {df['days_contact_to_app'].isna().sum()}")
    print()

    # ------------------------------------------------------------------ #
    # STEP 4 — Sentinel value encoding (Decision 3)
    # ------------------------------------------------------------------ #
    df['no_prior_record']         = (df['prior_year_enrolled'] == -1).astype(int)
    df['prior_year_enrolled_clean'] = (df['prior_year_enrolled'] == 1).astype(int)

    dist = df.groupby(['no_prior_record', 'prior_year_enrolled_clean'])['enrolled'].mean()
    print("Enrollment rate by sentinel encoding:")
    print(dist.rename("enrollment_rate"))
    print()

    # ------------------------------------------------------------------ #
    # STEP 5 — Categorical cleaning (Decision 5)
    # ------------------------------------------------------------------ #
    channel_map = {
        'EMAIL': 'Email',  'email': 'Email',  'e-mail': 'Email',  'Email': 'Email',
        'PHONE': 'Phone',  'phone': 'Phone',  'Phone': 'Phone',   'Call':  'Phone',
        'SMS':   'SMS',    'sms':   'SMS',    'Text':  'SMS',
        'none':  'Unknown',
    }
    tier_map = {
        'STANDARD': 'Standard', 'Standard': 'Standard', 'standard': 'Standard',
        'Silver':   'Silver',   'silver plan': 'Silver',
        'Bronze':   'Bronze',
        'BASIC':    'Basic',    'Basic': 'Basic',    'basic': 'Basic',
        'premium plan': 'Premium', 'Premium': 'Premium', 'PREMIUM': 'Premium',
        'gold':     'Gold',    'Gold Plan': 'Gold',    'Gold': 'Gold',
    }

    df['last_contact_channel_clean'] = df['last_contact_channel'].map(channel_map).fillna('Unknown')
    df['plan_tier_requested_clean']  = df['plan_tier_requested'].map(tier_map).fillna('Unknown')
    df['broker_channel_clean']       = df['broker_channel'].fillna('Unknown')

    # ------------------------------------------------------------------ #
    # STEP 6 — Simple binary encoding
    # ------------------------------------------------------------------ #
    df['has_dependents_bin'] = (df['has_dependents'] == 'Yes').astype(int)

    # ------------------------------------------------------------------ #
    # STEP 7 — Tenure inconsistency flag
    # ------------------------------------------------------------------ #
    df['tenure_inconsistent'] = (df['tenure_years'] > (df['age'] - 18)).astype(int)
    print(f"tenure_inconsistent == 1: {df['tenure_inconsistent'].sum()} rows")
    print()

    # ------------------------------------------------------------------ #
    # FINAL — Assemble feature list and print summary
    # ------------------------------------------------------------------ #
    # Features to KEEP for modeling (leakage review in M10 will finalise this)
    candidate_features = [
        # Demographics (will be reviewed for compliance)
        'age', 'gender', 'marital_status',
        # Employment
        'salary', 'employment_type', 'region',
        'has_dependents_bin', 'tenure_years', 'tenure_inconsistent',
        # Prior enrollment (encoded)
        'no_prior_record', 'prior_year_enrolled_clean',
        # Contact info (cleaned)
        'last_contact_channel_clean', 'broker_channel_clean',
        'plan_tier_requested_clean',
        # Date-derived
        'has_application_date', 'days_contact_to_app', 'contact_after_app',
        # Region features (leakage review pending)
        'hist_enrollment_rate_region', 'avg_premium_cost_usd',
        'benefits_broker_rating', 'hr_outreach_capacity',
        'open_enrollment_window_days', 'state_mandate_level',
        # Outreach notes (informational only)
        'outreach_notes',
    ]

    # Features to EXCLUDE before model training
    excluded_features = [
        'employee_id', 'enrolled',              # ID and target
        'application_date', 'last_contact_date', # raw date strings
        'app_date', 'contact_date',              # intermediate parsed dates
        'prior_year_enrolled', 'has_dependents', # replaced by encoded versions
        'last_contact_channel', 'plan_tier_requested', 'broker_channel',  # replaced by _clean
        'legacy_propensity_score',               # candidate for leakage (M10)
        'n_employees_region', 'avg_salary_region',  # informational only (Decision 6)
    ]

    print("=== Feature Summary ===")
    print(f"Total candidate features: {len(candidate_features)}")
    print(f"Total excluded features:  {len(excluded_features)}")
    print()
    print("Null counts in candidate features:")
    print(df[candidate_features].isnull().sum()[df[candidate_features].isnull().sum() > 0])

    print("\n=== Final dataframe shape ===")
    print(df.shape)

    return df, candidate_features

if __name__ == "__main__":
    df, features = engineer_features()
    print("\nDone. Feature engineering complete.")
