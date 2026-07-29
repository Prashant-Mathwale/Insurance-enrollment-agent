"""
Milestone 7: Categorical Cleaning
Investigates and normalises dirty categorical columns:
  - last_contact_channel
  - plan_tier_requested
  - broker_channel
  - state_mandate_level (from region_benefit_profiles)
"""
import os
import pandas as pd

def clean_categoricals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    employees_path = os.path.join(current_dir, "../data/employees_raw.csv")
    regions_path   = os.path.join(current_dir, "../data/region_benefit_profiles.csv")
    df     = pd.read_csv(employees_path)
    df_reg = pd.read_csv(regions_path)

    # Drop duplicates
    df = df.drop_duplicates(subset=['employee_id'], keep=False)

    # ------------------------------------------------------------------ #
    # 1. last_contact_channel
    # ------------------------------------------------------------------ #
    print("=== BEFORE: last_contact_channel ===")
    print(df['last_contact_channel'].value_counts(dropna=False))

    channel_map = {
        # Email variants
        'EMAIL':  'Email', 'email': 'Email', 'e-mail': 'Email', 'Email': 'Email',
        # Phone / Call variants
        'PHONE':  'Phone', 'phone': 'Phone', 'Phone': 'Phone', 'Call': 'Phone',
        # SMS / Text variants
        'SMS':    'SMS',   'sms':   'SMS',   'Text':  'SMS',
        # Explicit none
        'none':   'Unknown',
    }

    df['last_contact_channel_clean'] = (
        df['last_contact_channel']
        .map(channel_map)           # maps known values
        .fillna('Unknown')          # NaN → Unknown
    )

    print("\n=== AFTER: last_contact_channel_clean ===")
    print(df['last_contact_channel_clean'].value_counts(dropna=False))

    # Verify no raw values leaked through unmapped
    assert df['last_contact_channel_clean'].isnull().sum() == 0, "Unexpected nulls!"

    # ------------------------------------------------------------------ #
    # 2. plan_tier_requested
    # ------------------------------------------------------------------ #
    print("\n=== BEFORE: plan_tier_requested ===")
    print(df['plan_tier_requested'].value_counts(dropna=False))

    tier_map = {
        # Standard / Silver
        'STANDARD': 'Standard', 'Standard': 'Standard', 'standard': 'Standard',
        'Silver':   'Silver',   'silver plan': 'Silver',
        # Bronze
        'Bronze': 'Bronze',
        # Basic
        'BASIC': 'Basic', 'Basic': 'Basic', 'basic': 'Basic',
        # Premium
        'premium plan': 'Premium', 'Premium': 'Premium', 'PREMIUM': 'Premium',
        # Gold
        'gold': 'Gold', 'Gold Plan': 'Gold', 'Gold': 'Gold',
    }

    df['plan_tier_requested_clean'] = (
        df['plan_tier_requested']
        .map(tier_map)
        .fillna('Unknown')
    )

    print("\n=== AFTER: plan_tier_requested_clean ===")
    print(df['plan_tier_requested_clean'].value_counts(dropna=False))

    assert df['plan_tier_requested_clean'].isnull().sum() == 0, "Unexpected nulls!"

    # ------------------------------------------------------------------ #
    # 3. broker_channel (only NaN needs handling)
    # ------------------------------------------------------------------ #
    print("\n=== BEFORE: broker_channel ===")
    print(df['broker_channel'].value_counts(dropna=False))

    df['broker_channel_clean'] = df['broker_channel'].fillna('Unknown')

    print("\n=== AFTER: broker_channel_clean ===")
    print(df['broker_channel_clean'].value_counts(dropna=False))

    # ------------------------------------------------------------------ #
    # 4. state_mandate_level (region table)
    # ------------------------------------------------------------------ #
    print("\n=== BEFORE: state_mandate_level ===")
    print(df_reg['state_mandate_level'].value_counts(dropna=False))

    mandate_map = {
        'High': 'High', 'MED': 'Medium', 'low': 'Low', 'Low': 'Low',
    }

    df_reg['state_mandate_level_clean'] = df_reg['state_mandate_level'].map(mandate_map)

    print("\n=== AFTER: state_mandate_level_clean ===")
    print(df_reg['state_mandate_level_clean'].value_counts(dropna=False))

    unmapped = df_reg['state_mandate_level'][df_reg['state_mandate_level_clean'].isnull()]
    if len(unmapped) > 0:
        print(f"WARNING: Unmapped values: {unmapped.tolist()}")
    else:
        print("All values mapped successfully.")

    # ------------------------------------------------------------------ #
    # Summary of final canonical categories
    # ------------------------------------------------------------------ #
    print("\n=== Canonical Category Summary ===")
    print(f"last_contact_channel  -> {sorted(df['last_contact_channel_clean'].unique())}")
    print(f"plan_tier_requested   -> {sorted(df['plan_tier_requested_clean'].unique())}")
    print(f"broker_channel        -> {sorted(df['broker_channel_clean'].unique())}")
    print(f"state_mandate_level   -> {sorted(df_reg['state_mandate_level_clean'].dropna().unique())}")

if __name__ == "__main__":
    clean_categoricals()
