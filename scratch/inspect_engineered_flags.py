"""
Inspect engineered dataframe flags and employee_id.
"""
import sys
sys.path.append('src')
from feature_engineering import engineer_features

df_eng, _ = engineer_features()
print("Engineered dataframe columns:")
print(list(df_eng.columns))

messy_rows = df_eng[
    (df_eng['has_application_date'] == 0) |
    (df_eng['contact_after_app'] == 1) |
    (df_eng['tenure_inconsistent'] == 1) |
    (df_eng['no_prior_record'] == 1) |
    (df_eng['last_contact_channel_clean'] == 'Unknown') |
    (df_eng['plan_tier_requested_clean'] == 'Unknown')
]

print(f"\nTotal messy rows with at least one flag = 1 or Unknown: {len(messy_rows)}")
print("Sample messy employee IDs:", messy_rows['employee_id'].head(10).tolist())
