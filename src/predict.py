"""
Inference Module (Milestone 15)
Provides a clean API and CLI interface to load raw employee records, apply the full
preprocessing pipeline, and generate enrollment predictions / probabilities using
the trained LightGBM model (models/enrollment_model.joblib).
"""
import os
import joblib
import pandas as pd
import numpy as np

EXPECTED_FEATURES = [
    'salary', 'employment_type', 'region', 'tenure_years',
    'hist_enrollment_rate_region', 'avg_premium_cost_usd',
    'benefits_broker_rating', 'hr_outreach_capacity',
    'open_enrollment_window_days', 'state_mandate_level',
    'has_application_date', 'days_contact_to_app', 'contact_after_app',
    'no_prior_record', 'prior_year_enrolled_clean',
    'last_contact_channel_clean', 'plan_tier_requested_clean',
    'broker_channel_clean', 'has_dependents_bin', 'tenure_inconsistent'
]

REQUIRED_RAW_COLUMNS = ['region']

CATEGORICAL_LEVELS = {
    'employment_type': ['Contract', 'Full-time', 'Part-time'],
    'region': ['Midwest', 'Northeast', 'South', 'West'],
    'state_mandate_level': ['High', 'Low', 'Medium'],
    'last_contact_channel_clean': ['Email', 'Phone', 'SMS', 'Unknown'],
    'plan_tier_requested_clean': ['Basic', 'Bronze', 'Gold', 'Premium', 'Silver', 'Standard', 'Unknown'],
    'broker_channel_clean': ['Direct', 'Employer-Sponsored', 'Third-Party', 'Unknown']
}

def validate_input(input_data):
    """Validate input format and return a copy of DataFrame."""
    if input_data is None:
        raise ValueError("Input data cannot be None.")

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        if len(input_data) == 0:
            raise ValueError("Input list cannot be empty.")
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}. Expected DataFrame, dict, or list of dicts.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check for critical required columns
    missing_req = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required input column(s): {missing_req}")

    return df

def preprocess_raw_data(df_raw, df_reg=None, model=None):
    """
    Applies the exact cleaning and feature engineering pipeline to raw input records.
    Matches the logic in feature_engineering.py.
    """
    df = validate_input(df_raw)

    # Fill default columns if missing from partial raw input
    defaults = {
        'age': np.nan, 'gender': 'Unknown', 'marital_status': 'Unknown',
        'salary': np.nan, 'employment_type': 'Unknown', 'has_dependents': 'No',
        'tenure_years': np.nan, 'prior_year_enrolled': -1,
        'application_date': np.nan, 'last_contact_date': np.nan,
        'last_contact_channel': np.nan, 'plan_tier_requested': np.nan,
        'broker_channel': np.nan
    }
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Load region profiles if not passed
    if df_reg is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        regions_path = os.path.join(current_dir, "../data/region_benefit_profiles.csv")
        df_reg = pd.read_csv(regions_path)

    # Clean region table mandate level
    df_reg = df_reg.copy()
    mandate_map = {'High': 'High', 'MED': 'Medium', 'low': 'Low', 'Low': 'Low'}
    df_reg['state_mandate_level'] = df_reg['state_mandate_level'].map(mandate_map)

    # Merge on region
    df = df.merge(df_reg, on='region', how='left')

    # Date parsing & date features
    df['app_date']     = pd.to_datetime(df['application_date'], format='mixed', dayfirst=True, errors='coerce')
    df['contact_date'] = pd.to_datetime(df['last_contact_date'], format='mixed', errors='coerce')

    df['has_application_date'] = df['app_date'].notnull().astype(int)
    df['days_contact_to_app']  = (df['app_date'] - df['contact_date']).dt.days
    df['contact_after_app']    = ((df['contact_date'] > df['app_date']) & df['app_date'].notnull()).astype(int)

    # Sentinel value encoding
    df['no_prior_record']           = (df['prior_year_enrolled'] == -1).astype(int)
    df['prior_year_enrolled_clean'] = (df['prior_year_enrolled'] == 1).astype(int)

    # Categorical normalization
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

    # Binary flags
    df['has_dependents_bin']  = (df['has_dependents'] == 'Yes').astype(int)
    df['tenure_inconsistent'] = (df['tenure_years'] > (df['age'] - 18)).astype(int)

    # Reorder and select features exactly matching expected order
    X = df[EXPECTED_FEATURES].copy()

    # Convert object / categorical columns using exact predefined category levels to prevent LightGBM mismatches
    for col, cats in CATEGORICAL_LEVELS.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=cats)

    emp_ids = df['employee_id'] if 'employee_id' in df.columns else None
    return emp_ids, X

def predict(df_raw, model=None, model_path=None):
    """
    Generate enrollment predictions and probabilities for raw input records.
    Returns DataFrame with employee_id (if present), predicted_probability, and predicted_class.
    """
    if model is None:
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path  = os.path.join(current_dir, "../models/enrollment_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        model = joblib.load(model_path)

    emp_ids, X = preprocess_raw_data(df_raw, model=model)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = pd.DataFrame({
        'predicted_probability': probs,
        'predicted_class': preds,
        'enrollment_probability': probs,
        'predicted_enrolled': preds
    })
    if emp_ids is not None:
        results.insert(0, 'employee_id', emp_ids.values)

    return results

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(current_dir, "../data/employees_raw.csv")

    df_sample = pd.read_csv(sample_path).head(10)
    res = predict(df_sample)
    print(res)
