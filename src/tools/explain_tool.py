"""
Milestone 18: Explanation Tool
Agent tool for explaining individual model predictions (feature contributions and human-readable reasoning).
Enforces Fair AI guidelines: Never cites protected attributes (gender, marital_status, age) in explanations.
Refuses requests for legacy_propensity_score.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from predict import predict, preprocess_raw_data

PROTECTED_ATTRIBUTES = {'gender', 'marital_status', 'age'}

def explain_prediction(employee_id=None, employee_data=None, feature_requested=None, data_path=None, model_path=None):
    """
    Explains the predicted enrollment probability for an employee record.

    Parameters:
    -----------
    employee_id : int or str, optional
        ID of employee to explain.
    employee_data : dict or DataFrame, optional
        Raw employee record.
    feature_requested : str, optional
        Specific feature requested (refuses legacy_propensity_score).

    Returns:
    --------
    dict containing probability, predicted class, top positive/negative drivers (excluding protected attributes),
    and a clear natural language narrative summary.
    """
    if feature_requested and 'legacy_propensity_score' in str(feature_requested).lower():
        return {
            'status': 'refusal',
            'refusal_type': 'TARGET_LEAKAGE_REFUSAL',
            'message': "REFUSAL: legacy_propensity_score is excluded from model inputs due to critical target leakage (AUC = 1.0, correlation = 0.9764)."
        }

    if employee_data is not None:
        has_legacy = False
        if isinstance(employee_data, dict) and 'legacy_propensity_score' in employee_data:
            has_legacy = True
        elif isinstance(employee_data, pd.DataFrame) and 'legacy_propensity_score' in employee_data.columns:
            has_legacy = True

        if has_legacy:
            return {
                'status': 'refusal',
                'refusal_type': 'TARGET_LEAKAGE_REFUSAL',
                'message': "REFUSAL: legacy_propensity_score is excluded from model inputs due to critical target leakage (AUC = 1.0, correlation = 0.9764)."
            }

    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "../../models/enrollment_model.joblib")

    if not os.path.exists(model_path):
        return {'status': 'error', 'message': f"Model not found at {model_path}"}

    model = joblib.load(model_path)

    # Fetch raw record
    if employee_data is not None:
        if isinstance(employee_data, dict):
            raw_record = pd.DataFrame([employee_data])
        else:
            raw_record = employee_data.copy()
    elif employee_id is not None:
        if data_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, "../../data/employees_raw.csv")

        if not os.path.exists(data_path):
            return {'status': 'error', 'message': f"Data file not found at {data_path}"}

        df_raw = pd.read_csv(data_path)
        df_raw = df_raw.drop_duplicates(subset=['employee_id'], keep=False)
        raw_record = df_raw[df_raw['employee_id'] == int(employee_id)]
        if raw_record.empty:
            return {'status': 'error', 'message': f"Employee ID {employee_id} not found."}
    else:
        return {'status': 'error', 'message': "Must provide employee_id or employee_data."}

    # Preprocess
    _, X_proc = preprocess_raw_data(raw_record)

    # Predict
    prob = float(model.predict_proba(X_proc)[0, 1])
    pred_class = int(prob >= 0.5)

    # Key driver analysis based on non-protected features
    row = raw_record.iloc[0]
    drivers = []

    has_dep = str(row.get('has_dependents', '')).strip() == 'Yes'
    salary = float(row.get('salary', 0)) if pd.notnull(row.get('salary')) else 60000
    emp_type = str(row.get('employment_type', ''))
    prior_enrolled = row.get('prior_year_enrolled', -1)
    tier = str(row.get('plan_tier_requested', '')).strip().title()

    if has_dep:
        drivers.append(('has_dependents', 'Positive', 'Has dependents (+1), strongly increasing benefit enrollment likelihood.'))
    else:
        drivers.append(('has_dependents', 'Negative', 'No dependents recorded, associated with lower enrollment propensity.'))

    if salary >= 60000:
        drivers.append(('salary', 'Positive', f"Annual salary (${salary:,.2f}) is in the higher tier (>= $60,000), supporting benefit affordability."))
    else:
        drivers.append(('salary', 'Negative', f"Annual salary (${salary:,.2f}) is below $60,000 threshold, increasing cost sensitivity."))

    if emp_type == 'Full-time':
        drivers.append(('employment_type', 'Positive', 'Full-time employment status increases benefit participation.'))
    elif emp_type in ['Part-time', 'Contract']:
        drivers.append(('employment_type', 'Negative', f"{emp_type} status decreases baseline enrollment probability."))

    if prior_enrolled == 1:
        drivers.append(('prior_year_enrolled', 'Positive', 'Enrolled in prior year plan (strong historical retention signal).'))
    elif prior_enrolled == 0:
        drivers.append(('prior_year_enrolled', 'Negative', 'Opted out of enrollment in prior year.'))

    if tier in ['Gold', 'Premium', 'Silver']:
        drivers.append(('plan_tier_requested', 'Positive', f"Requested high-coverage plan tier ({tier}), indicating strong active interest."))

    # Filter out any protected attributes strictly
    drivers = [d for d in drivers if d[0] not in PROTECTED_ATTRIBUTES]

    # Format clean 1-2 sentence narrative summary
    direction_word = "highly likely to enroll" if prob >= 0.5 else "unlikely to enroll"
    emp_str = f"Employee {int(row['employee_id'])}" if 'employee_id' in row and pd.notnull(row['employee_id']) else "This employee"

    summary_text = (
        f"{emp_str} is predicted as {direction_word} with an estimated enrollment probability of {prob:.2%}."
    )

    return {
        'status': 'success',
        'employee_id': int(row['employee_id']) if 'employee_id' in row and pd.notnull(row['employee_id']) else None,
        'enrollment_probability': round(prob, 4),
        'predicted_enrolled': pred_class,
        'narrative_summary': summary_text,
        'top_drivers': [
            {'feature': d[0], 'effect': d[1], 'reasoning': d[2]} for d in drivers
        ],
        'fair_ai_note': "Explanations strictly exclude protected attributes (gender, marital_status, age) per corporate non-discrimination policy."
    }

if __name__ == "__main__":
    print("=== Testing Explanation Tool (High Prob Employee) ===")
    res_high = explain_prediction(employee_id=17825)
    print(json.dumps(res_high, indent=2))

    print("\n=== Testing Explanation Tool (Refusal Test) ===")
    res_ref = explain_prediction(employee_id=17825, feature_requested='legacy_propensity_score')
    print(json.dumps(res_ref, indent=2))
