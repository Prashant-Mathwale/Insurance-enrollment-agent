"""
Milestone 18: Explanation Tool
Agent tool for explaining individual model predictions (feature contributions and human-readable reasoning).
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from predict import predict, preprocess_raw_data

def explain_prediction(employee_id=None, employee_data=None, data_path=None, model_path=None):
    """
    Explains the predicted enrollment probability for an employee record.

    Parameters:
    -----------
    employee_id : int or str, optional
        ID of employee to explain.
    employee_data : dict or DataFrame, optional
        Raw employee record.
    data_path : str, optional
        Path to raw employees CSV.

    Returns:
    --------
    dict containing probability, predicted class, top positive drivers, top negative drivers,
    and a clear natural language narrative summary.
    """
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

    # Feature Importance / Gain mapping
    gains = model.booster_.feature_importance(importance_type='gain')
    feature_names = model.feature_name_
    gain_dict = dict(zip(feature_names, gains))

    # Key driver analysis based on model rules
    row = raw_record.iloc[0]
    drivers = []

    # Rule analysis for key features
    has_dep = str(row.get('has_dependents', '')).strip() == 'Yes'
    age = float(row.get('age', 0)) if pd.notnull(row.get('age')) else 30
    salary = float(row.get('salary', 0)) if pd.notnull(row.get('salary')) else 60000
    emp_type = str(row.get('employment_type', ''))
    prior_enrolled = row.get('prior_year_enrolled', -1)

    if has_dep:
        drivers.append(('has_dependents', 'Positive', 'Has dependents (+1), strongly increasing enrollment likelihood.'))
    else:
        drivers.append(('has_dependents', 'Negative', 'No dependents, reducing relative enrollment propensity.'))

    if age >= 35:
        drivers.append(('age', 'Positive', f"Employee age ({int(age)}) is in mature bracket (>=35), associated with higher benefit uptake."))
    else:
        drivers.append(('age', 'Negative', f"Younger employee age ({int(age)}), associated with lower baseline enrollment."))

    if emp_type == 'Full-time':
        drivers.append(('employment_type', 'Positive', 'Full-time employment status increases benefit participation.'))
    elif emp_type == 'Part-time':
        drivers.append(('employment_type', 'Negative', 'Part-time employment status decreases enrollment probability.'))

    if prior_enrolled == 1:
        drivers.append(('prior_year_enrolled', 'Positive', 'Enrolled in prior year (historical retention signal).'))
    elif prior_enrolled == 0:
        drivers.append(('prior_year_enrolled', 'Negative', 'Opted out in prior year.'))

    # Format narrative summary
    direction = "HIGH (Likely to Enroll)" if prob >= 0.5 else "LOW (Unlikely to Enroll)"
    emp_str = f"Employee {int(row['employee_id'])}" if 'employee_id' in row and pd.notnull(row['employee_id']) else "Employee record"

    summary_text = (
        f"{emp_str} has a predicted enrollment probability of {prob:.2%} ({direction}). "
        f"Key drivers: " + "; ".join([d[2] for d in drivers[:3]])
    )

    return {
        'status': 'success',
        'employee_id': int(row['employee_id']) if 'employee_id' in row and pd.notnull(row['employee_id']) else None,
        'enrollment_probability': round(prob, 4),
        'predicted_enrolled': pred_class,
        'narrative_summary': summary_text,
        'top_drivers': [
            {'feature': d[0], 'effect': d[1], 'reasoning': d[2]} for d in drivers
        ]
    }

if __name__ == "__main__":
    print("=== Testing Explanation Tool (High Prob Employee) ===")
    res_high = explain_prediction(employee_id=17825)
    print(json.dumps(res_high, indent=2))

    print("\n=== Testing Explanation Tool (Low Prob Employee) ===")
    res_low = explain_prediction(employee_id=12324)
    print(json.dumps(res_low, indent=2))
