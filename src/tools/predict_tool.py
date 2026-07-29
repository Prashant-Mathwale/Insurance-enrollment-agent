"""
Milestone 16: Prediction Tool
Agent tool for predicting insurance enrollment probability for an individual employee
or a list of employee IDs / raw employee data.
"""
import os
import sys
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from predict import predict

def predict_employee_enrollment(employee_id=None, employee_data=None, data_path=None):
    """
    Predict insurance enrollment probability for an employee.

    Parameters:
    -----------
    employee_id : int or str or list, optional
        ID or list of IDs of employee(s) to predict from raw data CSV.
    employee_data : dict or list of dicts or DataFrame, optional
        Raw employee record(s) passed directly.
    data_path : str, optional
        Path to raw employees CSV (defaults to data/employees_raw.csv).

    Returns:
    --------
    dict containing prediction status, employee summary, probability, and decision.
    """
    if employee_data is not None:
        try:
            res_df = predict(employee_data)
            results = []
            for _, row in res_df.iterrows():
                rec = {
                    'employee_id': int(row['employee_id']) if 'employee_id' in row and pd.notnull(row['employee_id']) else None,
                    'enrollment_probability': round(float(row['predicted_probability']), 4),
                    'predicted_enrolled': int(row['predicted_class'])
                }
                results.append(rec)
            return {'status': 'success', 'count': len(results), 'predictions': results}
        except Exception as e:
            return {'status': 'error', 'message': f"Prediction failed for input data: {str(e)}"}

    if employee_id is not None:
        if data_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, "../../data/employees_raw.csv")

        if not os.path.exists(data_path):
            return {'status': 'error', 'message': f"Data file not found at {data_path}"}

        df_raw = pd.read_csv(data_path)
        df_raw = df_raw.drop_duplicates(subset=['employee_id'], keep=False)

        if isinstance(employee_id, (int, str)):
            target_ids = [int(employee_id)]
        else:
            target_ids = [int(i) for i in employee_id]

        matched = df_raw[df_raw['employee_id'].isin(target_ids)]
        if matched.empty:
            return {'status': 'error', 'message': f"Employee ID(s) {target_ids} not found in database."}

        res_df = predict(matched)

        results = []
        for idx, row in res_df.iterrows():
            emp_rec = matched[matched['employee_id'] == row['employee_id']].iloc[0]
            summary = {
                'employee_id': int(row['employee_id']),
                'age': int(emp_rec['age']) if pd.notnull(emp_rec['age']) else None,
                'salary': float(emp_rec['salary']) if pd.notnull(emp_rec['salary']) else None,
                'region': str(emp_rec['region']),
                'plan_tier_requested': str(emp_rec['plan_tier_requested']) if pd.notnull(emp_rec['plan_tier_requested']) else 'Unknown',
                'enrollment_probability': round(float(row['predicted_probability']), 4),
                'predicted_enrolled': int(row['predicted_class'])
            }
            results.append(summary)

        return {'status': 'success', 'count': len(results), 'predictions': results}

    return {'status': 'error', 'message': "Must provide either employee_id or employee_data."}

if __name__ == "__main__":
    print("=== Testing Prediction Tool with Employee ID ===")
    res_id = predict_employee_enrollment(employee_id=12324)
    print(json.dumps(res_id, indent=2))

    print("\n=== Testing Prediction Tool with Direct Raw Dict ===")
    raw_dict = {
        'employee_id': 99999, 'age': 45, 'salary': 85000, 'region': 'West',
        'employment_type': 'Full-time', 'has_dependents': 'Yes',
        'tenure_years': 5.0, 'prior_year_enrolled': 1, 'application_date': '2023-05-10',
        'last_contact_date': '2023-05-01', 'last_contact_channel': 'Email',
        'plan_tier_requested': 'Silver'
    }
    res_dict = predict_employee_enrollment(employee_data=raw_dict)
    print(json.dumps(res_dict, indent=2))
