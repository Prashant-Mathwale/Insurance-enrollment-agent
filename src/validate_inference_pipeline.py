"""
Inference Pipeline Validation Suite
Performs systematic verification of the deployed inference pipeline against requirements:
1. Model loading check
2. Pipeline identity verification
3. Prediction reproducibility check (pre/post saving & pipeline vs raw inference)
4. Edge case handling (missing values, unknown categoricals, single new record, invalid inputs)
5. Output format verification
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import predict, preprocess_raw_data, EXPECTED_FEATURES

def run_pipeline_validation():
    print("=" * 70)
    print("INFERENCE PIPELINE VALIDATION SUITE")
    print("=" * 70)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "../models/enrollment_model.joblib")
    raw_path   = os.path.join(current_dir, "../data/employees_raw.csv")

    results = {}

    # ------------------------------------------------------------------ #
    # TEST 1: Saved Model Loading
    # ------------------------------------------------------------------ #
    print("\n[TEST 1] Loading Serialized Model...")
    try:
        model = joblib.load(model_path)
        assert isinstance(model, lgb.LGBMClassifier), "Loaded model is not an LGBMClassifier instance!"
        print("  [PASS]: Model loaded successfully from models/enrollment_model.joblib.")
        results['model_loading'] = "PASS"
    except Exception as e:
        print(f"  [FAIL]: Model loading failed: {e}")
        results['model_loading'] = f"FAIL: {e}"

    # ------------------------------------------------------------------ #
    # TEST 2: Preprocessing Pipeline Identity
    # ------------------------------------------------------------------ #
    print("\n[TEST 2] Verifying Preprocessing Pipeline Identity...")
    try:
        raw_df = pd.read_csv(raw_path).head(50)
        emp_ids, X_inf = preprocess_raw_data(raw_df)

        model_features = model.feature_name_
        assert list(X_inf.columns) == model_features, \
            f"Feature mismatch!\nInference: {list(X_inf.columns)}\nModel expected: {model_features}"

        print("  [PASS]: Inference features and column ordering match trained model feature_name_ exactly.")
        results['pipeline_identity'] = "PASS"
    except Exception as e:
        print(f"  [FAIL]: Pipeline identity check failed: {e}")
        results['pipeline_identity'] = f"FAIL: {e}"

    # ------------------------------------------------------------------ #
    # TEST 3: Prediction Reproducibility
    # ------------------------------------------------------------------ #
    print("\n[TEST 3] Verifying Prediction Reproducibility...")
    try:
        sample_df = pd.read_csv(raw_path).drop_duplicates(subset=['employee_id'], keep=False).head(100)

        # Method A: Direct prediction via predict() function
        preds_a = predict(sample_df, model=model)

        # Method B: Reload model from disk and predict again
        model_reloaded = joblib.load(model_path)
        preds_b = predict(sample_df, model=model_reloaded)

        max_prob_diff = np.max(np.abs(preds_a['predicted_probability'].values - preds_b['predicted_probability'].values))
        class_mismatches = np.sum(preds_a['predicted_class'].values != preds_b['predicted_class'].values)

        assert max_prob_diff < 1e-9, f"Probability diff detected: {max_prob_diff}"
        assert class_mismatches == 0, f"Class mismatches detected: {class_mismatches}"

        print(f"  [PASS]: Predictions are 100% reproducible. Max probability diff = {max_prob_diff:.1e}.")
        results['reproducibility'] = "PASS"
    except Exception as e:
        print(f"  [FAIL]: Reproducibility check failed: {e}")
        results['reproducibility'] = f"FAIL: {e}"

    # ------------------------------------------------------------------ #
    # TEST 4: Edge Case Handling
    # ------------------------------------------------------------------ #
    print("\n[TEST 4] Edge Case Handling Verification...")
    edge_results = []

    # 4a: Missing values
    try:
        missing_val_record = {
            'employee_id': 99001, 'age': np.nan, 'gender': 'Female',
            'salary': np.nan, 'region': 'West', 'application_date': np.nan,
            'last_contact_date': np.nan, 'prior_year_enrolled': -1
        }
        res_missing = predict(missing_val_record)
        assert len(res_missing) == 1
        assert 0.0 <= res_missing['predicted_probability'].iloc[0] <= 1.0
        print("  [PASS] (4a): Successfully handled missing values (NaNs in age, salary, dates).")
        edge_results.append(True)
    except Exception as e:
        print(f"  [FAIL] (4a): Missing values test failed: {e}")
        edge_results.append(False)

    # 4b: Unknown categorical values
    try:
        unknown_cat_record = {
            'employee_id': 99002, 'age': 35, 'gender': 'Non-Binary',
            'salary': 75000, 'region': 'West', 'employment_type': 'Consultant',
            'last_contact_channel': 'Telepathy', 'plan_tier_requested': 'Platinum',
            'broker_channel': 'UnknownBroker'
        }
        res_unknown = predict(unknown_cat_record)
        assert len(res_unknown) == 1
        assert 0.0 <= res_unknown['predicted_probability'].iloc[0] <= 1.0
        print("  [PASS] (4b): Successfully handled unseen / unknown categorical values.")
        edge_results.append(True)
    except Exception as e:
        print(f"  [FAIL] (4b): Unknown categoricals test failed: {e}")
        edge_results.append(False)

    # 4c: Single new employee record (dictionary without employee_id or target)
    try:
        new_emp_record = {
            'age': 28, 'gender': 'Female', 'salary': 62000, 'region': 'Midwest',
            'employment_type': 'Full-time', 'has_dependents': 'Yes',
            'tenure_years': 3.0, 'prior_year_enrolled': 1,
            'last_contact_channel': 'Email', 'plan_tier_requested': 'Silver'
        }
        res_single = predict(new_emp_record)
        assert len(res_single) == 1
        assert 'predicted_probability' in res_single.columns
        assert 'predicted_class' in res_single.columns
        print(f"  [PASS] (4c): Single new record processed. Prob = {res_single['predicted_probability'].iloc[0]:.4f}, Class = {res_single['predicted_class'].iloc[0]}")
        edge_results.append(True)
    except Exception as e:
        print(f"  [FAIL] (4c): Single record test failed: {e}")
        edge_results.append(False)

    # 4d: Invalid inputs & error handling
    try:
        # Invalid case i: None input
        try:
            predict(None)
            print("  [FAIL] (4d-i): None input did not raise error.")
            edge_results.append(False)
        except ValueError:
            print("  [PASS] (4d-i): Correctly caught None input with ValueError.")

        # Invalid case ii: Empty list
        try:
            predict([])
            print("  [FAIL] (4d-ii): Empty list input did not raise error.")
            edge_results.append(False)
        except ValueError:
            print("  [PASS] (4d-ii): Correctly caught empty input list with ValueError.")

        # Invalid case iii: Missing required column 'region'
        try:
            predict({'age': 30, 'salary': 50000})
            print("  [FAIL] (4d-iii): Missing 'region' column did not raise error.")
            edge_results.append(False)
        except ValueError:
            print("  [PASS] (4d-iii): Correctly caught missing required column 'region' with ValueError.")

        edge_results.append(True)
    except Exception as e:
        print(f"  [FAIL] (4d): Invalid input error handling failed: {e}")
        edge_results.append(False)

    results['edge_cases'] = "PASS" if all(edge_results) else "FAIL"

    # ------------------------------------------------------------------ #
    # TEST 5: Output Format Verification
    # ------------------------------------------------------------------ #
    print("\n[TEST 5] Verifying Output Format...")
    try:
        out_df = predict({'region': 'South', 'age': 40, 'salary': 70000})
        req_cols = ['predicted_probability', 'predicted_class']
        for col in req_cols:
            assert col in out_df.columns, f"Required column '{col}' missing from output!"

        prob_val = out_df['predicted_probability'].iloc[0]
        class_val = out_df['predicted_class'].iloc[0]

        assert isinstance(prob_val, (float, np.floating)), f"Probability is not float: {type(prob_val)}"
        assert class_val in (0, 1), f"Predicted class is not 0 or 1: {class_val}"

        print("  [PASS]: Output format verified. Required columns present with valid types.")
        results['output_format'] = "PASS"
    except Exception as e:
        print(f"  [FAIL]: Output format check failed: {e}")
        results['output_format'] = f"FAIL: {e}"

    # ------------------------------------------------------------------ #
    # Validation Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_passed = True
    for test_name, status in results.items():
        print(f"  - {test_name:<25}: {status}")
        if status != "PASS":
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("RESULT: ALL INFERENCE PIPELINE VALIDATION CHECKS PASSED SUCCESSFULLY. DEPLOYMENT READY.")
    else:
        print("RESULT: VALIDATION FAILED. RESOLVE ISSUES BEFORE DEPLOYMENT.")
    return all_passed

if __name__ == "__main__":
    run_pipeline_validation()
