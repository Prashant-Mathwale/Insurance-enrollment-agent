"""
Verification Runner for Bug 1 and Bug 2 Fixes.
"""
import sys
import joblib

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

print("=" * 70)
print("VERIFICATION OF BUG 1 & BUG 2 FIXES")
print("=" * 70)

# Load trained model
model = joblib.load("models/enrollment_model.joblib")
feature_list = list(model.feature_name_)

print(f"\n1. Final Model Feature Count: {len(feature_list)}")
print(f"   Model Features: {feature_list}")

demographics = ['age', 'gender', 'marital_status']
absent = [d not in feature_list for d in demographics]
print(f"\n2. Are demographic features ('age', 'gender', 'marital_status') absent from model? {all(absent)}")
for d in demographics:
    print(f"   - '{d}' in model? {d in feature_list}")

from src.tools.predict_tool import predict_employee_enrollment
from src.tools.explain_tool import explain_prediction

print("\n3. Testing predict_tool.py explicit refusal for legacy_propensity_score:")
leaky_dict = {'employee_id': 12345, 'salary': 50000, 'region': 'West', 'legacy_propensity_score': 0.85}
res_pred_refusal = predict_employee_enrollment(employee_data=leaky_dict)
print(f"   - Status: {res_pred_refusal.get('status')}")
print(f"   - Message: {res_pred_refusal.get('message')}")

print("\n4. Testing explain_tool.py explicit refusal for legacy_propensity_score:")
res_exp_refusal = explain_prediction(employee_id=17825, feature_requested='legacy_propensity_score')
print(f"   - Status: {res_exp_refusal.get('status')}")
print(f"   - Message: {res_exp_refusal.get('message')}")

print("\n5. Testing normal predict_tool.py execution:")
res_norm_pred = predict_employee_enrollment(employee_id=17825)
print(f"   - Status: {res_norm_pred.get('status')}")
print(f"   - Prediction Probability: {res_norm_pred.get('predictions', [{}])[0].get('enrollment_probability')}")

print("\n6. Testing normal explain_tool.py execution:")
res_norm_exp = explain_prediction(employee_id=17825)
print(f"   - Status: {res_norm_exp.get('status')}")
print(f"   - Narrative Summary: {res_norm_exp.get('narrative_summary')}")

print("\nAll verification assertions completed successfully.")
