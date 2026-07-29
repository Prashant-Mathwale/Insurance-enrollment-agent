"""
Investigates whether LightGBM and XGBoost natively handle NaN values
and compares NaN vs -999 sentinel for the days_contact_to_app feature.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def build_feature_matrix(df, sentinel_value=None):
    """Build a minimal feature set with days_contact_to_app filled by sentinel or NaN."""
    df = df.copy()

    # Parse dates
    df['app_date'] = pd.to_datetime(df['application_date'], format='mixed', dayfirst=True, errors='coerce')
    df['contact_date'] = pd.to_datetime(df['last_contact_date'], format='mixed', errors='coerce')

    # Engineer days_contact_to_app
    df['days_contact_to_app'] = (df['app_date'] - df['contact_date']).dt.days

    # Fill missing dates
    if sentinel_value is not None:
        df['days_contact_to_app'] = df['days_contact_to_app'].fillna(sentinel_value)

    # Binary flag for missing application_date
    df['has_application_date'] = df['app_date'].notnull().astype(int)

    # Select a simple feature set (numeric only for this test)
    features = ['age', 'salary', 'tenure_years', 'days_contact_to_app', 'has_application_date']
    X = df[features]
    y = df['enrolled']
    return X, y


def evaluate_model(clf, X, y, label):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    print(f"  [{label}] ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    employees_path = os.path.join(current_dir, "../data/employees_raw.csv")
    df = pd.read_csv(employees_path)
    df = df.drop_duplicates(subset=['employee_id'], keep=False)

    # --- LightGBM ---
    print("=== LightGBM: NaN native support check ===")
    print("LightGBM documentation: Handles NaN natively by learning the optimal split direction for missing values.")
    lgb_clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

    print("\nNaN (native handling):")
    X_nan, y = build_feature_matrix(df, sentinel_value=None)
    lgb_nan_score = evaluate_model(lgb_clf, X_nan, y, "LightGBM + NaN")

    print("\n-999 sentinel:")
    X_sentinel, y = build_feature_matrix(df, sentinel_value=-999)
    lgb_sentinel_score = evaluate_model(lgb_clf, X_sentinel, y, "LightGBM + -999")

    # --- XGBoost ---
    print("\n=== XGBoost: NaN native support check ===")
    print("XGBoost documentation: Handles NaN natively since v0.9, learns default split direction for missing.")
    xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)

    print("\nNaN (native handling):")
    xgb_nan_score = evaluate_model(xgb_clf, X_nan, y, "XGBoost + NaN")

    print("\n-999 sentinel:")
    xgb_sentinel_score = evaluate_model(xgb_clf, X_sentinel, y, "XGBoost + -999")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"{'Model':<30} {'NaN AUC':>10} {'-999 AUC':>10} {'Diff':>10}")
    print("-" * 65)
    print(f"{'LightGBM':<30} {lgb_nan_score:>10.4f} {lgb_sentinel_score:>10.4f} {lgb_nan_score - lgb_sentinel_score:>+10.4f}")
    print(f"{'XGBoost':<30} {xgb_nan_score:>10.4f} {xgb_sentinel_score:>10.4f} {xgb_nan_score - xgb_sentinel_score:>+10.4f}")

    print("\n=== Interpretation ===")
    print("""
Positive diff means NaN > -999 (NaN is better).
Negative diff means -999 > NaN.

Note: A -999 sentinel leaks an artificial pattern:
  - The value -999 is 1000+ units away from the true data range (~-10 to ~300 days).
  - Tree models treat -999 as a real value and may create spurious splits distinguishing
    missing vs non-missing based on the extreme distance rather than learning the true
    missingness direction.
  - With NaN, both LightGBM and XGBoost learn the OPTIMAL split direction during training,
    which is unbiased and more interpretable.
  - The `has_application_date` flag already encodes missingness explicitly,
    so NaN on the numeric feature is redundant signal, not lost signal.
""")

if __name__ == "__main__":
    main()
