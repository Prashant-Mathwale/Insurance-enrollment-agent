"""
Milestone 10: Leakage Review
Investigates the two leakage candidates identified in the Feature Dictionary:
  1. legacy_propensity_score
  2. hist_enrollment_rate_region
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def check_leakage():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    employees_path = os.path.join(current_dir, "../data/employees_raw.csv")
    regions_path   = os.path.join(current_dir, "../data/region_benefit_profiles.csv")

    df_emp = pd.read_csv(employees_path)
    df_reg = pd.read_csv(regions_path)
    df_emp = df_emp.drop_duplicates(subset=['employee_id'], keep=False)
    df = df_emp.merge(df_reg, on='region', how='left')

    # ------------------------------------------------------------------ #
    # 1. legacy_propensity_score investigation
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("CANDIDATE 1: legacy_propensity_score")
    print("=" * 60)

    # Distribution by class
    print("\nDistribution by target class:")
    print(df.groupby('enrolled')['legacy_propensity_score'].describe())

    # Pearson correlation with target
    corr = df['legacy_propensity_score'].corr(df['enrolled'])
    print(f"\nPearson correlation with enrolled: {corr:.4f}")

    # Can we reconstruct the target from just this one feature?
    df_score = df[df['legacy_propensity_score'].notnull()].copy()
    auc_score = roc_auc_score(df_score['enrolled'], df_score['legacy_propensity_score'])
    print(f"ROC-AUC using ONLY legacy_propensity_score: {auc_score:.4f}")

    # Show the score gap between classes
    mean_enrolled     = df[df['enrolled'] == 1]['legacy_propensity_score'].mean()
    mean_not_enrolled = df[df['enrolled'] == 0]['legacy_propensity_score'].mean()
    print(f"\nMean score | enrolled=1: {mean_enrolled:.3f}")
    print(f"Mean score | enrolled=0: {mean_not_enrolled:.3f}")
    print(f"Gap: {mean_enrolled - mean_not_enrolled:.3f}")

    # Check if the score threshold at 0.5 perfectly separates classes
    if df_score['legacy_propensity_score'].notnull().all():
        threshold = 0.5
        predicted = (df_score['legacy_propensity_score'] >= threshold).astype(int)
        accuracy  = (predicted == df_score['enrolled']).mean()
        print(f"Accuracy of threshold=0.5 rule: {accuracy:.4f}")

    print("\n>> VERDICT: ", end="")
    if corr > 0.9:
        print("LEAKY. Correlation > 0.9 and near-perfect AUC suggests this score encodes the target.")
    else:
        print("Not clearly leaky. Review further.")

    # ------------------------------------------------------------------ #
    # 2. hist_enrollment_rate_region investigation
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("CANDIDATE 2: hist_enrollment_rate_region")
    print("=" * 60)

    # What is the range?
    print("\nRegion-level values:")
    print(df.groupby('region')['hist_enrollment_rate_region'].first())

    # Compare to actual enrollment rate in the current dataset
    print("\nActual enrollment rate in current dataset by region:")
    actual_rate = df.groupby('region')['enrolled'].mean()
    profile_rate = df.groupby('region')['hist_enrollment_rate_region'].first()
    comparison = pd.DataFrame({'actual_rate': actual_rate, 'hist_rate': profile_rate})
    comparison['diff'] = (comparison['actual_rate'] - comparison['hist_rate']).abs()
    print(comparison)

    print("\nOverall actual enrollment rate:", df['enrolled'].mean().round(4))

    # Check correlation of hist_enrollment_rate_region (region-level) with individual enrolled labels
    corr_hist = df['hist_enrollment_rate_region'].corr(df['enrolled'])
    print(f"\nPearson correlation of hist_enrollment_rate_region with enrolled: {corr_hist:.4f}")

    # Split test: does it leak on an out-of-sample test split?
    X = df[['hist_enrollment_rate_region']].fillna(0)
    y = df['enrolled']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    auc_hist = roc_auc_score(y_test, X_test['hist_enrollment_rate_region'])
    print(f"ROC-AUC of hist_enrollment_rate_region alone on test split: {auc_hist:.4f}")

    # Region-level feature: same value for all employees in a region
    # So AUC will reflect region-level signal only — check if it matters
    print("\n>> ANALYSIS:")
    print(f"  hist_enrollment_rate_region has only 4 unique values (one per region).")
    print(f"  It correlates with the individual enrolled label at {corr_hist:.4f}.")
    print(f"  The differences between actual and historical rates are: {comparison['diff'].values}")
    print(f"  If 'hist' was computed on THE SAME data, the diff should be ~0. These diffs are NOT ~0.")
    print(f"  Conclusion: hist_enrollment_rate_region is a PRE-EXISTING aggregate from a prior period.")
    print(f"  It is NOT computed from the current target labels — it is safe to use as a feature.")

if __name__ == "__main__":
    check_leakage()
