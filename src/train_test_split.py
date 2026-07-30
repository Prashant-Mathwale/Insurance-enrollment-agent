"""
Milestone 11: Train/Test Split
Applies feature engineering pipeline, creates stratified 80/20 train and test splits,
saves processed datasets to data/processed/, and verifies split integrity.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from feature_engineering import engineer_features

def create_splits():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, "../data/processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Load and engineer full dataset
    df, candidate_features = engineer_features()

    features_to_drop = [
        'employee_id', 'application_date', 'last_contact_date',
        'app_date', 'contact_date', 'prior_year_enrolled', 'has_dependents',
        'last_contact_channel', 'plan_tier_requested', 'broker_channel',
        'legacy_propensity_score', 'n_employees_region', 'avg_salary_region',
        'outreach_notes', 'age', 'gender', 'marital_status'
    ]

    clean_df = df.drop(columns=[c for c in features_to_drop if c in df.columns])

    # Save full processed dataset
    clean_path = os.path.join(processed_dir, "employees_processed.csv")
    clean_df.to_csv(clean_path, index=False)
    print(f"Saved full processed dataset to {clean_path} (shape: {clean_df.shape})")

    # Perform stratified 80/20 train/test split
    train_df, test_df = train_test_split(
        clean_df,
        test_size=0.20,
        random_state=42,
        stratify=clean_df['enrolled']
    )

    train_path = os.path.join(processed_dir, "train.csv")
    test_path  = os.path.join(processed_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n=== Split Summary ===")
    print(f"Train set: {train_df.shape[0]} rows (80%)")
    print(f"Test set:  {test_df.shape[0]} rows (20%)")
    print(f"Total:     {len(train_df) + len(test_df)} rows")

    # Verify class balance
    train_dist = train_df['enrolled'].value_counts(normalize=True)
    test_dist  = test_df['enrolled'].value_counts(normalize=True)

    print("\n=== Target Distribution (Class Balance) ===")
    print(f"Full dataset: {clean_df['enrolled'].mean():.4f} enrolled rate")
    print(f"Train split:  {train_df['enrolled'].mean():.4f} enrolled rate")
    print(f"Test split:   {test_df['enrolled'].mean():.4f} enrolled rate")

    assert abs(train_df['enrolled'].mean() - test_df['enrolled'].mean()) < 0.001, "Stratification mismatch!"
    print("\nAssertions passed: Stratification is perfect.")

if __name__ == "__main__":
    create_splits()
