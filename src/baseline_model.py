"""
Milestone 12: Baseline Model
Trains simple baseline classifiers (Logistic Regression & Default LightGBM)
on data/processed/train.csv and evaluates them on data/processed/test.csv.
Logs key evaluation metrics: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1, LogLoss.
"""
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, log_loss
)
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def prepare_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, "../data/processed/train.csv")
    test_path  = os.path.join(current_dir, "../data/processed/test.csv")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    y_train = train_df['enrolled']
    X_train = train_df.drop(columns=['enrolled'])

    y_test = test_df['enrolled']
    X_test = test_df.drop(columns=['enrolled'])

    return X_train, y_train, X_test, y_test

def evaluate(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    return {
        'ROC-AUC': roc_auc_score(y_true, y_pred_prob),
        'PR-AUC': average_precision_score(y_true, y_pred_prob),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'LogLoss': log_loss(y_true, y_pred_prob)
    }

def run_baselines():
    X_train, y_train, X_test, y_test = prepare_data()

    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64', 'int8', 'int32']).columns.tolist()

    print(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical.")
    print(f"Categorical features: {cat_cols}")

    # --- Baseline 1: Logistic Regression ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # Impute numeric NaNs with median for Logistic Regression
    X_train_lr = X_train.copy()
    X_test_lr  = X_test.copy()
    for col in num_cols:
        med = X_train_lr[col].median()
        X_train_lr[col] = X_train_lr[col].fillna(med)
        X_test_lr[col]  = X_test_lr[col].fillna(med)

    lr_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])

    lr_pipeline.fit(X_train_lr, y_train)
    lr_probs = lr_pipeline.predict_proba(X_test_lr)[:, 1]
    lr_metrics = evaluate(y_test, lr_probs)

    # --- Baseline 2: Default LightGBM ---
    X_train_lgb = X_train.copy()
    X_test_lgb  = X_test.copy()
    for col in cat_cols:
        X_train_lgb[col] = X_train_lgb[col].astype('category')
        X_test_lgb[col]  = X_test_lgb[col].astype('category')

    lgb_clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgb_clf.fit(X_train_lgb, y_train)
    lgb_probs = lgb_clf.predict_proba(X_test_lgb)[:, 1]
    lgb_metrics = evaluate(y_test, lgb_probs)

    # --- Summary Table ---
    results = pd.DataFrame([lr_metrics, lgb_metrics], index=['Logistic Regression', 'Default LightGBM'])
    print("\n=== Baseline Model Performance (Test Set) ===")
    print(results.round(4))

    return results

if __name__ == "__main__":
    run_baselines()
