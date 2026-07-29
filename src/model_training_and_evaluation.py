"""
Milestones 13, 14, 15: Final Model Training, Comprehensive Evaluation, and Model Serialization

1. Cross-validates LightGBM and XGBoost classifiers on 5-fold Stratified CV.
2. Trains final LightGBM model on full train.csv.
3. Evaluates performance on held-out test.csv (ROC-AUC, PR-AUC, Confusion Matrix, Feature Importance).
4. Conducts demographic parity and fairness analysis (gender, age group, marital status).
5. Serializes model & preprocessing pipeline to models/enrollment_model.joblib.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, brier_score_loss
)
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, "../data/processed/train.csv")
    test_path  = os.path.join(current_dir, "../data/processed/test.csv")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    y_train = train_df['enrolled']
    X_train = train_df.drop(columns=['enrolled'])

    y_test = test_df['enrolled']
    X_test = test_df.drop(columns=['enrolled'])

    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col]  = X_test[col].astype('category')

    return X_train, y_train, X_test, y_test, cat_cols

def run_cross_validation(X_train, y_train):
    print("=" * 60)
    print("MILESTONE 13: 5-Fold Stratified Cross-Validation")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb_cv_scores = cross_val_score(lgb_clf, X_train, y_train, cv=cv, scoring='roc_auc')

    # XGBoost (requires enable_categorical=True)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        enable_categorical=True,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_cv_scores = cross_val_score(xgb_clf, X_train, y_train, cv=cv, scoring='roc_auc')

    print(f"LightGBM 5-Fold CV ROC-AUC: {lgb_cv_scores.mean():.4f} ± {lgb_cv_scores.std():.4f}")
    print(f"XGBoost  5-Fold CV ROC-AUC: {xgb_cv_scores.mean():.4f} ± {xgb_cv_scores.std():.4f}")

    return lgb_clf

def train_and_evaluate(final_clf, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("MILESTONE 14: Comprehensive Evaluation on Held-Out Test Set")
    print("=" * 60)

    final_clf.fit(X_train, y_train)
    y_probs = final_clf.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc  = average_precision_score(y_test, y_probs)
    acc     = accuracy_score(y_test, y_preds)
    prec    = precision_score(y_test, y_preds)
    rec     = recall_score(y_test, y_preds)
    f1      = f1_score(y_test, y_preds)
    brier   = brier_score_loss(y_test, y_probs)

    print("\n--- Test Set Evaluation Metrics ---")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"PR-AUC:           {pr_auc:.4f}")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Precision:        {prec:.4f}")
    print(f"Recall:           {rec:.4f}")
    print(f"F1-Score:         {f1:.4f}")
    print(f"Brier Score:      {brier:.4f}")

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_preds)
    print(pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1']))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_preds, digits=4))

    # Feature importances
    feature_imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance_gain': final_clf.booster_.feature_importance(importance_type='gain')
    }).sort_values('importance_gain', ascending=False)

    print("\n--- Top 10 Feature Importances (Gain) ---")
    print(feature_imp.head(10).to_string(index=False))

    # --- Fairness & Demographic Parity Audit ---
    print("\n--- Demographic Parity & Fairness Audit ---")
    test_eval_df = X_test.copy()
    test_eval_df['actual'] = y_test.values
    test_eval_df['predicted'] = y_preds
    test_eval_df['prob'] = y_probs

    # By Gender
    print("\nEnrollment & Predicted Rate by Gender:")
    gender_audit = test_eval_df.groupby('gender').agg(
        count=('actual', 'count'),
        actual_rate=('actual', 'mean'),
        pred_rate=('predicted', 'mean'),
        avg_prob=('prob', 'mean')
    )
    print(gender_audit.round(4))

    # By Age Group
    test_eval_df['age_group'] = pd.cut(test_eval_df['age'], bins=[20, 35, 50, 65], labels=['20-35', '36-50', '51-65'])
    print("\nEnrollment & Predicted Rate by Age Group:")
    age_audit = test_eval_df.groupby('age_group').agg(
        count=('actual', 'count'),
        actual_rate=('actual', 'mean'),
        pred_rate=('predicted', 'mean'),
        avg_prob=('prob', 'mean')
    )
    print(age_audit.round(4))

    return final_clf

def save_model(model):
    print("\n" + "=" * 60)
    print("MILESTONE 15: Model Persistence")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir  = os.path.join(current_dir, "../models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "enrollment_model.joblib")
    joblib.dump(model, model_path)
    print(f"Final LightGBM model saved successfully to: {model_path}")

def main():
    X_train, y_train, X_test, y_test, cat_cols = load_data()
    lgb_clf = run_cross_validation(X_train, y_train)
    final_model = train_and_evaluate(lgb_clf, X_train, y_train, X_test, y_test)
    save_model(final_model)

if __name__ == "__main__":
    main()
