import pandas as pd
import lightgbm as lgb
from baseline_model import prepare_data

X_train, y_train, X_test, y_test = prepare_data()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

X_train_lgb = X_train.copy()
X_test_lgb  = X_test.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_test_lgb[col]  = X_test_lgb[col].astype('category')

lgb_clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgb_clf.fit(X_train_lgb, y_train)

importance = pd.DataFrame({
    'feature': X_train_lgb.columns,
    'importance_split': lgb_clf.booster_.feature_importance(importance_type='split'),
    'importance_gain': lgb_clf.booster_.feature_importance(importance_type='gain')
}).sort_values('importance_gain', ascending=False)

print("=== Feature Importances (Default LightGBM) ===")
print(importance.to_string(index=False))
