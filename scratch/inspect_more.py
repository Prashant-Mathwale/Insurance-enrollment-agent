import pandas as pd
import numpy as np

df = pd.read_csv("employees_raw.csv")

print("--- Check application_date vs enrolled ---")
app_not_null = df['application_date'].notnull()
print(pd.crosstab(app_not_null, df['enrolled'], rownames=['has_application_date']))

print("\n--- Check last_contact_channel vs enrolled ---")
channel_not_null = df['last_contact_channel'].notnull()
print(pd.crosstab(channel_not_null, df['enrolled'], rownames=['has_last_contact_channel']))

print("\n--- Check broker_channel vs enrolled ---")
broker_not_null = df['broker_channel'].notnull()
print(pd.crosstab(broker_not_null, df['enrolled'], rownames=['has_broker_channel']))

print("\n--- legacy_propensity_score stats by enrolled ---")
print(df.groupby('enrolled')['legacy_propensity_score'].describe())

print("\n--- Let's build a quick random forest or gradient boosting model on simple features and check feature importance ---")
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Copy df and drop columns we want to inspect or handle
df_temp = df.copy()
df_temp = df_temp.dropna(subset=['enrolled'])

# Simple encoding for testing
for col in df_temp.select_dtypes(include=['object']).columns:
    df_temp[col] = df_temp[col].astype(str)
    df_temp[col] = LabelEncoder().fit_transform(df_temp[col])

# Fill na
df_temp = df_temp.fillna(-999)

X = df_temp.drop(columns=['employee_id', 'enrolled'])
y = df_temp['enrolled']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importances:")
print(importances)
