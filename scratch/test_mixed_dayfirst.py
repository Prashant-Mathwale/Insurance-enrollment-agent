import pandas as pd

df = pd.read_csv("employees_raw.csv")

# Let's test parsing with format='mixed' and dayfirst=True
p2 = pd.to_datetime(df['application_date'], format='mixed', dayfirst=True, errors='coerce')
print("Sample parsed with format='mixed' and dayfirst=True:")
for idx in [0, 2, 4, 5, 7, 10, 11, 20]:
    print(f"Original: {df.loc[idx, 'application_date']} | Parsed: {p2.loc[idx]}")

print("\nAre there any nulls in p2 that were not null in raw?")
print(p2.isnull().sum() - df['application_date'].isnull().sum())
