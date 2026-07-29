import pandas as pd

df = pd.read_csv("employees_raw.csv")

# Let's test different parsing options
p1 = pd.to_datetime(df['application_date'], dayfirst=True, errors='coerce')
print("Sample parsed with dayfirst=True:")
for idx in [2, 4, 5, 10, 11, 20]:
    print(f"Original: {df.loc[idx, 'application_date']} | Parsed: {p1.loc[idx]}")

print("\nAre there any nulls in p1 that were not null in raw?")
print(p1.isnull().sum() - df['application_date'].isnull().sum())
