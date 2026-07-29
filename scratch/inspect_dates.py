import pandas as pd

df = pd.read_csv("employees_raw.csv")

print("--- Print a few application dates ---")
print(df['application_date'].dropna().head(30))

print("\n--- Try parsing with format='mixed' ---")
parsed = pd.to_datetime(df['application_date'], errors='coerce', format='mixed')
print(f"Total rows: {len(df)}")
print(f"Null application dates initially: {df['application_date'].isnull().sum()}")
print(f"Null application dates after parsing: {parsed.isnull().sum()}")
print(f"Failed parses: {parsed.isnull().sum() - df['application_date'].isnull().sum()}")

# Print rows where parsing failed (if any)
failed_rows = df[df['application_date'].notnull() & parsed.isnull()]
if len(failed_rows) > 0:
    print("Failed to parse these:")
    print(failed_rows['application_date'].head(10))
else:
    print("All non-null application dates parsed successfully with format='mixed'.")
