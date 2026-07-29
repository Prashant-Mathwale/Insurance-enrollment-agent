import pandas as pd

df = pd.read_csv("employees_raw.csv")

slash_dates = df['application_date'].dropna().astype(str)
slash_dates = slash_dates[slash_dates.str.contains('/')]

print("A few slash-separated dates:")
print(slash_dates.head(20))

# Check if there are values where the first part is > 12 (must be DD/MM/YYYY)
first_part = slash_dates.apply(lambda x: int(x.split('/')[0]))
second_part = slash_dates.apply(lambda x: int(x.split('/')[1]))

print(f"Max first part: {first_part.max()}")
print(f"Max second part: {second_part.max()}")

# If max first part is > 12, then day is first.
# If max second part is > 12, then month is first.
# Let's see if both can be > 12 (indicating mixed or MM/DD/YYYY and DD/MM/YYYY).
print(f"Rows where first part > 12: {sum(first_part > 12)}")
print(f"Rows where second part > 12: {sum(second_part > 12)}")
