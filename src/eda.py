import pandas as pd
from data import load_data

pd.set_option("display.width", 200)
df = load_data()

print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

for col in ["SEX", "EDUCATION", "MARRIAGE"]:
    print(f"\nDefault rate by {col}")
    print(df.groupby(col)["default"].agg(["count", "mean"]).round(3))

print("\nDefault rate by latest repayment status (PAY_1)")
print(df.groupby("PAY_1")["default"].agg(["count", "mean"]).round(3))

print("\nDefault rate by credit-limit quintile")
q = pd.qcut(df["LIMIT_BAL"], 5)
print(df.groupby(q, observed=True)["default"].agg(["count", "mean"]).round(3))

print("\nCorrelation with default (top 10)")
print(df.corr()["default"].drop("default").abs().sort_values(ascending=False).head(10).round(3))

print("\nNumeric summary")
print(df[["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]].describe().round(0))