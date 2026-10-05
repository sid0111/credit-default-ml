from pathlib import Path
from ucimlrepo import fetch_ucirepo

ds = fetch_ucirepo(id=350)
df = ds.data.features.copy()
df["default"] = ds.data.targets.iloc[:, 0]

out = Path("data/raw/credit_default.csv")
df.to_csv(out, index=False)

print(df.shape)
print(df.head())
print("\nDefault rate:", df["default"].mean().round(3))
print("\nColumns:", df.columns.tolist())