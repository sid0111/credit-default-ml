import pandas as pd
from sklearn.model_selection import train_test_split

# Mapping from the UCI documentation.
# PAY_1 = September (most recent month) ... PAY_6 = April (oldest).
RENAME = {"X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION",
          "X4": "MARRIAGE", "X5": "AGE"}
for i in range(6):
    RENAME[f"X{6 + i}"] = f"PAY_{i + 1}"        # repayment status
    RENAME[f"X{12 + i}"] = f"BILL_AMT{i + 1}"   # bill statement
    RENAME[f"X{18 + i}"] = f"PAY_AMT{i + 1}"    # amount paid

TARGET = "default"


def load_data(path="data/raw/credit_default.csv"):
    df = pd.read_csv(path).rename(columns=RENAME)
    # Undocumented category codes -> merge into "others"
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def split_data(df, seed=42):
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)