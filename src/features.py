import pandas as pd

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
SENSITIVE = ["SEX", "MARRIAGE", "AGE"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pay, bill, paid, limit = df[PAY_COLS], df[BILL_COLS], df[PAYAMT_COLS], df["LIMIT_BAL"]

    # Delinquency history
    df["N_DELAYED"] = (pay >= 1).sum(axis=1)       # months with a delay
    df["MAX_DELAY"] = pay.max(axis=1)              # worst delay
    df["AVG_PAY_STATUS"] = pay.mean(axis=1)
    df["PAY_TREND"] = df["PAY_1"] - df["PAY_6"]    # getting worse or better?

    # Credit utilisation
    df["UTIL_1"] = bill["BILL_AMT1"] / limit
    df["UTIL_AVG"] = bill.mean(axis=1) / limit
    df["BILL_TREND"] = (bill["BILL_AMT1"] - bill["BILL_AMT6"]) / limit

    # Payment behaviour. Assumption: PAY_AMTk pays the previous month's bill,
    # i.e. PAY_AMT1..5 go with BILL_AMT2..6.
    prev_bills = bill[BILL_COLS[1:]].sum(axis=1).clip(lower=1)
    df["PAY_RATIO"] = (paid[PAYAMT_COLS[:5]].sum(axis=1) / prev_bills).clip(0, 5)
    df["LAST_PAY_RATIO"] = (df["PAY_AMT1"] / bill["BILL_AMT2"].clip(lower=1)).clip(0, 5)
    return df


def drop_sensitive(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=SENSITIVE, errors="ignore")