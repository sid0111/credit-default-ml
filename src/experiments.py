import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from data import load_data, split_data
from features import add_features, drop_sensitive

X_train, X_test, y_train, y_test = split_data(load_data())


def make(clf, feats=True, drop=False, scale=False):
    steps = []
    if feats:
        steps.append(("feat", FunctionTransformer(add_features)))
    if drop:
        steps.append(("drop", FunctionTransformer(drop_sensitive)))
    if scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def hgb():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
        l2_regularization=1.0, early_stopping=True, random_state=42)


experiments = {
    "LogReg, raw features":              make(LogisticRegression(max_iter=2000), feats=False, scale=True),
    "LogReg + engineered":               make(LogisticRegression(max_iter=2000), scale=True),
    "GradBoost, raw features":           make(hgb(), feats=False),
    "GradBoost + engineered":            make(hgb()),
    "GradBoost + engineered, no SEX/MARRIAGE/AGE": make(hgb(), drop=True),
}

rows = []
for name, model in experiments.items():
    cv = cross_validate(model, X_train, y_train, cv=5,
                        scoring=["roc_auc", "average_precision"], n_jobs=-1)
    rows.append({"model": name,
                 "ROC-AUC": cv["test_roc_auc"].mean(),
                 "AUC std": cv["test_roc_auc"].std(),
                 "PR-AUC": cv["test_average_precision"].mean()})
    print("done:", name)

print(pd.DataFrame(rows).round(3).to_string(index=False))