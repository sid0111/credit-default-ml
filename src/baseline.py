import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data import load_data, split_data

X_train, X_test, y_train, y_test = split_data(load_data())

# Baseline 1: simple rule - flag anyone whose latest payment is delayed
rule = (X_train["PAY_1"] >= 1).astype(int)
print("RULE (PAY_1 >= 1) on train")
print("  precision:", round(precision_score(y_train, rule), 3))
print("  recall:   ", round(recall_score(y_train, rule), 3))
print("  AUC:      ", round(roc_auc_score(y_train, X_train["PAY_1"]), 3))

# Baseline 2: logistic regression, 5-fold CV on the training set only
lr = Pipeline([("sc", StandardScaler()),
               ("clf", LogisticRegression(max_iter=1000))])
cv = cross_validate(lr, X_train, y_train, cv=5,
                    scoring=["roc_auc", "average_precision"])
print("\nLOGISTIC REGRESSION (5-fold CV)")
print("  ROC-AUC:", cv["test_roc_auc"].mean().round(3),
      "+/-", cv["test_roc_auc"].std().round(3))
print("  PR-AUC: ", cv["test_average_precision"].mean().round(3))
print("  (random guessing PR-AUC would be about", round(y_train.mean(), 3), ")")