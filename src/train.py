import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from data import load_data, split_data
from features import add_features, drop_sensitive

# ---- Business assumptions (CHANGE THESE) ---------------------------------
COST_FN = 5.0   # cost of missing a customer who then defaults
COST_FP = 1.0   # cost of wrongly flagging a good customer
# --------------------------------------------------------------------------

SEED = 42
X_train, X_test, y_train, y_test = split_data(load_data(), seed=SEED)


def build(**params):
    return Pipeline([
        ("feat", FunctionTransformer(add_features)),
        ("drop", FunctionTransformer(drop_sensitive)),
        ("clf", HistGradientBoostingClassifier(
            early_stopping=True, random_state=SEED, **params)),
    ])


def expected_cost(y_true, flagged):
    tn, fp, fn, tp = confusion_matrix(y_true, flagged).ravel()
    return (COST_FN * fn + COST_FP * fp) / len(y_true)


# 1. Light hyperparameter search (training data only)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(
    build(),
    {"clf__learning_rate": loguniform(0.02, 0.2),
     "clf__max_leaf_nodes": randint(6, 40),
     "clf__min_samples_leaf": randint(20, 200),
     "clf__l2_regularization": loguniform(0.1, 20),
     "clf__max_iter": randint(150, 600)},
    n_iter=25, scoring="roc_auc", cv=cv, n_jobs=-1,
    random_state=SEED, refit=True, verbose=1)
search.fit(X_train, y_train)
print("\nBest CV ROC-AUC:", round(search.best_score_, 4))
print("Best params:", search.best_params_)
model = search.best_estimator_

# 2. Choose threshold from out-of-fold predictions (NOT the test set)
oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
thresholds = np.arange(0.05, 0.95, 0.01)
costs = [expected_cost(y_train, oof >= t) for t in thresholds]
best_t = float(thresholds[int(np.argmin(costs))])
print(f"\nChosen threshold: {best_t:.2f}  (cost/customer {min(costs):.3f})")

print("\nThreshold table (out-of-fold, train):")
for t in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, best_t]:
    f = oof >= t
    print(f"  t={t:.2f}  flagged={f.mean():.1%}  "
          f"precision={precision_score(y_train, f):.3f}  "
          f"recall={recall_score(y_train, f):.3f}  "
          f"cost={expected_cost(y_train, f):.3f}")

# 3. Calibration check: do predicted probabilities match reality?
cal = pd.DataFrame({"p": oof, "y": y_train.values})
cal["bin"] = pd.qcut(cal["p"], 10, duplicates="drop")
print("\nCalibration by decile (predicted vs actual default rate):")
print(cal.groupby("bin", observed=True).agg(predicted=("p", "mean"),
                                            actual=("y", "mean")).round(3))

# 4. FINAL evaluation on the untouched test set (run once)
proba = model.predict_proba(X_test)[:, 1]
flag = proba >= best_t
tn, fp, fn, tp = confusion_matrix(y_test, flag).ravel()
test = {
    "roc_auc": roc_auc_score(y_test, proba),
    "pr_auc": average_precision_score(y_test, proba),
    "brier": brier_score_loss(y_test, proba),
    "precision": precision_score(y_test, flag),
    "recall": recall_score(y_test, flag),
    "cost_per_customer": expected_cost(y_test, flag),
    "cost_flag_nobody": expected_cost(y_test, np.zeros(len(y_test), int)),
    "cost_flag_everybody": expected_cost(y_test, np.ones(len(y_test), int)),
    "cost_rule_PAY_1": expected_cost(y_test, (X_test["PAY_1"] >= 1).astype(int)),
}
print("\n=== TEST SET ===")
for k, v in test.items():
    print(f"  {k}: {v:.4f}")
print(f"  confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

# 5. Retrain on ALL data with the chosen params, then save
final = build(**{k.replace("clf__", ""): v for k, v in search.best_params_.items()})
final.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

joblib.dump(final, "models/model_v1.joblib")
meta = {
    "version": "v1",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "sklearn_version": sklearn.__version__,
    "threshold": best_t,
    "cost_fn": COST_FN, "cost_fp": COST_FP,
    "cv_roc_auc": float(search.best_score_),
    "test_metrics": {k: float(v) for k, v in test.items()},
    "best_params": {k: (float(v) if isinstance(v, float) else int(v))
                    for k, v in search.best_params_.items()},
    "input_columns": X_train.columns.tolist(),
}
json.dump(meta, open("models/model_v1.json", "w"), indent=2)
print("\nSaved models/model_v1.joblib and models/model_v1.json")