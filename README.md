# Credit Card Default Risk: End-to-End ML Project

Predicts whether a cardholder will default on next month's payment. Includes
training, a cost-based decision threshold, a containerized REST API, CI, and
drift monitoring.

**Data:** UCI "Default of Credit Card Clients" (30,000 customers, Taiwan, 2005).
**Live demo:** <[your Render URL](https://credit-default-api-skuu.onrender.com/docs)> (free tier: first request after idle takes about a minute; endpoints need an API key).

## Results (held-out test set, 6,000 customers)

| Metric | Value |
|---|---|
| ROC-AUC | 0.780 |
| PR-AUC | 0.560 (random guessing: 0.22) |
| Brier score | 0.135 (probabilities well calibrated) |
| Precision / recall at threshold 0.15 | 0.35 / 0.78 |

Expected cost per customer (a missed defaulter costs 5x a false alarm):

| Strategy | Cost |
|---|---|
| Flag nobody | 1.106 |
| Flag everybody | 0.779 |
| Rule: latest payment delayed | 0.667 |
| **This model** | **0.563** |

## Key decisions

- **Gradient boosting beat logistic regression** (CV AUC 0.787 vs 0.726 on raw features) because repayment codes are not linear.
- **Sex, marital status, and age are excluded.** Dropping them cost no accuracy, and the API rejects them.
- **The threshold is chosen by business cost, not 0.5,** using out-of-fold predictions so the test set stays untouched. It can be changed without retraining via the `THRESHOLD` environment variable.
- **All preprocessing lives inside the model pipeline,** so training and serving cannot diverge.

## Project structure

```
src/data.py         loading and cleaning
src/features.py     feature engineering
src/train.py        tuning, threshold selection, evaluation, saving
src/app.py          FastAPI service
src/monitor.py      drift (PSI) monitoring
tests/              API tests
models/             trained model and metadata
Dockerfile          container image
.github/workflows/  CI: tests, image build, container smoke test
```

## Run locally

```bash
pip install -r requirements-dev.txt
python src/get_data.py
python src/train.py
pytest
uvicorn src.app:app --reload     # docs at http://127.0.0.1:8000/docs
```

Docker: `docker build -t credit-default-api . && docker run -p 8000:8000 credit-default-api`

## API

`POST /predict` (header `X-API-Key` required when the `API_KEY` variable is set)
returns `default_probability`, `flagged_high_risk`, and `model_version`.
Other endpoints: `/predict/batch`, `/model-info`, `/health`.

## Monitoring

`python src/monitor.py` compares new data to training data using the Population
Stability Index. Retrain when drift persists, when live AUC or calibration
degrades once outcomes arrive, or on a fixed schedule.

## Limitations

- The data is from 2005 Taiwan; a real deployment needs retraining on current local data.
- Live input logging, scheduled monitoring, and outcome matching are not automated yet.
- Authentication is a single shared key, and the free hosting tier sleeps when idle.
- Fairness was checked only by removing sensitive columns, not by a full group-level audit.