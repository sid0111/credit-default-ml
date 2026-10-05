import json
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import secrets
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
import features  # noqa: F401,E402  (the saved model references features.add_features)

MODEL_DIR = Path(os.getenv("MODEL_DIR", SRC_DIR.parent / "models"))
MODEL_NAME = os.getenv("MODEL_NAME", "model_v1")

model = joblib.load(MODEL_DIR / f"{MODEL_NAME}.joblib")
meta = json.loads((MODEL_DIR / f"{MODEL_NAME}.json").read_text())
# The threshold is a business knob: override via env var, no retraining needed.
THRESHOLD = float(os.getenv("THRESHOLD", meta["threshold"]))

API_KEY = os.getenv("API_KEY")  # if unset, auth is off (local dev and tests)

def require_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("credit-api")


class Applicant(BaseModel):
    # Unknown fields (e.g. SEX, AGE) are rejected: the model must not use them.
    model_config = ConfigDict(extra="forbid")

    LIMIT_BAL: float = Field(gt=0, le=10_000_000, description="Credit limit")
    EDUCATION: int = Field(ge=1, le=4, description="1 grad, 2 univ, 3 high school, 4 other")
    # Repayment status, PAY_1 = most recent month. -2/-1 = paid, 0 = revolving, 1+ = months late
    PAY_1: int = Field(ge=-2, le=9)
    PAY_2: int = Field(ge=-2, le=9)
    PAY_3: int = Field(ge=-2, le=9)
    PAY_4: int = Field(ge=-2, le=9)
    PAY_5: int = Field(ge=-2, le=9)
    PAY_6: int = Field(ge=-2, le=9)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class Batch(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=1000)


COLUMNS = list(Applicant.model_fields)  # same order the model was trained on

app = FastAPI(title="Credit Card Default Risk API", version=meta["version"])


def score(rows: list[Applicant]) -> list[float]:
    df = pd.DataFrame([r.model_dump() for r in rows], columns=COLUMNS)
    return model.predict_proba(df)[:, 1].tolist()


def result(p: float) -> dict:
    return {"default_probability": round(p, 4),
            "flagged_high_risk": p >= THRESHOLD,
            "threshold": THRESHOLD,
            "model_version": meta["version"]}


@app.get("/health")
def health():
    return {"status": "ok", "model_version": meta["version"]}


@app.get("/model-info", dependencies=[Depends(require_key)])
def model_info():
    return {"version": meta["version"], "trained_at": meta["trained_at"],
            "threshold": THRESHOLD, "test_metrics": meta["test_metrics"]}


@app.post("/predict", dependencies=[Depends(require_key)])
def predict(applicant: Applicant):
    p = score([applicant])[0]
    log.info("prediction p=%.4f flagged=%s model=%s", p, p >= THRESHOLD, meta["version"])
    return result(p)


@app.post("/predict/batch", dependencies=[Depends(require_key)])
def predict_batch(batch: Batch):
    probs = score(batch.applicants)
    log.info("batch size=%d flagged=%d", len(probs), sum(p >= THRESHOLD for p in probs))
    return {"results": [result(p) for p in probs]}