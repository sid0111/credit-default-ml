from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

GOOD = dict(LIMIT_BAL=120000, EDUCATION=2,
            PAY_1=-1, PAY_2=2, PAY_3=0, PAY_4=0, PAY_5=0, PAY_6=2,
            BILL_AMT1=2682, BILL_AMT2=1725, BILL_AMT3=2682,
            BILL_AMT4=3272, BILL_AMT5=3455, BILL_AMT6=3261,
            PAY_AMT1=0, PAY_AMT2=1000, PAY_AMT3=1000,
            PAY_AMT4=1000, PAY_AMT5=0, PAY_AMT6=2000)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_valid():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["default_probability"] <= 1
    assert isinstance(body["flagged_high_risk"], bool)


def test_delinquent_scores_higher_than_clean():
    clean = {**GOOD, **{f"PAY_{i}": 0 for i in range(1, 7)}}
    late = {**GOOD, **{f"PAY_{i}": 3 for i in range(1, 7)}}
    p_clean = client.post("/predict", json=clean).json()["default_probability"]
    p_late = client.post("/predict", json=late).json()["default_probability"]
    assert p_late > p_clean + 0.2


def test_invalid_education_rejected():
    assert client.post("/predict", json={**GOOD, "EDUCATION": 9}).status_code == 422


def test_missing_field_rejected():
    bad = {k: v for k, v in GOOD.items() if k != "PAY_1"}
    assert client.post("/predict", json=bad).status_code == 422


def test_sensitive_field_rejected():
    assert client.post("/predict", json={**GOOD, "SEX": 1}).status_code == 422


def test_batch():
    r = client.post("/predict/batch", json={"applicants": [GOOD, GOOD]})
    assert r.status_code == 200 and len(r.json()["results"]) == 2

def test_api_key_enforced(monkeypatch):
    import src.app as appmod
    monkeypatch.setattr(appmod, "API_KEY", "secret")
    assert client.post("/predict", json=GOOD).status_code == 401
    ok = client.post("/predict", json=GOOD, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200