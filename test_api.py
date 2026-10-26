from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_predict_returns_label_and_probabilities():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"setosa", "versicolor", "virginica"}
    assert set(body["probabilities"]) == {"setosa", "versicolor", "virginica"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01

def test_predict_rejects_missing_feature():
    payload = dict(VALID_PAYLOAD)
    payload.pop("petal_width")
    assert client.post("/predict", json=payload).status_code == 422

def test_predict_rejects_non_positive_measurement():
    payload = dict(VALID_PAYLOAD)
    payload["petal_width"] = 0
    assert client.post("/predict", json=payload).status_code == 422
