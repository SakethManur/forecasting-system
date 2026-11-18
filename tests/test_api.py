"""API tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client(trained_registry):
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_lists_each_state_with_its_best_model(client):
    response = client.get("/models")

    assert response.status_code == 200
    models = {row["state"]: row for row in response.json()["models"]}
    assert sorted(models) == ["Alpha", "Beta"]
    for row in models.values():
        assert row["best_model"] in row["available_models"]
        assert row["best_metrics"]["rmse"] > 0


def test_predict_returns_one_value_per_week(client):
    response = client.post("/predict", json={"state": "Alpha", "weeks_ahead": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "Alpha"
    assert body["weeks_ahead"] == 3
    assert len(body["forecast_values"]) == 3
    assert all(value["predicted_sales"] >= 0 for value in body["forecast_values"])


def test_predict_unknown_state_returns_400(client):
    response = client.post("/predict", json={"state": "Atlantis", "weeks_ahead": 2})
    assert response.status_code == 400
    assert "Unknown state" in response.json()["detail"]


@pytest.mark.parametrize("payload", [{"state": "Alpha", "weeks_ahead": 0}, {"weeks_ahead": 2}])
def test_predict_rejects_invalid_requests_with_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422
