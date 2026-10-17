import pytest
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def valid_iris():
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_single_prediction():
    response = client.post("/predict", json=valid_iris())

    assert response.status_code == 200

    result = response.json()

    assert result["class_id"] == 0
    assert result["species"] == "setosa"
    assert 0 <= result["confidence"] <= 1


def test_batch_prediction():
    samples = [
        valid_iris(),
        {
            "sepal_length": 6.0,
            "sepal_width": 2.9,
            "petal_length": 4.5,
            "petal_width": 1.5,
        },
        {
            "sepal_length": 6.7,
            "sepal_width": 3.0,
            "petal_length": 5.2,
            "petal_width": 2.3,
        },
    ]

    response = client.post("/predict/batch", json={"samples": samples})

    assert response.status_code == 200

    result = response.json()

    assert result["total_predictions"] == 3
    assert len(result["predictions"]) == 3
    assert [item["species"] for item in result["predictions"]] == [
        "setosa",
        "versicolor",
        "virginica",
    ]


@pytest.mark.parametrize(
    "field",
    ["sepal_length", "sepal_width", "petal_length", "petal_width"],
)
def test_negative_measurements_rejected(field):
    payload = valid_iris()
    payload[field] = -1.0

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_field_rejected():
    payload = valid_iris()
    del payload["petal_width"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_excessive_measurement_rejected():
    payload = valid_iris()
    payload["sepal_length"] = 20.0

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_empty_batch_rejected():
    response = client.post("/predict/batch", json={"samples": []})

    assert response.status_code == 422


def test_batch_limit_enforced():
    samples = [valid_iris()] * 101

    response = client.post("/predict/batch", json={"samples": samples})

    assert response.status_code == 422