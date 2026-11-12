# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient

CLIENT = {"age": 19, "sex": "female", "bmi": 27.9, "children": 0,
          "smoker": "yes", "region": "southwest"}


@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    def mock_make_inference(*args, **kwargs) -> dict[str, float]:
        return {"charges": 17765.48}

    def mock_load_model(*args, **kwargs) -> None:
        return None

    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setattr("model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("model_utils.load_model", mock_load_model)

    from main import app
    return TestClient(app)


def test_healthcheck(init_test_client) -> None:
    response = init_test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_token_correctness(init_test_client) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=CLIENT,
    )
    assert response.status_code == 200
    assert "charges" in response.json()


def test_token_not_correctness(init_test_client) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer kedjkj"},
        json=CLIENT,
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid authentication credentials"
    }


def test_token_absent(init_test_client) -> None:
    response = init_test_client.post("/predictions", json=CLIENT)
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_inference(init_test_client) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=CLIENT,
    )
    assert response.status_code == 200
    assert response.json()["charges"] == 17765.48


@pytest.mark.parametrize(
    "field, value",
    [
        ("sex", "unknown"),
        ("smoker", "maybe"),
        ("region", "moscow"),
        ("age", 5),
        ("bmi", -1.0),
        ("children", -2),
    ],
)
def test_invalid_values(init_test_client, field, value) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json={**CLIENT, field: value},
    )
    assert response.status_code == 422


def test_missing_field(init_test_client) -> None:
    data = {key: value for key, value in CLIENT.items() if key != "smoker"}
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=data,
    )
    assert response.status_code == 422
