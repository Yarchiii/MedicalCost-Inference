# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from model_utils import make_inference, load_model
from sklearn.pipeline import Pipeline
from pickle import dumps


@pytest.fixture
def create_data() -> dict[str, int | float | str]:
    return {"age": 19, "sex": "female", "bmi": 27.9, "children": 0,
            "smoker": "yes", "region": "southwest"}


def test_make_inference(monkeypatch, create_data) -> None:
    def mock_get_predictions(_, data: pd.DataFrame) -> list[float]:
        assert create_data == {
            key: value[0] for key, value in data.to_dict("list").items()
        }
        return [17765.4795]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict", mock_get_predictions)

    result = make_inference(in_model, create_data)
    assert result == {"charges": 17765.48}


def test_make_inference_2d_output(monkeypatch, create_data) -> None:
    def mock_get_predictions(_, data: pd.DataFrame) -> list[list[float]]:
        return [[1725.5523]]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict", mock_get_predictions)

    assert make_inference(in_model, create_data) == {"charges": 1725.55}


@pytest.fixture()
def filepath_and_data(tmpdir):
    p = tmpdir.mkdir("datadir").join("fakedmodel.pkl")
    example: str = "Test message!"
    p.write_binary(dumps(example))
    return str(p), example


def test_load_model(filepath_and_data) -> None:
    assert filepath_and_data[1] == load_model(filepath_and_data[0])
