# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from pickle import load


def make_inference(in_model: Pipeline, in_data: dict) -> dict[str, float]:
    """Return the predicted insurance charges for in_data using in_model."""
    prediction = in_model.predict(pd.DataFrame(in_data, index=[0]))
    charges = float(np.ravel(prediction)[0])
    return {"charges": round(charges, 2)}


def load_model(path: str) -> Pipeline:
    """Return the model being read which stored on the path."""
    with open(path, "rb") as file:
        model: Pipeline = load(file)

    return model
