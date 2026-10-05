"""Load the saved model pipeline and make validated AQI estimates."""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing import FEATURE_COLUMNS, aqi_to_category

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_INFO_PATH = MODELS_DIR / "best_model_name.json"
_model = None

FEATURE_RANGES = {
    "T": (-50, 60), "TM": (-50, 70), "Tm": (-90, 50),
    "SLP": (800, 1100), "H": (0, 100), "VV": (0, 100),
    "V": (0, 200), "VM": (0, 200),
}


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODELS_DIR / "best_model.pkl")
    return _model


def validate_inputs(inputs: dict) -> dict:
    """Return numeric weather values or raise a readable validation error."""
    values = {}
    for column in FEATURE_COLUMNS:
        if column not in inputs:
            raise ValueError(f"Please provide a value for {column}.")
        try:
            value = float(inputs[column])
        except (TypeError, ValueError):
            raise ValueError(f"{column} must be a valid number.") from None
        if not np.isfinite(value):
            raise ValueError(f"{column} must be a finite number.")
        lower, upper = FEATURE_RANGES[column]
        if not lower <= value <= upper:
            raise ValueError(f"{column} must be between {lower} and {upper}.")
        values[column] = value

    if values["TM"] < values["T"]:
        raise ValueError("Maximum temperature (TM) must be at least the mean temperature (T).")
    if values["T"] < values["Tm"]:
        raise ValueError("Mean temperature (T) must be at least the minimum temperature (Tm).")
    if values["VM"] < values["V"]:
        raise ValueError("Maximum wind speed (VM) must be at least the mean wind speed (V).")
    return values


def predict_aqi(inputs: dict) -> dict:
    """inputs: dict with keys T, TM, Tm, SLP, H, VV, V, VM (weather values).
    Returns predicted AQI and its CPCB category."""
    model = load_model()
    values = validate_inputs(inputs)
    row = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    aqi_pred = float(model.predict(row)[0])
    aqi_pred = float(np.clip(aqi_pred, 0, 500))
    category = aqi_to_category(aqi_pred)
    model_name = "Unknown model"
    if MODEL_INFO_PATH.exists():
        with MODEL_INFO_PATH.open(encoding="utf-8") as model_info_file:
            model_name = json.load(model_info_file).get("best_model", model_name)
    return {"aqi": round(aqi_pred, 1), "category": category, "model": model_name}


if __name__ == "__main__":
    sample = {"T": 15.0, "TM": 20.0, "Tm": 10.0, "SLP": 1018.0,
              "H": 80, "VV": 1.2, "V": 4.0, "VM": 10.0}
    print(predict_aqi(sample))
