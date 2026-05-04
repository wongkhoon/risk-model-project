"""
FastAPI application for internal batch model serving.

This endpoint expects model-ready engineered features, not raw user input.
Because the trained model contains many features, this API is intended for
internal/system use rather than manual human entry.

Run
---
python -m uvicorn app:app --reload
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.predict import get_model_instance

app = FastAPI(
    title="Risk Model Batch Prediction API",
    description="Internal model-serving endpoint expecting full model-ready feature schema.",
)


@app.post("/predict-batch")
def predict_batch(data: List[Dict]) -> dict:
    """
    Generate predictions for a batch of model-ready records.

    Request body format:
    [
        {"feature_1": value, "feature_2": value, ...},
        {"feature_1": value, "feature_2": value, ...}
    ]
    """
    try:
        if not data:
            raise ValueError("Input batch is empty.")

        df = pd.DataFrame(data)
        predictions = get_model_instance().predict_batch(df)
        return {"predictions": predictions.tolist()}

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
