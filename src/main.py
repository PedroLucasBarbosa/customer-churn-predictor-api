"""
FastAPI application that serves the churn prediction model.

Run locally:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

Docs available at:
    http://localhost:8000/docs
"""

import json
import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.schemas import (
    CustomerData,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

MODEL_PATH = "models/churn_model.joblib"
METADATA_PATH = "models/model_metadata.json"

ml_artifacts = {"model": None, "metadata": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from %s", MODEL_PATH)
    ml_artifacts["model"] = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        ml_artifacts["metadata"] = json.load(f)
    logger.info("Model loaded: %s", ml_artifacts["metadata"]["model_type"])
    yield
    ml_artifacts["model"] = None
    ml_artifacts["metadata"] = None


app = FastAPI(
    title="Customer Churn Predictor API",
    description="Predicts the probability that a customer will churn, based on account and service data.",
    version="1.0.0",
    lifespan=lifespan,
)


def _risk_level(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Medium"
    return "High"


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    metadata = ml_artifacts.get("metadata")
    return HealthResponse(
        status="ok",
        model_loaded=ml_artifacts.get("model") is not None,
        model_version=metadata["version"] if metadata else "unknown",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    metadata = ml_artifacts.get("metadata")
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")
    return ModelInfoResponse(**metadata)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerData):
    model = ml_artifacts.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame([customer.model_dump()])

    try:
        proba = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        churn_probability = float(proba[classes.index("Yes")])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    prediction = "Yes" if churn_probability >= 0.5 else "No"

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(churn_probability, 4),
        risk_level=_risk_level(churn_probability),
    )


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Customer Churn Predictor API",
        "docs": "/docs",
        "health": "/health",
    }
