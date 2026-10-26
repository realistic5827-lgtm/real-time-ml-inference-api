"""FastAPI REST service for real-time Iris species inference."""
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
if not MODEL_PATH.exists():
    raise RuntimeError("model.joblib not found. Run `python train_model.py` first.")
_artifact = joblib.load(MODEL_PATH)
_model = _artifact["model"]
_target_names = _artifact["target_names"]

app = FastAPI(title="Real-Time ML Inference API", version="1.0.0", description="Predict Iris species from four flower measurements.")

class PredictionRequest(BaseModel):
    sepal_length: float = Field(gt=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(gt=0, le=10, description="Petal length in cm")
    petal_width: float = Field(gt=0, le=10, description="Petal width in cm")

class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]

@app.get("/")
def root():
    return {"message": "Iris ML inference API is running", "docs": "/docs", "predict_endpoint": "/predict"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    features = np.array([[payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width]], dtype=float)
    try:
        probs = _model.predict_proba(features)[0]
        predicted_index = int(np.argmax(probs))
        return PredictionResponse(
            prediction=_target_names[predicted_index],
            probabilities={name: round(float(prob), 6) for name, prob in zip(_target_names, probs)},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
