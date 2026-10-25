"""Model server for the breast cancer classifier (runs on Hugging Face Spaces)."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow import keras

BASE_DIR = Path(__file__).parent

# compile=False: we only need predictions, so no training setup is loaded.
model = keras.models.load_model(BASE_DIR / "breast_cancer_model.keras", compile=False)
scaler = joblib.load(BASE_DIR / "scaler.pkl")
feature_names = json.loads((BASE_DIR / "feature_names.json").read_text())

# "*" lets any website call this server. This is fine for a public demo.
# To restrict it later, set ALLOWED_ORIGINS to your site address.
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="Breast cancer classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictRequest(BaseModel):
    features: list[float]  # 30 numbers, in the same order as feature_names.json


@app.get("/")
def home():
    return {"message": "Breast cancer classifier API is running. Open /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != len(feature_names):
        raise HTTPException(status_code=422, detail=f"Expected {len(feature_names)} values, got {len(req.features)}.")

    row = pd.DataFrame([req.features], columns=feature_names)
    scaled = scaler.transform(row).astype("float32")
    probability = float(model.predict(scaled, verbose=0).ravel()[0])

    return {"label": "Malignant" if probability >= 0.5 else "Benign", "probability": probability}
