from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    version="1.0.0",
    description="Production-style inference API for the Telco churn ML pipeline.",
)

model = None


class Customer(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: str


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: Customer):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    data = customer.model_dump()
    df = pd.DataFrame([data])

    try:
        probability = float(model.predict_proba(df)[0, 1])
        prediction = int(probability >= 0.5)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return {
        "prediction": prediction,
        "prediction_label": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(probability, 4),
    }
