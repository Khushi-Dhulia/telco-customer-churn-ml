from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_churn(customer: Dict[str, Any]) -> Dict[str, Any]:
    model = load_model()
    df = pd.DataFrame([customer])

    probability = float(model.predict_proba(df)[0, 1])
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "prediction_label": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(probability, 4),
    }


if __name__ == "__main__":
    sample_customer = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
    }

    print(predict_churn(sample_customer))
