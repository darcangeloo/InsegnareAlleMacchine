from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

models_folder = Path(__file__).parent / "models"
encoder = joblib.load(models_folder / "encoder.pkl")
scaler = joblib.load(models_folder / "scaler.pkl")
classifier = joblib.load(models_folder / "classifier.pkl")
soglia = joblib.load(models_folder / "soglia.pkl")

categorical_columns = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"]
numeric_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

app = FastAPI()


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
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
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict(customer: Customer):
    customer_data = pd.DataFrame([customer.model_dump()])
    encoded_categories = encoder.transform(customer_data[categorical_columns])
    scaled_numbers = scaler.transform(customer_data[numeric_columns])
    all_features = np.hstack([scaled_numbers, encoded_categories])
    churn_probability = classifier.predict_proba(all_features)[0][1]
    return {"churn_probability": round(float(churn_probability), 4),
            "a_rischio": bool(churn_probability >= soglia)}
