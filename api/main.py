from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Literal

app = FastAPI(
    title="Maintenance Predictive API",
    description="API de prediction de panne industrielle",
    version="1.0.0"
)

# Chargement modele et preprocessor
preprocessor = joblib.load('../models/preprocessor.joblib')
model = joblib.load('../models/xgboost.joblib')

# Schema d'entrée
class MachineData(BaseModel):
    machine_type: Literal['CNC', 'Pump', 'Compressor', 'Robotic Arm']
    operating_mode: Literal['normal', 'idle', 'peak']
    vibration_rms: float = Field(..., ge=0.35, le=10.0)
    temperature_motor: float = Field(..., ge=28.0, le=95.0)
    current_phase_avg: float = Field(..., ge=2.2, le=35.0)
    pressure_level: float = Field(..., ge=10.0, le=206.0)
    rpm: float = Field(..., ge=124.0, le=4098.0)
    hours_since_maintenance: float = Field(..., ge=0.0, le=575.0)
    ambient_temp: float = Field(..., ge=8.0, le=18.0)

@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "version": "1.0.0"}

@app.post("/predict")
def predict(data: MachineData):
    try:
        input_df = pd.DataFrame([{
            'machine_type': data.machine_type,
            'vibration_rms': data.vibration_rms,
            'temperature_motor': data.temperature_motor,
            'current_phase_avg': data.current_phase_avg,
            'pressure_level': data.pressure_level,
            'rpm': data.rpm,
            'operating_mode': data.operating_mode,
            'hours_since_maintenance': data.hours_since_maintenance,
            'ambient_temp': data.ambient_temp
        }])

        input_processed = preprocessor.transform(input_df)
        proba = float(model.predict_proba(input_processed)[0][1])
        prediction = 1 if proba >= 0.70 else 0

        return {
            "prediction": prediction,
            "probability": round(proba, 4),
            "risk_level": "ELEVE" if proba >= 0.70 else "FAIBLE" if proba < 0.30 else "MOYEN",
            "recommendation": "Intervention requise dans les 24h" if prediction == 1 else "Aucune action immediate requise"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "XGBoost",
        "target": "failure_within_24h",
        "features": 14,
        "metrics": {
            "ROC-AUC": 0.9955,
            "Recall": 0.9551,
            "F1": 0.9171
        },
        "threshold": 0.70
    }