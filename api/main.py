from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Literal
from pathlib import Path

"""
FastAPI = librairie Python pour créer des APIs rapidement
→ Validation automatique des données (via Pydantic)
→ Documentation Swagger automatique à /docs
→ Très rapide et moderne
"""

# ── Chemins absolus (fonctionne peu importe d'où on lance uvicorn) ────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT_DIR / "models"

app = FastAPI(
    title="Maintenance Predictive API",
    description="API de prediction de panne industrielle — EFREI 2025-26",
    version="1.0.0"
)

# ── Chargement modèle classification ─────────────────────────────────────────
try:
    preprocessor  = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model         = joblib.load(MODELS_DIR / "xgboost.joblib")
    _model_loaded = True
except Exception as e:
    _model_loaded = False
    _load_error   = str(e)

# ── Chargement modèle RUL ────────────────────────────────────────────────────
try:
    rul_model        = joblib.load(MODELS_DIR / "rul_model.joblib")
    rul_preprocessor = joblib.load(MODELS_DIR / "rul_preprocessor.joblib")
    _rul_loaded      = True
except Exception as e:
    _rul_loaded  = False
    _rul_error   = str(e)

# ── Schéma d'entrée avec validation Pydantic ─────────────────────────────────
class MachineData(BaseModel):
    machine_type:            Literal["CNC", "Pump", "Compressor", "Robotic Arm"]
    operating_mode:          Literal["normal", "idle", "peak"]
    vibration_rms:           float = Field(..., ge=0.35,  le=10.0,  description="Vibration RMS en mm/s")
    temperature_motor:       float = Field(..., ge=28.0,  le=95.0,  description="Température moteur en °C")
    current_phase_avg:       float = Field(..., ge=2.2,   le=35.0,  description="Courant de phase moyen en A")
    pressure_level:          float = Field(..., ge=10.0,  le=206.0, description="Pression hydraulique en bar")
    rpm:                     float = Field(..., ge=124.0, le=4098.0,description="Vitesse de rotation en tr/min")
    hours_since_maintenance: float = Field(..., ge=0.0,   le=575.0, description="Heures depuis dernière maintenance")
    ambient_temp:            float = Field(..., ge=8.0,   le=18.0,  description="Température ambiante en °C")

    model_config = {
        "json_schema_extra": {
            "example": {
                "machine_type": "CNC",
                "operating_mode": "normal",
                "vibration_rms": 0.9,
                "temperature_motor": 42.0,
                "current_phase_avg": 8.0,
                "pressure_level": 50.0,
                "rpm": 850.0,
                "hours_since_maintenance": 50.0,
                "ambient_temp": 13.0
            }
        }
    }

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Vérification de l'état du service")
def health():
    return {
        "status":        "ok" if _model_loaded else "error",
        "model":         "XGBoost",
        "model_loaded":  _model_loaded,
        "rul_loaded":    _rul_loaded,
        "version":       "1.0.0",
        "models_dir":    str(MODELS_DIR),
    }


@app.post("/predict", summary="Prédiction de panne à partir des capteurs")
def predict(data: MachineData):
    if not _model_loaded:
        raise HTTPException(status_code=503,
                            detail=f"Modèle non chargé : {_load_error}")
    try:
        input_df = pd.DataFrame([{
            "machine_type":            data.machine_type,
            "vibration_rms":           data.vibration_rms,
            "temperature_motor":       data.temperature_motor,
            "current_phase_avg":       data.current_phase_avg,
            "pressure_level":          data.pressure_level,
            "rpm":                     data.rpm,
            "operating_mode":          data.operating_mode,
            "hours_since_maintenance": data.hours_since_maintenance,
            "ambient_temp":            data.ambient_temp,
        }])

        input_processed = preprocessor.transform(input_df)
        proba           = float(model.predict_proba(input_processed)[0][1])
        prediction      = 1 if proba >= 0.70 else 0

        if proba >= 0.70:
            risk_level     = "ELEVE"
            recommendation = "Intervention de maintenance requise dans les 24h"
        elif proba >= 0.30:
            risk_level     = "MOYEN"
            recommendation = "Surveiller les capteurs — planifier une inspection"
        else:
            risk_level     = "FAIBLE"
            recommendation = "Aucune action immédiate requise"

        return {
            "prediction":     prediction,
            "probability":    round(proba, 4),
            "risk_level":     risk_level,
            "recommendation": recommendation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-rul", summary="Estimation de la durée de vie restante (RUL)")
def predict_rul(data: MachineData):
    if not _rul_loaded:
        raise HTTPException(status_code=503,
                            detail=f"Modèle RUL non chargé : {_rul_error}")
    try:
        input_df = pd.DataFrame([{
            "machine_type":            data.machine_type,
            "vibration_rms":           data.vibration_rms,
            "temperature_motor":       data.temperature_motor,
            "current_phase_avg":       data.current_phase_avg,
            "pressure_level":          data.pressure_level,
            "rpm":                     data.rpm,
            "operating_mode":          data.operating_mode,
            "hours_since_maintenance": data.hours_since_maintenance,
            "ambient_temp":            data.ambient_temp,
        }])

        input_processed = rul_preprocessor.transform(input_df)
        rul_hours       = max(0, float(rul_model.predict(input_processed)[0]))

        if rul_hours < 10:
            urgency        = "URGENT"
            recommendation = "Intervention immédiate requise !"
        elif rul_hours < 24:
            urgency        = "ATTENTION"
            recommendation = "Planifier maintenance sous 24h"
        else:
            urgency        = "OK"
            recommendation = "Machine opérationnelle"

        return {
            "rul_hours":      round(rul_hours, 1),
            "urgency":        urgency,
            "recommendation": recommendation,
            "mae":            9.42,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", summary="Informations sur le modèle déployé")
def model_info():
    return {
        "model_type": "XGBoost",
        "target":     "failure_within_24h",
        "features":   14,
        "threshold":  0.70,
        "metrics": {
            "ROC-AUC":   0.9955,
            "Recall":    0.9551,
            "Precision": 0.8820,
            "F1":        0.9171,
            "Accuracy":  0.9744,
        },
        "rul_model":          "Random Forest",
        "rul_mae":            9.42,
        "rul_r2":             0.67,
        "imbalance_strategy": "scale_pos_weight + seuil 0.70",
        "training_size":      19233,
        "test_size":          4809,
    }
