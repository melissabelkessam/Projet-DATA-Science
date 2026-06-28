"""
config.py — Paramètres centralisés du projet
Maintenance Prédictive Industrielle · EFREI 2025-26
Amelia BOUKRI & Mélissa BELKESSAM
"""
"""
config.py centralise tous les paramètres du projet — chemins, 
features, seuil de décision, hyperparamètres. 
Si on veut changer un paramètre, on le modifie une seule fois ici et ça se propage automatiquement partout.
 C'est une bonne pratique d'ingénierie logicielle.


L'ordre de travail
DÉVELOPPEMENT (notebooks) :
→ On teste, on découvre, on trouve le meilleur seuil = 0.70
→ Les notebooks gardent les valeurs en dur

PRODUCTION (modules src) :
→ On écrit les vraies valeurs dans config.py
→ Les modules importent config.py

Notebook = cahier de brouillon de nous les data scinetist 
→ plein de ratures, de tests, de valeurs essayées

config.py = rapport final propre
→ contient les valeurs définitives validées
→ utilisé par l'équipe en production
"""

import os
from pathlib import Path

# ── Racine du projet ──────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
PROCESSED_DIR = DATA_DIR / "processed"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_FILE = DATA_DIR / "predictive_maintenance_v3.csv"
TARGET    = "failure_within_24h"

COLS_TO_DROP = ["timestamp", "machine_id", "rul_hours",
                "failure_type", "estimated_repair_cost"]

NUMERIC_FEATURES = [
    "vibration_rms", "temperature_motor", "current_phase_avg",
    "pressure_level", "rpm", "hours_since_maintenance", "ambient_temp",
]
CATEGORICAL_FEATURES = ["machine_type", "operating_mode"]

# ── Split ─────────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── Modèles ───────────────────────────────────────────────────────────────────
MODEL_FILES = {
    "logistic_regression": MODELS_DIR / "logistic_regression.joblib",
    "random_forest":       MODELS_DIR / "random_forest.joblib",
    "xgboost":             MODELS_DIR / "xgboost.joblib",
    "mlp":                 MODELS_DIR / "mlp_model.keras",
    "preprocessor":        MODELS_DIR / "preprocessor.joblib",
}

# Seuil de décision optimisé (maximise F1 sur le jeu de validation)
DECISION_THRESHOLD = 0.70

# Poids XGBoost (ratio classe majoritaire / minoritaire)
# Calculé sur 24 042 obs : 20 482 saines / 3 560 pannes ≈ 5.75
SCALE_POS_WEIGHT = 16385 / 2848  # valeur issue du notebook 03

# ── MLP ───────────────────────────────────────────────────────────────────────
MLP_EPOCHS      = 100
MLP_BATCH_SIZE  = 256
MLP_LR          = 0.001
MLP_PATIENCE    = 10   # EarlyStopping

# ── Validation croisée ────────────────────────────────────────────────────────
CV_FOLDS = 5

# ── Dashboard / API ───────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# Palettes couleurs cohérentes avec le dashboard
COLOR_BLUE  = "#163767"
COLOR_RED   = "#E53935"
COLOR_GREEN = "#43A047"
COLOR_ORANGE = "#FF8C00"
