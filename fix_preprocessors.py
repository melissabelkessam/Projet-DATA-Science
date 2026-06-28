"""
Script de réparation des preprocessors
Exécuter depuis le dossier racine du projet :
    python fix_preprocessors.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

print(f"sklearn version : {sklearn.__version__}")
print("=" * 50)

# ── Chemins ──────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'predictive_maintenance_v3.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')

# ── Features ─────────────────────────────────────────
NUMERIC_FEATURES     = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                         'pressure_level', 'rpm', 'hours_since_maintenance', 'ambient_temp']
CATEGORICAL_FEATURES = ['machine_type', 'operating_mode']

# ── Pipeline commun ───────────────────────────────────
def build_preprocessor():
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', numeric_pipeline,     NUMERIC_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
    ])

# ══════════════════════════════════════════════════════
# 1. PREPROCESSOR CLASSIFICATION (failure_within_24h)
# ══════════════════════════════════════════════════════
print("\n[1/2] Recréation du preprocessor classification...")

df = pd.read_csv(DATA_PATH)

cols_to_drop = ['timestamp', 'machine_id', 'rul_hours',
                'failure_type', 'estimated_repair_cost']
df_clf = df.drop(columns=cols_to_drop)

TARGET_CLF = 'failure_within_24h'
X_clf = df_clf.drop(columns=[TARGET_CLF])
y_clf = df_clf[TARGET_CLF]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)

preprocessor_clf = build_preprocessor()
preprocessor_clf.fit(X_train_clf)

save_path = os.path.join(MODEL_DIR, 'preprocessor.joblib')
joblib.dump(preprocessor_clf, save_path)
print(f"✅ preprocessor.joblib sauvegardé → {save_path}")
print(f"   Train shape : {X_train_clf.shape} | Test shape : {X_test_clf.shape}")

# ══════════════════════════════════════════════════════
# 2. PREPROCESSOR RUL (rul_hours)
# ══════════════════════════════════════════════════════
print("\n[2/2] Recréation du preprocessor RUL...")

cols_to_drop_rul = ['timestamp', 'machine_id', 'failure_within_24h',
                    'failure_type', 'estimated_repair_cost']

TARGET_RUL = 'rul_hours'
df_rul = df.dropna(subset=[TARGET_RUL]).copy()

X_rul = df_rul[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_rul = df_rul[TARGET_RUL]

X_train_rul, X_test_rul, y_train_rul, y_test_rul = train_test_split(
    X_rul, y_rul, test_size=0.20, random_state=42
)

preprocessor_rul = build_preprocessor()
preprocessor_rul.fit(X_train_rul)

save_path_rul = os.path.join(MODEL_DIR, 'rul_preprocessor.joblib')
joblib.dump(preprocessor_rul, save_path_rul)
print(f"✅ rul_preprocessor.joblib sauvegardé → {save_path_rul}")
print(f"   Train shape : {X_train_rul.shape} | Test shape : {X_test_rul.shape}")

# ══════════════════════════════════════════════════════
# Vérification
# ══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("VÉRIFICATION FINALE")
print("=" * 50)

# Test preprocessor classification
test_input = pd.DataFrame([{
    'vibration_rms': 1.5, 'temperature_motor': 50.0,
    'current_phase_avg': 8.0, 'pressure_level': 50.0,
    'rpm': 1000.0, 'hours_since_maintenance': 100.0,
    'ambient_temp': 13.0, 'machine_type': 'CNC', 'operating_mode': 'normal'
}])

result_clf = preprocessor_clf.transform(test_input)
print(f"✅ preprocessor classification OK → shape : {result_clf.shape}")

result_rul = preprocessor_rul.transform(test_input)
print(f"✅ preprocessor RUL OK            → shape : {result_rul.shape}")

print("\n🎉 Tous les preprocessors sont recréés avec sklearn", sklearn.__version__)
print("   Relancez le dashboard : streamlit run dashboard/app.py")
