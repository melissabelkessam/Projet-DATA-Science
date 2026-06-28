"""
Script de réparation du modèle RUL
Exécuter depuis le dossier racine du projet :
    python fix_rul_model.py
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print(f"sklearn version : {sklearn.__version__}")
print("=" * 50)

# ── Chemins ──────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'predictive_maintenance_v3.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ── Features ─────────────────────────────────────────
NUMERIC_FEATURES     = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                         'pressure_level', 'rpm', 'hours_since_maintenance', 'ambient_temp']
CATEGORICAL_FEATURES = ['machine_type', 'operating_mode']
TARGET_RUL           = 'rul_hours'

print(f"\n[1/3] Chargement du dataset...")
df = pd.read_csv(DATA_PATH)
df_rul = df.dropna(subset=[TARGET_RUL]).copy()
print(f"Dataset RUL : {len(df_rul):,} lignes")

# ── Split ─────────────────────────────────────────────
X = df_rul[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df_rul[TARGET_RUL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train : {X_train.shape} | Test : {X_test.shape}")

# ── Pipeline complet preprocessor + modèle ────────────
print(f"\n[2/3] Entraînement du pipeline Random Forest RUL...")

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline,     NUMERIC_FEATURES),
    ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
])

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline_rf.fit(X_train, y_train)
print("✅ Pipeline entraîné !")

# ── Évaluation ────────────────────────────────────────
y_pred = pipeline_rf.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"\nRésultats :")
print(f"MAE = {mae:.2f}h")
print(f"R²  = {r2:.4f}")

# ── Sauvegarde ────────────────────────────────────────
print(f"\n[3/3] Sauvegarde...")
save_path = os.path.join(MODEL_DIR, 'rul_model.joblib')
joblib.dump(pipeline_rf, save_path)
print(f"✅ rul_model.joblib sauvegardé → {save_path}")

# ── Test final ───────────────────────────────────────
test_input = pd.DataFrame([{
    'vibration_rms': 2.0, 'temperature_motor': 55.0,
    'current_phase_avg': 9.0, 'pressure_level': 60.0,
    'rpm': 1200.0, 'hours_since_maintenance': 150.0,
    'ambient_temp': 13.0, 'machine_type': 'CNC', 'operating_mode': 'normal'
}])
pred = pipeline_rf.predict(test_input)[0]
print(f"\n✅ Test prédiction RUL : {pred:.1f}h ← fonctionne !")
print(f"\n🎉 rul_model.joblib recréé avec sklearn {sklearn.__version__}")
print("   Relancez le dashboard : streamlit run dashboard/app.py")
