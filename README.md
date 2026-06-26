# Maintenance Prédictive Industrielle — EFREI 2025-26

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-success)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35-red)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.111-009688)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-28%2F28%20passed-brightgreen)]()
[![RNCP](https://img.shields.io/badge/RNCP40875-Bloc%202-blueviolet)](https://www.francecompetences.fr/)

> **Projet Data Science** · M1 Mastère Data Engineering & IA · EFREI Paris · 2025-2026  
> **Certification** · RNCP40875 Expert en Ingénierie de Données — Bloc 2 : Piloter et implémenter des solutions d'IA  
> **Auteurs** · Amelia BOUKRI & Mélissa BELKESSAM  
> **Encadrante** · Sarah Malaeb  
> **GitHub** · https://github.com/MelissaBelkessam/maintenance-predictive-industrielle

---

## Problématique

Dans un contexte industriel, les arrêts non planifiés de machines représentent un coût considérable. Ce projet développe un **système intelligent de maintenance prédictive** capable de prédire si une machine va tomber en panne dans les **24 heures** à partir de données capteurs (vibration, température, pression, RPM…), afin d'anticiper les interventions et réduire les arrêts non planifiés.

---

## Résultats principaux

| Modèle | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** ✓ | **0.9744** | **0.9551** | **0.9171** | **0.9955** |
| Random Forest | 0.9682 | 0.8820 | 0.8914 | 0.9938 |
| MLP (Deep Learning) | 0.9310 | 0.8989 | 0.7940 | 0.9743 |
| Logistic Regression | 0.9102 | 0.8947 | 0.7468 | 0.9588 |

> Seuil de décision optimisé : **0.70** (maximise F1) · PR-AUC : **0.9741**  
> Variable cible : `failure_within_24h` · Déséquilibre 85.2% / 14.8%  
> **Bonus RUL** — Random Forest · MAE = 9.42h · R² = 0.6743

---

## Structure du projet

```
Projet Data Science/
├── api/                      → API REST FastAPI
│   └── main.py               → Endpoints /predict /health /model-info
├── assets/                   → Logo EFREI
├── dashboard/                → Interface Streamlit (8 pages)
│   └── app.py
├── data/
│   ├── predictive_maintenance_v3.csv  → Dataset (24 042 lignes · 15 variables)
│   └── processed/            → Splits numpy (X_train, X_test, y_train, y_test)
├── models/                   → Modèles sérialisés
│   ├── xgboost.joblib        → Modèle final (classification)
│   ├── random_forest.joblib
│   ├── logistic_regression.joblib
│   ├── mlp_model.keras
│   ├── rul_model.joblib      → Modèle bonus RUL (régression)
│   └── preprocessor.joblib
├── notebooks/                → Jupyter Notebooks
│   ├── 01_EDA.ipynb          → Analyse exploratoire
│   ├── 02_Preprocessing.ipynb → Pipeline de preprocessing
│   ├── 03_Models.ipynb       → Modélisation, évaluation, KFold
│   ├── 04_Interpretability.ipynb → Feature Importance + SHAP
│   └── 05_Regression_RUL.ipynb   → Tâche bonus RUL (4 modèles)
├── reports/                  → Figures et graphiques (EDA, évaluation, SHAP)
├── src/                      → Modules Python réutilisables
│   ├── config.py             → Paramètres centralisés (chemins, features, seuil)
│   ├── data_loader.py        → Chargement et description du dataset
│   ├── preprocessing.py      → Pipeline sklearn (imputation + scaling + encoding)
│   ├── models.py             → Entraînement LR / RF / XGBoost / MLP
│   ├── evaluation.py         → Métriques, CV, optimisation seuil
│   ├── interpretability.py   → Feature Importance + SHAP
│   └── ecoresponsabilite.py  → Mesure CO₂ via CodeCarbon
├── streamlit/
│   └── config.toml           → Configuration thème Streamlit
├── tests/                    → Tests unitaires (pytest) — 28/28 passent
│   ├── test_smoke.py         → Tests d'imports et de configuration
│   ├── test_preprocessing.py → Tests du pipeline preprocessing
│   ├── test_models.py        → Tests des modèles ML
│   └── test_api.py           → Tests de l'API FastAPI
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/MelissaBelkessam/maintenance-predictive-industrielle.git
cd maintenance-predictive-industrielle

# Installer les dépendances
pip install -r requirements.txt
```

---

## Dataset

- **Source** : [Industrial Machine Predictive Maintenance — Kaggle](https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance)
- **Fichier** : `data/predictive_maintenance_v3.csv` (inclus dans le dépôt)
- **Taille** : 24 042 enregistrements · 15 variables
- **Variable cible** : `failure_within_24h` (0 = sain, 1 = panne imminente)
- **Déséquilibre** : 85.2% sain / 14.8% panne

---

## Lancer le dashboard

```bash
# Depuis la racine du projet
streamlit run dashboard/app.py
```
→ Ouvre automatiquement http://localhost:8501

**8 pages disponibles :**
1. **Vue d'ensemble** — KPI globaux et répartition des classes
2. **Analyse des données** — Distributions et corrélations interactives des capteurs
3. **Prédiction temps réel** — Formulaire de saisie + jauge de risque (connecté à l'API)
4. **Durée de vie restante (RUL)** — Estimation RUL via régression (tâche bonus)
5. **État du parc machines** — Vue d'ensemble des 15 machines avec statut et risque
6. **Impact économique** — ROI du système IA vs maintenance corrective
7. **Performance des modèles** — Tableau comparatif + radar chart des 4 modèles
8. **Interprétabilité SHAP** — Feature importance globale et explication individuelle

> **Note** : le dashboard fonctionne en mode local (sans API) ou connecté à l'API FastAPI pour les prédictions temps réel.

---

## Lancer l'API REST

```bash
# Depuis la racine du projet
uvicorn api.main:app --reload --port 8000
```
→ Documentation Swagger interactive : http://localhost:8000/docs

**Endpoints :**

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | État du service et statut du modèle |
| POST | `/predict` | Prédiction de panne à partir des features capteurs |
| GET | `/model-info` | Métriques et paramètres du modèle déployé |

**Exemple de requête `/predict` :**
```json
{
  "machine_type": "Compressor",
  "operating_mode": "peak",
  "vibration_rms": 7.8,
  "temperature_motor": 85.0,
  "current_phase_avg": 25.0,
  "pressure_level": 150.0,
  "rpm": 3500.0,
  "hours_since_maintenance": 420.0,
  "ambient_temp": 17.0
}
```

**Réponse :**
```json
{
  "prediction": 1,
  "probability": 0.987,
  "risk_level": "ELEVE",
  "recommendation": "Intervention de maintenance requise dans les 24h"
}
```

---

## Lancer les tests

```bash
# Depuis la racine du projet
pytest tests/ -v
```

28 tests unitaires couvrant : imports, preprocessing, modèles ML, et API REST.

---

## Démo complète (dashboard + API)

Pour la démo avec l'API connectée, lancer les deux services en parallèle :

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run dashboard/app.py
```

---

## Technologies

| Composant | Technologie |
|---|---|
| Langage | Python 3.10+ |
| ML | Scikit-learn, XGBoost, imbalanced-learn |
| Deep Learning | TensorFlow / Keras (MLP) |
| Interprétabilité | SHAP |
| Écoresponsabilité | CodeCarbon |
| Dashboard | Streamlit + Plotly |
| API | FastAPI + Uvicorn + Pydantic |
| Sérialisation | joblib |
| Tests | pytest |
| Versioning | Git + GitHub |

---

## Approche méthodologique

1. **EDA** — Distributions, corrélations, déséquilibre 85/15, valeurs manquantes 2–4%
2. **Preprocessing** — Imputation médiane + StandardScaler + OneHotEncoder en pipeline sklearn (sans data leakage)
3. **Gestion du déséquilibre** — SMOTE, ROS, RUS, class_weight + optimisation seuil 0.70
4. **Modélisation** — 4 modèles comparés : Logistic Regression, Random Forest, XGBoost, MLP
5. **Validation** — Stratified K-Fold 5 folds · Métriques : Recall, F1, ROC-AUC, PR-AUC
6. **Interprétabilité** — Feature Importance (Gini) + SHAP global et individuel
7. **Déploiement** — Dashboard Streamlit 8 pages + API REST FastAPI
8. **Écoresponsabilité** — Mesure CO₂ via CodeCarbon (XGBoost : 60× moins que le MLP)
9. **Bonus RUL** — Régression sur `rul_hours` : 4 modèles comparés, Random Forest retenu (MAE = 9.42h)

---

## Écoresponsabilité

| Modèle | Durée entraînement | CO₂ (g) | Inférence (ms) |
|---|---|---|---|
| Logistic Regression | 0.42s | 0.00012 | 0.08 |
| **XGBoost** ✓ | 3.1s | 0.0009 | 0.21 |
| Random Forest | 8.3s | 0.0024 | 0.45 |
| MLP (Deep Learning) | 187s | 0.054 | 2.8 |

> XGBoost émet **60× moins de CO₂** que le MLP pour des performances supérieures sur ce dataset tabulaire.