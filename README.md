# Maintenance Prédictive Industrielle — EFREI 2025-26

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-success)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35-red)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.111-009688)](https://fastapi.tiangolo.com/)
[![RNCP](https://img.shields.io/badge/RNCP40875-Bloc%202-blueviolet)](https://www.francecompetences.fr/)

> **Projet Data Science** · M1 Mastère Data Engineering & IA · EFREI Paris · 2025-2026  
> **Certification** · RNCP40875 Expert en Ingénierie de Données — Bloc 2 : Piloter et implémenter des solutions d'IA  
> **Auteurs** · Amelia BOUKRI & Mélissa BELKESSAM  
> **Encadrante** · Sarah Malaeb

---

## Problématique

Prédire si une machine industrielle va tomber en panne dans les **24 heures** à partir de données capteurs (vibration, température, pression, RPM…) afin d'anticiper les interventions et réduire les arrêts non planifiés.

---

## Résultats principaux

| Modèle | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** ✓ | **0.9744** | **0.9551** | **0.9171** | **0.9955** |
| Random Forest | 0.9682 | 0.8820 | 0.8914 | 0.9938 |
| MLP (Deep Learning) | 0.9310 | 0.8989 | 0.7940 | 0.9743 |
| Logistic Regression | 0.9102 | 0.8947 | 0.7468 | 0.9588 |

> Seuil de décision optimisé : **0.70** (maximise F1)  
> Variable cible : `failure_within_24h` · Déséquilibre 85.2% / 14.8%

---

## Structure du projet

```
Projet Data Science/
├── api/                    → API REST FastAPI (/predict /health /model-info)
├── assets/                 → Logo EFREI
├── dashboard/              → Interface Streamlit (5 pages)
│   └── app.py
├── data/
│   └── processed/          → Splits numpy (X_train, X_test, y_train, y_test)
├── models/                 → Modèles sérialisés (joblib / keras)
├── notebooks/              → Jupyter Notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Models.ipynb
│   └── 04_Interpretability.ipynb
├── reports/                → Figures et graphiques EDA / évaluation
├── src/                    → Modules Python réutilisables
│   ├── config.py           → Paramètres centralisés
│   ├── data_loader.py      → Chargement et description du dataset
│   ├── preprocessing.py    → Pipeline sklearn (imputation + scaling + encoding)
│   ├── models.py           → Entraînement LR / RF / XGBoost / MLP
│   ├── evaluation.py       → Métriques, CV, seuil de décision
│   └── interpretability.py → Feature Importance + SHAP
├── tests/                  → Tests unitaires (pytest)
│   ├── test_smoke.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
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
- **Fichier** : `data/predictive_maintenance_v3.csv` *(non versionné — à télécharger depuis Kaggle)*
- **Taille** : 24 042 enregistrements · 15 variables
- **Variable cible** : `failure_within_24h` (0 = sain, 1 = panne imminente)

---

## Lancer le dashboard

```bash
cd dashboard
streamlit run app.py
```
→ Ouvre automatiquement http://localhost:8501

**5 pages disponibles :**
1. Vue d'ensemble — KPI et répartition des classes
2. Analyse des données — distributions et corrélations interactives
3. Prédiction temps réel — formulaire de saisie + jauge de risque
4. Performance des modèles — tableau comparatif + radar chart
5. Interprétabilité SHAP — feature importance interactive

---

## Lancer l'API REST

```bash
cd api
uvicorn main:app --reload --port 8000
```
→ Documentation Swagger : http://localhost:8000/docs

**Endpoints :**

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | État du service |
| POST | `/predict` | Prédiction à partir des features capteurs |
| GET | `/model-info` | Métriques et paramètres du modèle |

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
  "recommendation": "Intervention requise dans les 24h"
}
```

---

## Lancer les tests

```bash
# Depuis la racine du projet
pytest tests/ -v
```

---

## Technologies

| Composant | Technologie |
|---|---|
| Langage | Python 3.10+ |
| ML | Scikit-learn, XGBoost, imbalanced-learn |
| Deep Learning | TensorFlow / Keras (MLP) |
| Interprétabilité | SHAP |
| Dashboard | Streamlit + Plotly |
| API | FastAPI + Uvicorn + Pydantic |
| Sérialisation | joblib |
| Versioning | Git + GitHub |

---

## Approche méthodologique

1. **EDA** — distributions, corrélations, déséquilibre 85/15
2. **Preprocessing** — imputation médiane + StandardScaler + OneHotEncoder (pipeline sklearn, sans data leakage)
3. **Modélisation** — 4 modèles comparés : Logistic Regression, Random Forest, XGBoost, MLP
4. **Gestion du déséquilibre** — class_weight, SMOTE, ROS, RUS + seuil de décision 0.70
5. **Validation** — Stratified K-Fold 5 folds
6. **Interprétabilité** — Feature Importance + SHAP (global + individuel)
7. **Déploiement** — Dashboard Streamlit + API FastAPI
