# Maintenance Prédictive Industrielle

## Description
Projet Data Science - M1 Data Engineering & AI (EFREI 2025-26)  
Système intelligent de maintenance prédictive basé sur des données de capteurs industriels.

**Problématique :** Prédire si une machine va tomber en panne dans les 24h (classification binaire)

## Auteurs
- Melissa BELKESSAM
- Amelia BOUKRI

## Structure du projet
Projet Data Science/

├── api/          → API REST (FastAPI)

├── assets/       → Images et screenshots

├── dashboard/    → Interface Streamlit

├── data/         → Dataset (non versionné)

├── models/       → Modèles entraînés sauvegardés

├── notebooks/    → Jupyter Notebooks (EDA, modélisation)

├── reports/      → Rapports et figures

├── src/          → Code source Python

└── tests/        → Tests unitaires

## Dataset
- **Source :** [AI4I 2020 Predictive Maintenance Dataset - UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Taille :** 10 000 entrées, 14 variables
- **Variable cible :** `Machine failure` (0 = pas de panne, 1 = panne)

## Technologies
- Python, Pandas, Scikit-learn, TensorFlow/Keras
- Streamlit (dashboard)
- FastAPI (API REST)
- GitHub (versioning)

## Installation
```bash
pip install -r requirements.txt
```

## Modèles utilisés
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (XGBoost)
- MLP Neural Network (Deep Learning)

