"""
data_loader.py — Chargement et inspection rapide du dataset
Maintenance Prédictive Industrielle · EFREI 2025-26
Amelia BOUKRI & Mélissa BELKESSAM
"""

import pandas as pd
import numpy as np
from src.config import DATA_FILE, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def load_and_describe(path=None) -> pd.DataFrame:
    """Charge le CSV et affiche un résumé statistique."""
    path = path or DATA_FILE
    df = pd.read_csv(path)

    print(f"Shape         : {df.shape}")
    print(f"Colonnes      : {list(df.columns)}")
    print(f"\nValeurs manquantes :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDistribution cible :\n{df[TARGET].value_counts(normalize=True).round(3)}")
    return df


def get_class_imbalance_ratio(df: pd.DataFrame) -> float:
    """Retourne le ratio classe majoritaire / minoritaire."""
    counts = df[TARGET].value_counts()
    ratio = counts.max() / counts.min()
    print(f"Ratio déséquilibre : {ratio:.2f}:1")
    return ratio


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau des valeurs manquantes par colonne."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    result = pd.DataFrame({
        "Valeurs manquantes": missing,
        "Pourcentage (%)":    (missing / len(df) * 100).round(2),
    })
    return result
