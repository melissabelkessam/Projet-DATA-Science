"""
preprocessing.py — Pipeline de préparation des données
Maintenance Prédictive Industrielle · EFREI 2025-26
Amelia BOUKRI & Mélissa BELKESSAM
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import (
    DATA_FILE, COLS_TO_DROP, TARGET,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TEST_SIZE, RANDOM_STATE,
    PROCESSED_DIR, MODEL_FILES,
)


def load_raw_data(path=None) -> pd.DataFrame:
    """Charge le CSV brut et supprime les colonnes inutiles / à risque de leakage."""
    path = path or DATA_FILE
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Construit le ColumnTransformer sklearn :
    - Variables numériques  : imputation médiane + StandardScaler
    - Variables catégorielles : imputation mode + OneHotEncoder
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline,      NUMERIC_FEATURES),
        ("cat", categorical_pipeline,  CATEGORICAL_FEATURES),
    ])
    return preprocessor


def prepare_data(df: pd.DataFrame):
    """
    Effectue le split stratifié train/test et applique le pipeline de
    preprocessing (fit uniquement sur le train — évite le data leakage).

    Retourne
    --------
    X_train, X_test : np.ndarray  (données transformées)
    y_train, y_test : np.ndarray
    preprocessor    : ColumnTransformer fitté
    feature_names   : list[str]
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Noms des colonnes après encoding
    cat_cols = (preprocessor
                .named_transformers_["cat"]["encoder"]
                .get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + list(cat_cols)

    return (
        X_train_proc, X_test_proc,
        y_train.values, y_test.values,
        preprocessor, feature_names,
    )


def save_artifacts(X_train, X_test, y_train, y_test, preprocessor):
    """Sauvegarde le preprocessor et les splits numpy."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILES["preprocessor"].parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, MODEL_FILES["preprocessor"])
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy",  X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy",  y_test)
    print("✓ Preprocessor et données sauvegardés")


def load_processed_data():
    """Charge les splits numpy déjà preprocessés."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test  = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")
    return X_train, X_test, y_train, y_test


def load_preprocessor():
    """Charge le preprocessor sérialisé."""
    return joblib.load(MODEL_FILES["preprocessor"])


# ── Point d'entrée direct ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Chargement des données…")
    df = load_raw_data()
    print(f"  Dataset brut : {df.shape}")

    X_train, X_test, y_train, y_test, preprocessor, feat_names = prepare_data(df)
    print(f"  Train : {X_train.shape} | Test : {X_test.shape}")
    print(f"  Proportion pannes train : {y_train.mean():.1%}")
    print(f"  Features ({len(feat_names)}) : {feat_names}")

    save_artifacts(X_train, X_test, y_train, y_test, preprocessor)
