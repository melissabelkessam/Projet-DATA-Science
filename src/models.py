"""
models.py — Entraînement et sérialisation des modèles ML / DL
Maintenance Prédictive Industrielle · EFREI 2025-26
Amelia BOUKRI & Mélissa BELKESSAM
"""

import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE, SCALE_POS_WEIGHT,
    MLP_EPOCHS, MLP_BATCH_SIZE, MLP_LR, MLP_PATIENCE,
    MODEL_FILES,
)


# ── Modèle 1 : Logistic Regression (baseline) ────────────────────────────────

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Baseline interprétable.
    class_weight='balanced' gère le déséquilibre des classes.
    """
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILES["logistic_regression"])
    print("✓ Logistic Regression entraînée et sauvegardée")
    return model


# ── Modèle 2 : Random Forest ─────────────────────────────────────────────────

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Capture les non-linéarités ; pas besoin de standardisation.
    class_weight='balanced' gère le déséquilibre.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILES["random_forest"])
    print("✓ Random Forest entraîné et sauvegardé")
    return model


# ── Modèle 3 : XGBoost ───────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, scale_pos_weight=None) -> XGBClassifier:
    """
    Gradient boosting — meilleure performance sur ce dataset tabulaire.
    scale_pos_weight = ratio classe majoritaire / minoritaire.
    """
    spw = scale_pos_weight or SCALE_POS_WEIGHT
    model = XGBClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        scale_pos_weight=spw,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILES["xgboost"])
    print("✓ XGBoost entraîné et sauvegardé")
    return model


# ── Modèle 4 : MLP (Deep Learning) ───────────────────────────────────────────

def build_mlp(input_dim: int):
    """
    Architecture MLP :
    128 (relu) → BatchNorm → Dropout(0.3)
     64 (relu) → BatchNorm → Dropout(0.2)
     32 (relu) → 1 (sigmoid)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=MLP_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_mlp(X_train, y_train):
    """
    Entraîne le MLP avec EarlyStopping et class_weight.
    Retourne (model, history).

    Note : sur ce dataset tabulaire (24 042 lignes, 14 features),
    XGBoost surpasse le MLP (ROC-AUC 0.9955 vs 0.9743).
    Le MLP reste inclus pour la comparaison ML vs DL.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    model = build_mlp(input_dim=X_train.shape[1])

    # Poids des classes proportionnels au déséquilibre
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: n_neg / n_pos}

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=MLP_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0,
    )
    model.save(MODEL_FILES["mlp"])
    print(f"✓ MLP entraîné ({len(history.history['loss'])} epochs) et sauvegardé")
    return model, history


# ── Chargement ────────────────────────────────────────────────────────────────

def load_model(name: str):
    """
    Charge un modèle sérialisé.
    name : 'logistic_regression' | 'random_forest' | 'xgboost' | 'mlp'
    """
    if name == "mlp":
        from tensorflow.keras.models import load_model as keras_load
        return keras_load(MODEL_FILES["mlp"])
    return joblib.load(MODEL_FILES[name])


# ── Point d'entrée direct ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocessing import load_processed_data

    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"Données chargées : train {X_train.shape} | test {X_test.shape}")

    train_logistic_regression(X_train, y_train)
    train_random_forest(X_train, y_train)
    train_xgboost(X_train, y_train)
    train_mlp(X_train, y_train)
    print("\n✓ Tous les modèles sont entraînés.")
