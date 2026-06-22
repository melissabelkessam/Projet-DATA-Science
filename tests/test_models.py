"""
test_models.py — Tests unitaires des modèles ML
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import train_logistic_regression, train_random_forest, train_xgboost
from src.config import MODEL_FILES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 14)
    # Classe déséquilibrée ~85/15
    y = np.random.choice([0, 1], n, p=[0.85, 0.15])
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]
    return X_train, X_test, y_train, y_test


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_logistic_regression_trains(synthetic_data, tmp_path, monkeypatch):
    X_train, X_test, y_train, y_test = synthetic_data
    # Rediriger la sauvegarde vers tmp_path
    monkeypatch.setitem(MODEL_FILES, "logistic_regression",
                        tmp_path / "lr.joblib")
    model = train_logistic_regression(X_train, y_train)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_random_forest_trains(synthetic_data, tmp_path, monkeypatch):
    X_train, X_test, y_train, y_test = synthetic_data
    monkeypatch.setitem(MODEL_FILES, "random_forest",
                        tmp_path / "rf.joblib")
    model = train_random_forest(X_train, y_train)
    assert hasattr(model, "feature_importances_")


def test_xgboost_trains(synthetic_data, tmp_path, monkeypatch):
    X_train, X_test, y_train, y_test = synthetic_data
    monkeypatch.setitem(MODEL_FILES, "xgboost",
                        tmp_path / "xgb.joblib")
    model = train_xgboost(X_train, y_train)
    assert hasattr(model, "feature_importances_")


def test_predictions_binary(synthetic_data, tmp_path, monkeypatch):
    """Les prédictions doivent être 0 ou 1."""
    X_train, X_test, y_train, _ = synthetic_data
    monkeypatch.setitem(MODEL_FILES, "xgboost", tmp_path / "xgb.joblib")
    model = train_xgboost(X_train, y_train)
    preds = model.predict(X_test)
    assert set(preds).issubset({0, 1}), "Prédictions non binaires"


def test_probabilities_in_range(synthetic_data, tmp_path, monkeypatch):
    """Les probabilités doivent être dans [0, 1]."""
    X_train, X_test, y_train, _ = synthetic_data
    monkeypatch.setitem(MODEL_FILES, "xgboost", tmp_path / "xgb.joblib")
    model = train_xgboost(X_train, y_train)
    probas = model.predict_proba(X_test)[:, 1]
    assert probas.min() >= 0.0
    assert probas.max() <= 1.0


def test_scale_pos_weight_applied(synthetic_data, tmp_path, monkeypatch):
    """XGBoost doit utiliser scale_pos_weight pour le déséquilibre."""
    X_train, _, y_train, _ = synthetic_data
    monkeypatch.setitem(MODEL_FILES, "xgboost", tmp_path / "xgb.joblib")
    model = train_xgboost(X_train, y_train, scale_pos_weight=5.7)
    assert model.get_params()["scale_pos_weight"] == 5.7
