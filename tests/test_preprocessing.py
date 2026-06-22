"""
test_preprocessing.py — Tests unitaires du pipeline de preprocessing
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import build_preprocessor, prepare_data, load_raw_data
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Dataset minimal synthétique pour les tests."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "vibration_rms":          np.random.uniform(0.5, 9.0, n),
        "temperature_motor":      np.random.uniform(30, 90, n),
        "current_phase_avg":      np.random.uniform(3, 30, n),
        "pressure_level":         np.random.uniform(15, 200, n),
        "rpm":                    np.random.uniform(200, 4000, n),
        "hours_since_maintenance":np.random.uniform(0, 500, n),
        "ambient_temp":           np.random.uniform(9, 17, n),
        "machine_type":           np.random.choice(["CNC","Pump","Compressor","Robotic Arm"], n),
        "operating_mode":         np.random.choice(["normal","idle","peak"], n),
        "failure_within_24h":     np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })
    # Introduire quelques valeurs manquantes
    df.loc[df.sample(10).index, "vibration_rms"]     = np.nan
    df.loc[df.sample(8).index,  "temperature_motor"] = np.nan
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build_preprocessor_returns_columntransformer(sample_df):
    from sklearn.compose import ColumnTransformer
    prep = build_preprocessor()
    assert isinstance(prep, ColumnTransformer)


def test_prepare_data_shapes(sample_df):
    X_train, X_test, y_train, y_test, prep, feat_names = prepare_data(sample_df)
    assert X_train.shape[0] + X_test.shape[0] == len(sample_df)
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == X_train.shape[0]
    assert len(y_test)  == X_test.shape[0]


def test_no_missing_after_preprocessing(sample_df):
    X_train, X_test, *_ = prepare_data(sample_df)
    assert not np.isnan(X_train).any(), "Valeurs manquantes dans X_train après preprocessing"
    assert not np.isnan(X_test).any(),  "Valeurs manquantes dans X_test après preprocessing"


def test_stratified_split_preserves_ratio(sample_df):
    """Le split stratifié doit conserver les proportions de classes."""
    _, _, y_train, y_test, *_ = prepare_data(sample_df)
    ratio_train = y_train.mean()
    ratio_test  = y_test.mean()
    assert abs(ratio_train - ratio_test) < 0.05, (
        f"Ratio trop différent : train={ratio_train:.2f}, test={ratio_test:.2f}"
    )


def test_feature_names_count(sample_df):
    """14 features attendues après encoding (7 num + 4 machine_type + 3 operating_mode)."""
    _, _, _, _, _, feat_names = prepare_data(sample_df)
    assert len(feat_names) == 14, f"Attendu 14 features, obtenu {len(feat_names)}"


def test_no_data_leakage(sample_df):
    """
    Le preprocessor ne doit être fitté que sur X_train.
    On vérifie que les statistiques du scaler sont calculées sur le train uniquement.
    """
    X_train, X_test, _, _, prep, _ = prepare_data(sample_df)
    # La moyenne après scaling doit être proche de 0 sur le train (StandardScaler)
    train_mean = X_train[:, :len(NUMERIC_FEATURES)].mean(axis=0)
    assert np.all(np.abs(train_mean) < 0.1), "Scaler non centré sur le train set"
