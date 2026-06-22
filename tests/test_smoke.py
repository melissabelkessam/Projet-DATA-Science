"""
test_smoke.py — Tests de fumée : vérification des imports et de la config
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_config_imports():
    from src.config import (
        ROOT_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR,
        TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        DECISION_THRESHOLD, CV_FOLDS, RANDOM_STATE,
    )
    assert TARGET == "failure_within_24h"
    assert len(NUMERIC_FEATURES) == 7
    assert len(CATEGORICAL_FEATURES) == 2
    assert DECISION_THRESHOLD == 0.70
    assert CV_FOLDS == 5


def test_preprocessing_imports():
    from src.preprocessing import (
        load_raw_data, build_preprocessor,
        prepare_data, load_processed_data,
        save_artifacts, load_preprocessor,
    )


def test_models_imports():
    from src.models import (
        train_logistic_regression,
        train_random_forest,
        train_xgboost,
        build_mlp,
        load_model,
    )


def test_evaluation_imports():
    from src.evaluation import (
        evaluate_model, compare_models,
        cross_validate_model, optimize_threshold,
        plot_confusion_matrix, plot_model_comparison,
    )


def test_interpretability_imports():
    from src.interpretability import (
        plot_feature_importance,
        compute_shap_values,
        plot_shap_summary,
        plot_shap_bar,
        explain_single_prediction,
    )


def test_data_loader_imports():
    from src.data_loader import (
        load_and_describe,
        get_class_imbalance_ratio,
        get_missing_summary,
    )


def test_paths_exist():
    """Les dossiers principaux doivent exister."""
    from src.config import ROOT_DIR, MODELS_DIR, REPORTS_DIR
    assert ROOT_DIR.exists(), f"ROOT_DIR introuvable : {ROOT_DIR}"
    assert MODELS_DIR.exists(), f"MODELS_DIR introuvable : {MODELS_DIR}"


def test_models_files_exist():
    """Les modèles entraînés doivent être présents."""
    from src.config import MODEL_FILES
    for name, path in MODEL_FILES.items():
        assert path.exists(), f"Modèle manquant : {name} → {path}"
