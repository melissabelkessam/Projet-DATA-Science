"""
interpretability.py — Feature Importance et analyse SHAP
Maintenance Prédictive Industrielle · EFREI 2025-26
Amelia BOUKRI & Mélissa BELKESSAM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.config import REPORTS_DIR, COLOR_BLUE, COLOR_RED


# ── Feature Importance native (modèles à base d'arbres) ─────────────────────

def plot_feature_importance(model, feature_names: list,
                             model_name: str = "XGBoost",
                             top_n: int = 15,
                             save: bool = True):
    """
    Affiche l'importance des variables (réduction d'impureté Gini).
    Compatible avec Random Forest et XGBoost.
    """
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color=COLOR_BLUE,
        alpha=0.85,
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name} (réduction d'impureté Gini)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save:
        path = REPORTS_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()

    # Tableau des top variables
    df_imp = pd.DataFrame({
        "Variable":   [feature_names[i] for i in idx[::-1]],
        "Importance": [round(importances[i], 4) for i in idx[::-1]],
    })
    return df_imp


# ── Analyse SHAP ─────────────────────────────────────────────────────────────

def compute_shap_values(model, X_sample: np.ndarray, feature_names: list,
                         n_samples: int = 500):
    """
    Calcule les valeurs SHAP sur un sous-échantillon.
    Utilise TreeExplainer pour XGBoost / Random Forest.

    Retourne : (shap_values, explainer)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Installez shap : pip install shap")

    # Sous-échantillon pour accélérer le calcul
    idx = np.random.choice(len(X_sample), size=min(n_samples, len(X_sample)), replace=False)
    X_sub = X_sample[idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    # XGBoost renvoie un array 2D directement ; RF renvoie une liste [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, explainer, X_sub


def plot_shap_summary(shap_values, X_sub: np.ndarray,
                       feature_names: list, save: bool = True):
    """SHAP summary plot (beeswarm) — importance globale des features."""
    import shap
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sub,
        feature_names=feature_names,
        show=False,
    )
    plt.title("SHAP — Impact global des features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_bar(shap_values, X_sub: np.ndarray,
                   feature_names: list, save: bool = True):
    """SHAP bar plot — importance moyenne absolue (globale)."""
    import shap
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in idx], mean_abs[idx], color=COLOR_BLUE, alpha=0.85)
    ax.set_xlabel("|SHAP value| moyen")
    ax.set_title("Interpretabilité SHAP — XGBoost")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(REPORTS_DIR / "shap_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()


def explain_single_prediction(model, X_single: np.ndarray,
                                feature_names: list,
                                explainer=None) -> dict:
    """
    Explique une prédiction individuelle via SHAP.
    Retourne un dict {feature: shap_value} trié par impact.
    """
    import shap
    if explainer is None:
        explainer = shap.TreeExplainer(model)

    sv = explainer.shap_values(X_single.reshape(1, -1))
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv.flatten()

    explanation = dict(zip(feature_names, sv))
    explanation = dict(sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True))
    return explanation
