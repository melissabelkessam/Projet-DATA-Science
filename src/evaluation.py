"""
evaluation.py — Métriques, comparaison des modèles, seuil de décision
Maintenance Prédictive Industrielle · EFREI 2025-26
Amelia BOUKRI & Mélissa BELKESSAM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import CV_FOLDS, RANDOM_STATE, DECISION_THRESHOLD, REPORTS_DIR, COLOR_BLUE, COLOR_RED, COLOR_GREEN, COLOR_ORANGE


# ── Évaluation d'un modèle ───────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test,
                   threshold: float = DECISION_THRESHOLD) -> dict:
    """
    Calcule les métriques clés pour un modèle.
    Utilise le seuil de décision optimisé (0.70) plutôt que 0.5 par défaut.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # MLP Keras renvoie des probabilités directement
        y_proba = model.predict(X_test, verbose=0).flatten()

    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    print(classification_report(
        y_test, y_pred,
        target_names=["Pas de panne", "Panne"],
        zero_division=0,
    ))
    print(f"  ROC-AUC : {metrics['ROC-AUC']}")
    return metrics


def compare_models(models_dict: dict, X_test, y_test,
                   threshold: float = DECISION_THRESHOLD) -> pd.DataFrame:
    """
    Évalue plusieurs modèles et retourne un DataFrame comparatif.

    Parameters
    ----------
    models_dict : {"Nom du modèle": model_object, ...}
    """
    results = {}
    for name, model in models_dict.items():
        results[name] = evaluate_model(name, model, X_test, y_test, threshold)
    df = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)
    return df


# ── Validation croisée ────────────────────────────────────────────────────────

def cross_validate_model(name: str, model, X_train, y_train,
                          n_splits: int = CV_FOLDS) -> dict:
    """
    Stratified K-Fold cross-validation.
    Préserve les proportions de classes à chaque fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    recall_scores  = cross_val_score(model, X_train, y_train, cv=skf, scoring="recall")
    f1_scores      = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")
    roc_auc_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")

    cv_results = {
        "Recall moyen":   round(recall_scores.mean(), 4),
        "Recall std":     round(recall_scores.std(),  4),
        "F1 moyen":       round(f1_scores.mean(),     4),
        "ROC-AUC moyen":  round(roc_auc_scores.mean(),4),
    }
    print(f"\n[CV {n_splits}-Fold] {name}")
    for k, v in cv_results.items():
        print(f"  {k:20s}: {v}")
    return cv_results


# ── Optimisation du seuil de décision ────────────────────────────────────────

def optimize_threshold(model, X_test, y_test, step: float = 0.01) -> tuple:
    """
    Teste plusieurs seuils et retourne (seuil_optimal, métriques_par_seuil).
    Critère : maximisation du F1-score.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test, verbose=0).flatten()

    thresholds = np.arange(0.1, 0.9, step)
    records = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        records.append({
            "threshold": round(t, 2),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "F1":        f1_score(y_test, y_pred, zero_division=0),
        })

    df_thresh = pd.DataFrame(records).set_index("threshold")
    best_threshold = df_thresh["F1"].idxmax()
    print(f"Seuil optimal (F1 max) : {best_threshold} "
          f"→ F1={df_thresh.loc[best_threshold,'F1']:.4f}, "
          f"Recall={df_thresh.loc[best_threshold,'Recall']:.4f}")
    return best_threshold, df_thresh


# ── Matrice de confusion ─────────────────────────────────────────────────────

def plot_confusion_matrix(name: str, model, X_test, y_test,
                           threshold: float = DECISION_THRESHOLD,
                           save: bool = True):
    """Affiche et sauvegarde la matrice de confusion."""
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test, verbose=0).flatten()

    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pas de panne", "Panne"])
    ax.set_yticklabels(["Pas de panne", "Panne"])
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion — {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        path = REPORTS_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Visualisation comparative ─────────────────────────────────────────────────

def plot_model_comparison(df_results: pd.DataFrame, save: bool = True):
    """Graphique comparatif des métriques clés des 4 modèles."""
    colors = [COLOR_BLUE, COLOR_RED, COLOR_GREEN, COLOR_ORANGE]
    models = list(df_results.index)
    metrics = ["Recall", "F1", "ROC-AUC"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(metrics))
    width = 0.2
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [df_results.loc[model, m] for m in metrics]
        axes[0].bar(x + i * width, vals, width, label=model, color=color)
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_title("Comparaison des métriques clés")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    roc_vals = [df_results.loc[m, "ROC-AUC"] for m in models]
    bars = axes[1].barh(models, roc_vals, color=colors)
    for bar, val in zip(bars, roc_vals):
        axes[1].text(bar.get_width() - 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", ha="right",
                     color="white", fontweight="bold")
    axes[1].set_xlim(0.9, 1.0)
    axes[1].set_title("ROC-AUC par modèle")
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("Évaluation comparative des 4 modèles", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(REPORTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
