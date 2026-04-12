# -*- coding: utf-8 -*-
"""ml_eval.py

Evaluation utilities for ML models on structured data.

Includes:
- Bootstrapped F1 CI (classification)
- Bootstrapped AUC CI (multiclass classification)
- Bootstrapped RMSE CI (regression)
- Bootstrapped R-squared CI (regression)
- Confusion matrix plot
- Multi-class ROC plot
- Regression diagnostics plot (predicted vs actual + residual histogram)
- evaluate_classifier: end-to-end classification evaluation
- evaluate_regressor: end-to-end regression evaluation
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample


# ===================================================================
# CLASSIFICATION METRICS
# ===================================================================

# -------------------------------------------------------------------
# 1. Bootstrapped F1 CI
# -------------------------------------------------------------------

def bootstrap_f1_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_iterations: int = 1000,
    average: str = "weighted",
) -> Tuple[float, float, float]:
    """
    Calculates F1 score and 95% CI using bootstrapping.

    Returns
    -------
    f1_mean, ci_lower, ci_upper
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1_scores: list[float] = []

    n = len(y_true)
    indices_all = np.arange(n)

    for _ in range(n_iterations):
        indices = resample(indices_all)
        try:
            f1 = f1_score(y_true[indices], y_pred[indices], average=average)
            f1_scores.append(f1)
        except ValueError:
            continue

    if not f1_scores:
        return np.nan, np.nan, np.nan

    f1_scores = np.asarray(f1_scores, dtype=float)
    f1_mean = float(np.mean(f1_scores))
    ci_lower = float(np.percentile(f1_scores, 2.5))
    ci_upper = float(np.percentile(f1_scores, 97.5))
    return f1_mean, ci_lower, ci_upper


# -------------------------------------------------------------------
# 2. Bootstrapped Macro-AUC CI (multiclass)
# -------------------------------------------------------------------

def bootstrap_auc_ci_multiclass(
    y_true: Sequence[int],
    y_scores: np.ndarray,
    n_iterations: int = 1000,
    average: str = "macro",
) -> Tuple[float, float, float]:
    """
    Calculates Macro-AUC and 95% CI using bootstrapping.

    Parameters
    ----------
    y_true : 1D integer labels (0..K-1 or any finite label set)
    y_scores : 2D array [n_samples, n_classes] of probabilities
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)
    auc_scores: list[float] = []

    n = len(y_true)
    indices_all = np.arange(n)

    for _ in range(n_iterations):
        indices = resample(indices_all)
        y_true_sample = y_true[indices]
        y_scores_sample = y_scores[indices]

        try:
            auc_sample = roc_auc_score(
                y_true_sample,
                y_scores_sample,
                average=average,
                multi_class="ovr",
            )
            auc_scores.append(auc_sample)
        except ValueError:
            continue

    if not auc_scores:
        return np.nan, np.nan, np.nan

    auc_scores = np.asarray(auc_scores, dtype=float)
    auc_mean = float(np.mean(auc_scores))
    ci_lower = float(np.percentile(auc_scores, 2.5))
    ci_upper = float(np.percentile(auc_scores, 97.5))
    return auc_mean, ci_lower, ci_upper


# ===================================================================
# REGRESSION METRICS
# ===================================================================

# -------------------------------------------------------------------
# 3. Bootstrapped RMSE CI
# -------------------------------------------------------------------

def bootstrap_rmse_ci(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_iterations: int = 1000,
) -> Tuple[float, float, float]:
    """
    Calculates RMSE and 95% CI using bootstrapping.

    Returns
    -------
    rmse_mean, ci_lower, ci_upper
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse_scores: list[float] = []

    n = len(y_true)
    indices_all = np.arange(n)

    for _ in range(n_iterations):
        indices = resample(indices_all)
        try:
            rmse = np.sqrt(mean_squared_error(y_true[indices], y_pred[indices]))
            rmse_scores.append(rmse)
        except ValueError:
            continue

    if not rmse_scores:
        return np.nan, np.nan, np.nan

    rmse_scores = np.asarray(rmse_scores, dtype=float)
    rmse_mean = float(np.mean(rmse_scores))
    ci_lower = float(np.percentile(rmse_scores, 2.5))
    ci_upper = float(np.percentile(rmse_scores, 97.5))
    return rmse_mean, ci_lower, ci_upper


# -------------------------------------------------------------------
# 4. Bootstrapped R-squared CI
# -------------------------------------------------------------------

def bootstrap_r2_ci(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_iterations: int = 1000,
) -> Tuple[float, float, float]:
    """
    Calculates R-squared and 95% CI using bootstrapping.

    Returns
    -------
    r2_mean, ci_lower, ci_upper
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2_scores: list[float] = []

    n = len(y_true)
    indices_all = np.arange(n)

    for _ in range(n_iterations):
        indices = resample(indices_all)
        try:
            r2 = r2_score(y_true[indices], y_pred[indices])
            r2_scores.append(r2)
        except ValueError:
            continue

    if not r2_scores:
        return np.nan, np.nan, np.nan

    r2_scores = np.asarray(r2_scores, dtype=float)
    r2_mean = float(np.mean(r2_scores))
    ci_lower = float(np.percentile(r2_scores, 2.5))
    ci_upper = float(np.percentile(r2_scores, 97.5))
    return r2_mean, ci_lower, ci_upper


def bootstrap_mae_ci(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_iterations: int = 1000,
) -> Tuple[float, float, float]:
    """
    Calculates MAE and 95% CI using bootstrapping.

    Returns
    -------
    mae_mean, ci_lower, ci_upper
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae_scores: list[float] = []

    n = len(y_true)
    indices_all = np.arange(n)

    for _ in range(n_iterations):
        indices = resample(indices_all)
        try:
            mae = mean_absolute_error(y_true[indices], y_pred[indices])
            mae_scores.append(mae)
        except ValueError:
            continue

    if not mae_scores:
        return np.nan, np.nan, np.nan

    mae_scores = np.asarray(mae_scores, dtype=float)
    mae_mean = float(np.mean(mae_scores))
    ci_lower = float(np.percentile(mae_scores, 2.5))
    ci_upper = float(np.percentile(mae_scores, 97.5))
    return mae_mean, ci_lower, ci_upper


# ===================================================================
# PLOTS
# ===================================================================

# -------------------------------------------------------------------
# 5. Confusion matrix plot
# -------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence,
    model_name: str = "Model",
    save_dir: Optional[str] = None,
    cmap_name: str = "Blues",
) -> None:
    """
    Plots a confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : label arrays
    labels : sequence of labels in the order to display
    model_name : used in title and filename
    save_dir : if provided, saves PNG there
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap=cmap_name,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=conf_matrix.max(),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        print(f"Saved confusion matrix to: {cm_path}")

    plt.show()


# -------------------------------------------------------------------
# 6. Multi-class ROC plot
# -------------------------------------------------------------------

def plot_multiclass_roc(
    y_true: Sequence[int],
    y_score: np.ndarray,
    classes_list: Sequence,
    label_prefix: str = "Model",
    save_dir: Optional[str] = None,
) -> None:
    """
    Plots per-class ROC curves for a multiclass classifier.

    Parameters
    ----------
    y_true : 1D labels
    y_score : 2D array [n_samples, n_classes] of probabilities
    classes_list : class labels in the same order as columns in y_score
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    # `classes_list` MUST be aligned with the column order of y_score
    # (typically `model.classes_`). If callers pass an arbitrary class
    # ordering, per-class ROC curves get silently mis-labelled.
    if y_score.ndim != 2:
        raise ValueError(
            f"y_score must be 2D [n_samples, n_classes], got shape {y_score.shape}"
        )
    if y_score.shape[1] != len(classes_list):
        raise ValueError(
            f"y_score has {y_score.shape[1]} columns but classes_list has "
            f"{len(classes_list)} entries — these must match. Pass "
            f"classes_list=model.classes_ to guarantee correct alignment."
        )

    y_true_bin = label_binarize(y_true, classes=classes_list)
    n_classes = len(classes_list)

    # For binary targets, label_binarize returns shape (N, 1) — expand to (N, 2)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr: Dict[int, np.ndarray] = {}
    tpr: Dict[int, np.ndarray] = {}
    roc_auc_vals: Dict[int, float] = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc_vals[i] = auc(fpr[i], tpr[i])

    colors = sns.color_palette("mako", n_classes)

    plt.figure(figsize=(10, 8))
    for i, (name, color) in enumerate(zip(classes_list, colors)):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=f"Class {name} (AUC = {roc_auc_vals[i]:.2f})",
            color=color,
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{label_prefix} - Multi-class ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        roc_path = os.path.join(save_dir, f"{label_prefix}_roc_curve.png")
        plt.savefig(roc_path, dpi=300)
        print(f"Saved ROC curve to: {roc_path}")

    plt.show()


# -------------------------------------------------------------------
# 7. Regression diagnostics plot
# -------------------------------------------------------------------

def plot_regression_diagnostics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    model_name: str = "Model",
    save_dir: Optional[str] = None,
) -> None:
    """
    Plots regression diagnostics:
    - Left panel: Predicted vs Actual scatter with identity line
    - Right panel: Residual histogram

    Parameters
    ----------
    y_true, y_pred : arrays of actual and predicted values
    model_name : used in title and filename
    save_dir : if provided, saves PNG there
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_style("whitegrid")

    # Predicted vs Actual
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Identity")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title(f"{model_name} - Predicted vs Actual")
    ax1.legend()

    # Residual histogram
    ax2 = axes[1]
    ax2.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Residual (Actual - Predicted)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"{model_name} - Residual Distribution")

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        diag_path = os.path.join(save_dir, f"{model_name}_regression_diagnostics.png")
        plt.savefig(diag_path, dpi=300)
        print(f"Saved regression diagnostics to: {diag_path}")

    plt.show()


# ===================================================================
# END-TO-END EVALUATION WRAPPERS
# ===================================================================

# -------------------------------------------------------------------
# 8. Classification evaluation
# -------------------------------------------------------------------

def evaluate_classifier(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model",
    label_names: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    compute_bootstrap_ci: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end evaluation for a classifier.

    Parameters
    ----------
    y_true : 1D array-like of true labels
    y_pred : 1D array-like of predicted labels
    y_proba : 2D array [n_samples, n_classes] of predicted probabilities (or None)
    model_name : str for printouts and plot titles
    label_names : list of class name strings
    output_dir : directory to save plots
    compute_bootstrap_ci : whether to compute bootstrap CIs

    Returns
    -------
    metrics : dict with accuracy, F1 scores, optional CIs, optional AUC
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if label_names is None:
        unique_labels = np.unique(y_true)
        label_names = [str(l) for l in unique_labels]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report_dict,
    }

    # Bootstrap F1 CI
    if compute_bootstrap_ci:
        f1_mean_bs, f1_ci_low, f1_ci_high = bootstrap_f1_ci(
            y_true, y_pred, average="weighted"
        )
        metrics["f1_bs_mean"] = f1_mean_bs
        metrics["f1_bs_ci"] = (f1_ci_low, f1_ci_high)

    # Print report
    print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy      : {acc:.4f}")
    print(f"F1 (macro)    : {f1_macro:.4f}")
    print(f"F1 (weighted) : {f1_weighted:.4f}")
    if compute_bootstrap_ci:
        print(
            f"Weighted F1 (bootstrapped): {f1_mean_bs:.4f} "
            f"[{f1_ci_low:.4f}, {f1_ci_high:.4f}]"
        )
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            zero_division=0,
        )
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=np.unique(y_true),
        model_name=model_name,
        save_dir=output_dir,
    )

    # AUC + ROC (only if probabilities are provided)
    if y_proba is not None:
        y_scores = np.asarray(y_proba, dtype=float)
        classes_list = np.unique(y_true)

        if compute_bootstrap_ci:
            auc_mean, auc_ci_low, auc_ci_high = bootstrap_auc_ci_multiclass(
                y_true, y_scores
            )
            metrics["auc_macro"] = auc_mean
            metrics["auc_ci"] = (auc_ci_low, auc_ci_high)

            print(f"\nMacro-Averaged AUC: {auc_mean:.4f}")
            print(f"95% CI for Macro-Averaged AUC: [{auc_ci_low:.4f}, {auc_ci_high:.4f}]")

        plot_multiclass_roc(
            y_true,
            y_scores,
            classes_list=classes_list,
            label_prefix=model_name,
            save_dir=output_dir,
        )

    return metrics


# -------------------------------------------------------------------
# 9. Regression evaluation
# -------------------------------------------------------------------

def evaluate_regressor(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    model_name: str = "Model",
    output_dir: Optional[str] = None,
    compute_bootstrap_ci: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end evaluation for a regressor.

    Parameters
    ----------
    y_true : 1D array-like of actual values
    y_pred : 1D array-like of predicted values
    model_name : str for printouts and plot titles
    output_dir : directory to save plots
    compute_bootstrap_ci : whether to compute bootstrap CIs

    Returns
    -------
    metrics : dict with RMSE, R-squared, MAE, optional CIs
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
    }

    print(f"\n=== {model_name} Regression Evaluation ===")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")

    # Bootstrap CIs
    if compute_bootstrap_ci:
        rmse_mean, rmse_ci_low, rmse_ci_high = bootstrap_rmse_ci(y_true, y_pred)
        r2_mean, r2_ci_low, r2_ci_high = bootstrap_r2_ci(y_true, y_pred)
        mae_mean, mae_ci_low, mae_ci_high = bootstrap_mae_ci(y_true, y_pred)

        metrics["rmse_bs_mean"] = rmse_mean
        metrics["rmse_bs_ci"] = (rmse_ci_low, rmse_ci_high)
        metrics["r2_bs_mean"] = r2_mean
        metrics["r2_bs_ci"] = (r2_ci_low, r2_ci_high)
        metrics["mae_bs_mean"] = mae_mean
        metrics["mae_bs_ci"] = (mae_ci_low, mae_ci_high)

        print(f"\nRMSE (bootstrapped): {rmse_mean:.4f} [{rmse_ci_low:.4f}, {rmse_ci_high:.4f}]")
        print(f"R2   (bootstrapped): {r2_mean:.4f} [{r2_ci_low:.4f}, {r2_ci_high:.4f}]")
        print(f"MAE  (bootstrapped): {mae_mean:.4f} [{mae_ci_low:.4f}, {mae_ci_high:.4f}]")

    # Regression diagnostics plot
    plot_regression_diagnostics(
        y_true=y_true,
        y_pred=y_pred,
        model_name=model_name,
        save_dir=output_dir,
    )

    return metrics
