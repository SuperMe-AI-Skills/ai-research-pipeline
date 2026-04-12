# -*- coding: utf-8 -*-
"""dl_eval.py

Evaluation utilities for deep learning models (PyTorch)
on structured/tabular data.

Includes:
- Logits to probabilities conversion
- End-to-end evaluation from logits (classification)
- Regression evaluation from predictions

Reuses bootstrapped CI functions and plotting from ml_eval.py
via inline implementations to keep this module self-contained.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
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


# -------------------------------------------------------------------
# 1. Bootstrapped F1 CI
# -------------------------------------------------------------------

def bootstrap_f1_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_iterations: int = 1000,
    average: str = "weighted",
) -> Tuple[float, float, float]:
    """Compute F1 score and 95% CI via bootstrapping."""
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
    return float(f1_scores.mean()), float(np.percentile(f1_scores, 2.5)), float(np.percentile(f1_scores, 97.5))


# -------------------------------------------------------------------
# 2. Bootstrapped Macro-AUC CI (multiclass)
# -------------------------------------------------------------------

def bootstrap_auc_ci_multiclass(
    y_true: Sequence[int],
    y_scores: np.ndarray,
    n_iterations: int = 1000,
    average: str = "macro",
) -> Tuple[float, float, float]:
    """Compute macro AUC and 95% CI via bootstrapping."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    auc_scores: list[float] = []
    n = len(y_true)
    indices_all = np.arange(n)

    for _ in range(n_iterations):
        indices = resample(indices_all)
        try:
            auc_sample = roc_auc_score(
                y_true[indices], y_scores[indices],
                average=average, multi_class="ovr",
            )
            auc_scores.append(auc_sample)
        except ValueError:
            continue

    if not auc_scores:
        return np.nan, np.nan, np.nan

    auc_scores = np.asarray(auc_scores, dtype=float)
    return float(auc_scores.mean()), float(np.percentile(auc_scores, 2.5)), float(np.percentile(auc_scores, 97.5))


# -------------------------------------------------------------------
# 3. Plots
# -------------------------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, labels, model_name, save_dir=None):
    """Plot a confusion matrix heatmap."""
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=conf_matrix.max())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.show()


def _plot_multiclass_roc(y_true, y_score, classes_list, label_prefix, save_dir=None):
    """Plot per-class ROC curves."""
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
            f"{len(classes_list)} entries — these must match."
        )

    y_true_bin = label_binarize(y_true, classes=classes_list)
    n_classes = len(classes_list)

    # For binary targets, label_binarize returns shape (N, 1) — expand to (N, 2)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr, tpr, roc_auc_vals = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc_vals[i] = auc(fpr[i], tpr[i])

    colors = sns.color_palette("mako", n_classes)
    plt.figure(figsize=(10, 8))
    for i, (name, color) in enumerate(zip(classes_list, colors)):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f"Class {name} (AUC = {roc_auc_vals[i]:.2f})", color=color)
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{label_prefix} - Multi-class ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{label_prefix}_roc_curve.png"), dpi=300)
    plt.show()


# -------------------------------------------------------------------
# 4. Logits -> probabilities
# -------------------------------------------------------------------

def _logits_to_probs(logits) -> np.ndarray:
    """Convert logits (torch.Tensor or np.ndarray) to softmax probabilities."""
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = np.asarray(logits, dtype=float)

    # Stable softmax along last axis
    logits_np = logits_np - logits_np.max(axis=1, keepdims=True)
    exp = np.exp(logits_np)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs


# -------------------------------------------------------------------
# 5. End-to-end evaluation from logits (classification)
# -------------------------------------------------------------------

def evaluate_dl_from_logits(
    y_true: Sequence[int],
    logits,
    classes_list: Optional[Sequence[int]] = None,
    model_name: str = "DL model",
    output_dir: Optional[str] = None,
    compute_bootstrap_ci: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate a deep learning classifier given true labels and logits.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    logits : torch.Tensor or np.ndarray of shape (n_samples, n_classes)
    classes_list : class labels. If None, inferred from y_true.
    model_name : str
    output_dir : directory to save plots
    compute_bootstrap_ci : whether to compute bootstrap CIs

    Returns
    -------
    f1_stats : dict with F1-related metrics
    auc_stats : dict with AUC-related metrics
    """
    y_true = np.asarray(y_true)
    probs = _logits_to_probs(logits)
    y_pred = probs.argmax(axis=1)

    if classes_list is None:
        classes_list = np.unique(y_true)
    classes_list = np.asarray(classes_list)
    class_names = [str(c) for c in classes_list]

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names[:len(np.unique(y_true))],
        output_dict=True, zero_division=0,
    )

    f1_stats: Dict[str, Any] = {
        "model_name": model_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report_dict,
    }

    # Bootstrap F1 CI
    if compute_bootstrap_ci:
        f1_mean_bs, f1_ci_low, f1_ci_high = bootstrap_f1_ci(y_true, y_pred, average="weighted")
        f1_stats["f1_bs_mean"] = f1_mean_bs
        f1_stats["f1_bs_ci"] = (f1_ci_low, f1_ci_high)
    else:
        f1_mean_bs = f1_ci_low = f1_ci_high = None

    # Confusion Matrix
    _plot_confusion_matrix(y_true, y_pred, labels=classes_list,
                           model_name=model_name, save_dir=output_dir)

    # AUC + ROC
    auc_stats: Dict[str, Any] = {"model_name": model_name}
    if compute_bootstrap_ci:
        auc_mean, auc_ci_low, auc_ci_high = bootstrap_auc_ci_multiclass(y_true, probs)
        auc_stats["auc_macro"] = auc_mean
        auc_stats["auc_ci"] = (auc_ci_low, auc_ci_high)
    else:
        auc_mean = auc_ci_low = auc_ci_high = None

    _plot_multiclass_roc(y_true, probs, classes_list=classes_list,
                         label_prefix=model_name, save_dir=output_dir)

    # Print
    print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy      : {acc:.4f}")
    print(f"F1 (macro)    : {f1_macro:.4f}")
    print(f"F1 (weighted) : {f1_weighted:.4f}")
    if compute_bootstrap_ci and f1_mean_bs is not None:
        print(f"Weighted F1 (bootstrapped): {f1_mean_bs:.4f} [{f1_ci_low:.4f}, {f1_ci_high:.4f}]")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred,
          target_names=class_names[:len(np.unique(y_true))], zero_division=0))
    if compute_bootstrap_ci and auc_mean is not None:
        print(f"\nMacro-Averaged AUC: {auc_mean:.4f}")
        print(f"95% CI for Macro-Averaged AUC: [{auc_ci_low:.4f}, {auc_ci_high:.4f}]")

    return f1_stats, auc_stats


# -------------------------------------------------------------------
# 6. Regression evaluation from predictions
# -------------------------------------------------------------------

def evaluate_dl_regression(
    y_true: Sequence[float],
    predictions,
    model_name: str = "DL model",
    output_dir: Optional[str] = None,
    compute_bootstrap_ci: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a deep learning regressor given true values and predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    predictions : torch.Tensor or np.ndarray of shape (n_samples,)
    model_name : str
    output_dir : directory to save plots
    compute_bootstrap_ci : whether to compute bootstrap CIs

    Returns
    -------
    metrics : dict with RMSE, R2, MAE, optional CIs
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)

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
        n = len(y_true)
        indices_all = np.arange(n)

        rmse_scores = []
        r2_scores = []
        for _ in range(1000):
            idx = resample(indices_all)
            try:
                rmse_scores.append(np.sqrt(mean_squared_error(y_true[idx], y_pred[idx])))
                r2_scores.append(r2_score(y_true[idx], y_pred[idx]))
            except ValueError:
                continue

        if rmse_scores:
            rmse_arr = np.asarray(rmse_scores)
            r2_arr = np.asarray(r2_scores)
            metrics["rmse_bs_mean"] = float(rmse_arr.mean())
            metrics["rmse_bs_ci"] = (float(np.percentile(rmse_arr, 2.5)), float(np.percentile(rmse_arr, 97.5)))
            metrics["r2_bs_mean"] = float(r2_arr.mean())
            metrics["r2_bs_ci"] = (float(np.percentile(r2_arr, 2.5)), float(np.percentile(r2_arr, 97.5)))

            print(f"\nRMSE (bootstrapped): {metrics['rmse_bs_mean']:.4f} "
                  f"[{metrics['rmse_bs_ci'][0]:.4f}, {metrics['rmse_bs_ci'][1]:.4f}]")
            print(f"R2   (bootstrapped): {metrics['r2_bs_mean']:.4f} "
                  f"[{metrics['r2_bs_ci'][0]:.4f}, {metrics['r2_bs_ci'][1]:.4f}]")

    # Regression diagnostics plot
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_style("whitegrid")

    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Identity")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title(f"{model_name} - Predicted vs Actual")
    ax1.legend()

    ax2 = axes[1]
    ax2.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Residual (Actual - Predicted)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"{model_name} - Residual Distribution")

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{model_name}_regression_diagnostics.png"), dpi=300)
    plt.show()

    return metrics
