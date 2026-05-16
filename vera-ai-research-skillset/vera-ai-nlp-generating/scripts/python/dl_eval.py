# -*- coding: utf-8 -*-
"""dl_eval.py

Evaluation utilities for deep learning models (PyTorch)
that output logits (pre-softmax scores).

Includes:
- Bootstrapped F1 and AUC CIs
- Confusion matrix + ROC plots
- End-to-end evaluation from logits
"""

from __future__ import annotations

from typing import Optional, Sequence, Dict, Any, Tuple

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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
    """Compute F1 score and 95% CI via bootstrapping.

    Parameters
    ----------
    y_true, y_pred : sequences of labels
    n_iterations : number of bootstrap samples
    average : F1 averaging scheme ("macro", "weighted", etc.)

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
            # e.g. only one class present in this bootstrap sample
            continue

    if not f1_scores:
        return np.nan, np.nan, np.nan

    f1_scores = np.asarray(f1_scores, dtype=float)
    f1_mean = float(f1_scores.mean())
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
    """Compute macro AUC and 95% CI via bootstrapping (multiclass).

    Parameters
    ----------
    y_true : 1D integer labels (0..K-1 or any finite label set)
    y_scores : 2D array [n_samples, n_classes] of probabilities

    Returns
    -------
    auc_mean, ci_lower, ci_upper
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
            # e.g. one class missing in this bootstrap sample
            continue

    if not auc_scores:
        return np.nan, np.nan, np.nan

    auc_scores = np.asarray(auc_scores, dtype=float)
    auc_mean = float(auc_scores.mean())
    ci_lower = float(np.percentile(auc_scores, 2.5))
    ci_upper = float(np.percentile(auc_scores, 97.5))
    return auc_mean, ci_lower, ci_upper


# -------------------------------------------------------------------
# 3. Confusion matrix plot
# -------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence,
    model_name: str = "Model",
    save_dir: Optional[str] = None,
    cmap_name: str = "Blues",
) -> None:
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : label arrays (same type as labels elements)
    labels : sequence of labels in the order to display
    model_name : used in title and filename
    save_dir : if provided, save PNG there
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

    plt.close()


# -------------------------------------------------------------------
# 4. Multi-class ROC plot
# -------------------------------------------------------------------

def plot_multiclass_roc(
    y_true: Sequence[int],
    y_score: np.ndarray,
    classes_list: Sequence,
    label_prefix: str = "Model",
    save_dir: Optional[str] = None,
) -> None:
    """Plot per-class ROC curves for a multiclass classifier.

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

    # Binarize labels for ROC computation
    y_true_bin = label_binarize(y_true, classes=classes_list)
    n_classes = y_true_bin.shape[1]

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

    plt.close()


# -------------------------------------------------------------------
# 5. Logits → probabilities
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
# 6. End-to-end evaluation from logits
# -------------------------------------------------------------------

def evaluate_dl_from_logits(
    y_true: Sequence[int],
    logits,
    classes_list: Optional[Sequence[int]] = None,
    model_name: str = "DL model",
    output_dir: Optional[str] = None,
    save_dir: Optional[str] = None,   # alias for backward compatibility
    compute_bootstrap_ci: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate a deep learning model given true labels and logits.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Encoded integer labels.
    logits : torch.Tensor or np.ndarray of shape (n_samples, n_classes)
        Raw output scores from the DL model.
    classes_list : sequence of class labels (same order as columns in logits).
        If None, inferred from y_true.
    model_name : str
        Used in prints and plot titles.
    output_dir : str or Path or None
        Directory to save confusion matrix / ROC figures.
    save_dir : str or Path or None
        Deprecated alias for output_dir (kept for backward compatibility).
    compute_bootstrap_ci : bool
        Whether to compute bootstrap CIs for F1 and AUC.

    Returns
    -------
    f1_stats : dict
        F1-related metrics and classification report.
    auc_stats : dict
        AUC-related metrics (if probabilities available).
    """
    # Resolve directory alias
    out_dir = output_dir if output_dir is not None else save_dir

    # ---------- Prepare ----------
    y_true = np.asarray(y_true)
    probs = _logits_to_probs(logits)    # [n_samples, n_classes]
    y_pred = probs.argmax(axis=1)

    # Infer classes if not provided
    if classes_list is None:
        classes_list = np.unique(y_true)
    classes_list = np.asarray(classes_list)

    # Build class_names as strings for reports
    class_names = [str(c) for c in classes_list]

    # ---------- Basic metrics ----------
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names[: len(np.unique(y_true))],
        output_dict=True,
        zero_division=0,
    )

    f1_stats: Dict[str, Any] = {
        "model_name": model_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report_dict,
    }

    # ---------- Bootstrap F1 CI ----------
    if compute_bootstrap_ci:
        f1_mean_bs, f1_ci_low, f1_ci_high = bootstrap_f1_ci(
            y_true, y_pred, average="weighted"
        )
        f1_stats["f1_bs_mean"] = f1_mean_bs
        f1_stats["f1_bs_ci"] = (f1_ci_low, f1_ci_high)
    else:
        f1_mean_bs = f1_ci_low = f1_ci_high = None

    # ---------- Confusion Matrix ----------
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=classes_list,
        model_name=model_name,
        save_dir=out_dir,
    )

    # ---------- AUC + ROC curves ----------
    auc_stats: Dict[str, Any] = {"model_name": model_name}
    if compute_bootstrap_ci:
        auc_mean, auc_ci_low, auc_ci_high = bootstrap_auc_ci_multiclass(
            y_true, probs
        )
        auc_stats["auc_macro"] = auc_mean
        auc_stats["auc_ci"] = (auc_ci_low, auc_ci_high)
    else:
        auc_mean = auc_ci_low = auc_ci_high = None

    # ROC curves (per class)
    plot_multiclass_roc(
        y_true,
        probs,
        classes_list=classes_list,
        label_prefix=model_name,
        save_dir=out_dir,
    )

    # ---------- Pretty print ----------
    print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy      : {acc:.4f}")
    print(f"F1 (macro)    : {f1_macro:.4f}")
    print(f"F1 (weighted) : {f1_weighted:.4f}")
    if compute_bootstrap_ci and f1_mean_bs is not None:
        print(
            f"Weighted F1 (bootstrapped): {f1_mean_bs:.4f} "
            f"[{f1_ci_low:.4f}, {f1_ci_high:.4f}]"
        )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names[: len(np.unique(y_true))],
            zero_division=0,
        )
    )

    if compute_bootstrap_ci and auc_mean is not None:
        print(f"\nMacro-Averaged AUC: {auc_mean:.4f}")
        print(
            f"95% CI for Macro-Averaged AUC: "
            f"[{auc_ci_low:.4f}, {auc_ci_high:.4f}]"
        )

    return f1_stats, auc_stats