"""
dl_eval.py
==========
Evaluation utilities for image classification models.

Provides softmax conversion, bootstrap confidence intervals for F1 and
multiclass AUC, confusion matrix plotting, multiclass ROC plotting,
and a unified evaluation pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


# ===================================================================== #
#  Softmax helper                                                        #
# ===================================================================== #
def _logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Convert raw logits to probabilities via numerically-stable softmax.

    Parameters
    ----------
    logits : np.ndarray
        Array of shape ``(N, C)``.

    Returns
    -------
    np.ndarray
        Probability array of the same shape, rows summing to 1.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


# ===================================================================== #
#  Bootstrap CI for F1                                                   #
# ===================================================================== #
def bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    average: str = "weighted",
    random_state: int = 2025,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for the F1 score.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    n_iterations : int
        Number of bootstrap resamples.
    average : str
        Averaging strategy passed to :func:`f1_score`.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    mean_f1, ci_low, ci_high : float
        Mean F1 and 95 % confidence bounds.
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    scores: List[float] = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        score = f1_score(y_true[idx], y_pred[idx], average=average, zero_division=0)
        scores.append(float(score))

    scores_arr = np.array(scores)
    mean_f1 = float(scores_arr.mean())
    ci_low = float(np.percentile(scores_arr, 2.5))
    ci_high = float(np.percentile(scores_arr, 97.5))
    return mean_f1, ci_low, ci_high


# ===================================================================== #
#  Bootstrap CI for multiclass AUC                                       #
# ===================================================================== #
def bootstrap_auc_ci_multiclass(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_iterations: int = 1000,
    random_state: int = 2025,
) -> Tuple[float, float, float]:
    """Compute bootstrap CI for macro-averaged one-vs-rest AUC.

    Parameters
    ----------
    y_true : np.ndarray  (N,)
        True integer labels.
    y_scores : np.ndarray  (N, C)
        Probability scores (softmax output).
    n_iterations : int
    random_state : int

    Returns
    -------
    mean_auc, ci_low, ci_high : float
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)
    classes = np.unique(y_true)
    n_classes = len(classes)

    y_true_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    scores: List[float] = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        yt = y_true_bin[idx]
        ys = y_scores[idx]

        per_class_auc: List[float] = []
        for c in range(n_classes):
            if len(np.unique(yt[:, c])) < 2:
                continue
            fpr, tpr, _ = roc_curve(yt[:, c], ys[:, c])
            per_class_auc.append(float(auc(fpr, tpr)))

        if per_class_auc:
            scores.append(float(np.mean(per_class_auc)))

    if not scores:
        return 0.0, 0.0, 0.0

    scores_arr = np.array(scores)
    mean_auc_val = float(scores_arr.mean())
    ci_low = float(np.percentile(scores_arr, 2.5))
    ci_high = float(np.percentile(scores_arr, 97.5))
    return mean_auc_val, ci_low, ci_high


# ===================================================================== #
#  Confusion matrix plot                                                 #
# ===================================================================== #
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    model_name: str,
    save_dir: str,
) -> str:
    """Plot and save a confusion-matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like
    class_names : list of str
    model_name : str
        Used in the title and file name.
    save_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)

    fig, ax = plt.subplots(
        figsize=(max(6, n_classes * 0.8), max(5, n_classes * 0.7))
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    fpath = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


# ===================================================================== #
#  Multiclass ROC plot                                                   #
# ===================================================================== #
def plot_multiclass_roc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: Sequence[str],
    model_name: str,
    save_dir: str,
) -> str:
    """Plot and save per-class ROC curves (one-vs-rest).

    Parameters
    ----------
    y_true : np.ndarray  (N,)
    y_scores : np.ndarray  (N, C)
    class_names : list of str
    model_name, save_dir : str

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    os.makedirs(save_dir, exist_ok=True)
    classes = np.unique(y_true)
    n_classes = len(classes)

    y_true_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        label = f"{class_names[i]} (AUC = {roc_auc:.3f})"
        ax.plot(fpr, tpr, linewidth=1.5, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Multiclass ROC - {model_name}")
    ax.legend(loc="lower right", fontsize="small")
    plt.tight_layout()

    fpath = os.path.join(save_dir, f"roc_multiclass_{model_name}.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


# ===================================================================== #
#  Unified evaluation pipeline                                           #
# ===================================================================== #
def evaluate_image_model(
    y_true: np.ndarray,
    logits: np.ndarray,
    class_names: Sequence[str],
    model_name: str,
    output_dir: str,
    compute_bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """End-to-end evaluation pipeline for an image classification model.

    Steps:
      1. Convert logits to probabilities (softmax).
      2. Derive predicted labels.
      3. Compute accuracy, weighted F1, classification report.
      4. Optionally compute bootstrap 95 % CIs for F1 and AUC.
      5. Plot confusion matrix and multiclass ROC curves.

    Parameters
    ----------
    y_true : np.ndarray  (N,)
    logits : np.ndarray   (N, C)
    class_names : list of str
    model_name : str
    output_dir : str
    compute_bootstrap_ci : bool
    n_bootstrap : int

    Returns
    -------
    dict
        ``accuracy``, ``f1_weighted``, ``f1_ci`` (tuple or None),
        ``auc_macro``, ``auc_ci`` (tuple or None),
        ``classification_report`` (str),
        ``confusion_matrix_path`` (str),
        ``roc_path`` (str).
    """
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.asarray(y_true)
    logits = np.asarray(logits)
    y_proba = _logits_to_probs(logits)
    y_pred = y_proba.argmax(axis=1)

    # --- scalar metrics ------------------------------------------------ #
    acc = float(accuracy_score(y_true, y_pred))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        zero_division=0,
    )

    # --- bootstrap CIs ------------------------------------------------- #
    f1_ci: Optional[Tuple[float, float, float]] = None
    auc_ci: Optional[Tuple[float, float, float]] = None
    auc_macro: Optional[float] = None

    if compute_bootstrap_ci:
        f1_ci = bootstrap_f1_ci(
            y_true, y_pred, n_iterations=n_bootstrap
        )
        auc_ci = bootstrap_auc_ci_multiclass(
            y_true, y_proba, n_iterations=n_bootstrap
        )
        auc_macro = auc_ci[0]

    # --- plots --------------------------------------------------------- #
    cm_path = plot_confusion_matrix(
        y_true, y_pred, class_names, model_name, output_dir
    )
    roc_path = plot_multiclass_roc(
        y_true, y_proba, class_names, model_name, output_dir
    )

    # --- summary ------------------------------------------------------- #
    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_ci": f1_ci,
        "auc_macro": auc_macro,
        "auc_ci": auc_ci,
        "classification_report": cls_report,
        "confusion_matrix_path": cm_path,
        "roc_path": roc_path,
    }

    # Print a brief summary
    print(f"\n{'=' * 50}")
    print(f"  {model_name} — Evaluation Summary")
    print(f"{'=' * 50}")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  F1 (weighted):  {f1_w:.4f}")
    if f1_ci is not None:
        print(f"  F1 95% CI:      [{f1_ci[1]:.4f}, {f1_ci[2]:.4f}]")
    if auc_macro is not None:
        print(f"  AUC (macro):    {auc_macro:.4f}")
    if auc_ci is not None:
        print(f"  AUC 95% CI:     [{auc_ci[1]:.4f}, {auc_ci[2]:.4f}]")
    print(f"{'=' * 50}\n")

    return metrics
