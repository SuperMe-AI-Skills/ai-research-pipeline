# -*- coding: utf-8 -*-
"""ml_eval.py

Evaluation utilities for multiclass ML models.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
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
            # e.g. only one class in the bootstrap sample
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
            # e.g. one class missing in this bootstrap sample
            continue

    if not auc_scores:
        return np.nan, np.nan, np.nan

    auc_scores = np.asarray(auc_scores, dtype=float)
    auc_mean = float(np.mean(auc_scores))
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
    """
    Plots a confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : label arrays (same type as labels elements)
    labels : sequence of labels in the order you want them on axes
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

    # Binarize labels for ROC computation
    y_true_bin = label_binarize(y_true, classes=classes_list)
    n_classes = len(classes_list)

    # For binary targets, label_binarize returns shape (N, 1) — expand to (N, 2)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr = {}
    tpr = {}
    roc_auc_vals = {}

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
# 5. Main wrapper for scikit-learn models (model + X)
# -------------------------------------------------------------------

def evaluate_ml_multiclass(
    model,
    X_test,
    y_test: Sequence[int],
    model_name: str = "Model",
    save_dir: Optional[str] = None,
) -> None:
    """
    Full evaluation pipeline for a fitted scikit-learn classifier.

    - classification_report
    - weighted F1 + 95% CI
    - confusion matrix
    - macro AUC + 95% CI (if predict_proba available)
    - multi-class ROC curves (if predict_proba available)

    Parameters
    ----------
    model : fitted sklearn classifier
    X_test : test features
    y_test : test labels
    model_name : label used in plots & logs
    save_dir : if provided, saves CM and ROC figures there
    """
    print(f"\n--- Evaluation Report: {model_name} ---\n")

    # 1. Predictions & basic report
    y_pred = model.predict(X_test)
    classes_list = getattr(model, "classes_", None)

    print("Classification Report:")
    if classes_list is not None:
        print(
            classification_report(
                y_test,
                y_pred,
                digits=4,
                labels=classes_list,
            )
        )
    else:
        print(classification_report(y_test, y_pred, digits=4))

    # 2. F1 with CI
    f1_mean, f1_ci_low, f1_ci_high = bootstrap_f1_ci(
        y_test,
        y_pred,
        average="weighted",
    )
    print(f"Weighted F1 Score: {f1_mean:.4f}")
    print(f"95% CI for F1 Score: [{f1_ci_low:.4f}, {f1_ci_high:.4f}]")

    # 3. Confusion Matrix
    labels_for_cm = classes_list if classes_list is not None else np.unique(y_test)
    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=labels_for_cm,
        model_name=model_name,
        save_dir=save_dir,
    )

    # 4. AUC + ROC curves (requires predict_proba)
    if not hasattr(model, "predict_proba"):
        print("\nModel has no predict_proba; skipping AUC/ROC.")
        return

    y_scores = model.predict_proba(X_test)
    if classes_list is None:
        classes_list = np.arange(y_scores.shape[1])

    auc_score, auc_ci_low, auc_ci_high = bootstrap_auc_ci_multiclass(
        y_test,
        y_scores,
    )
    print(f"\nMacro-Averaged AUC: {auc_score:.4f}")
    print(f"95% CI for Macro-Averaged AUC: [{auc_ci_low:.4f}, {auc_ci_high:.4f}]")

    plot_multiclass_roc(
        y_test,
        y_scores,
        classes_list=classes_list,
        label_prefix=model_name,
        save_dir=save_dir,
    )

def evaluate_ML_multiclass_with_extra(
    model,
    predict_fn,
    X_test_text,
    X_test_extra,
    y_test,
    model_name="Model",
    save_dir=None,
):
    """
    Evaluation for models that use (text + extra) features, but reusing
    your existing bootstrap CI + plotting functions.

    - predict_fn is one of: predict_logreg, predict_svm, predict_rf, predict_lgbm
    - X_test_text is TF-IDF
    - X_test_extra is e.g. [ai_entropy_norm, ai_max_prob_norm]
    """
    print(f"\n--- Evaluation Report (Text + Extra): {model_name} ---\n")

    # 1) Get predictions and scores using the wrapper
    y_pred, y_scores = predict_fn(
        model=model,
        X_test_text=X_test_text,
        X_test_extra=X_test_extra,
    )

    classes_list = model.classes_

    # 2) Classification report
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
            labels=classes_list,
        )
    )

    # 3) F1 with bootstrap CI
    f1_mean, f1_ci_low, f1_ci_high = bootstrap_f1_ci(
        y_test,
        y_pred,
        average="weighted",
    )
    print(f"Weighted F1 Score: {f1_mean:.4f}")
    print(f"95% CI for F1 Score: [{f1_ci_low:.4f}, {f1_ci_high:.4f}]")

    # 4) Confusion matrix
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        labels=classes_list,
        model_name=model_name,
        save_dir=save_dir,
        cmap_name="Blues",
    )

    # 5) AUC + ROC (if scores/probabilities available)
    if y_scores is None:
        print("\nModel doesn't provide probabilities/scores; skipping AUC/ROC.")
        return

    auc_score, auc_ci_low, auc_ci_high = bootstrap_auc_ci_multiclass(
        y_true=y_test,
        y_scores=y_scores,
    )
    print(f"\nMacro-Averaged AUC: {auc_score:.4f}")
    print(f"95% CI for Macro-Averaged AUC: [{auc_ci_low:.4f}, {auc_ci_high:.4f}]")

    plot_multiclass_roc(
        y_true=y_test,
        y_score=y_scores,
        classes_list=classes_list,
        label_prefix=model_name,
        save_dir=save_dir,
    )
    
# -------------------------------------------------------------------
# 6. New generic wrapper: works with y_true / y_pred / y_proba directly
# -------------------------------------------------------------------

def evaluate_ml_model(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model",
    label_names: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    print_report: bool = True,
    compute_bootstrap_ci: bool = True,
) -> dict:
    """
    Generic evaluation helper for ML classifiers when you already have
    y_pred and (optionally) y_proba.

    Works nicely with manual predict_* helpers (e.g., predict_lgbm).

    - accuracy
    - macro & weighted F1
    - optional bootstrap F1 CI
    - confusion matrix (saved if output_dir provided)
    - optional macro AUC + 95% CI + ROC curves if y_proba is given

    Parameters
    ----------
    y_true : 1D array-like
        True labels (encoded).
    y_pred : 1D array-like
        Predicted labels.
    y_proba : 2D array or None
        Predicted probabilities [n_samples, n_classes].
    model_name : str
        Used in printouts and plot titles.
    label_names : list of str or None
        Names for classes in order of encoded label values. If None,
        uses sorted unique(y_true).
    output_dir : str or Path or None
        If provided, saves confusion matrix (and ROC) PNG here.
    prefix : str or None
        Prefix for saved files. If None, uses model_name-based name.
    print_report : bool
        If True, prints classification report and metrics.
    compute_bootstrap_ci : bool
        If True, compute bootstrap CIs for F1 and AUC (if y_proba).

    Returns
    -------
    metrics : dict with keys:
        - model_name
        - accuracy
        - f1_macro
        - f1_weighted
        - f1_ci (optional)
        - auc_macro (optional)
        - auc_ci (optional)
        - report (classification_report dict)
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

    metrics = {
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

    if print_report:
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
        # Use numeric class order inferred from y_true
        classes_list = np.unique(y_true)

        if compute_bootstrap_ci:
            auc_mean, auc_ci_low, auc_ci_high = bootstrap_auc_ci_multiclass(
                y_true, y_scores
            )
            metrics["auc_macro"] = auc_mean
            metrics["auc_ci"] = (auc_ci_low, auc_ci_high)

        # ROC curves
        prefix_for_roc = prefix if prefix is not None else model_name
        plot_multiclass_roc(
            y_true,
            y_scores,
            classes_list=classes_list,
            label_prefix=prefix_for_roc,
            save_dir=output_dir,
        )

    return metrics