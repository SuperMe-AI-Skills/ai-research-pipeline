# -*- coding: utf-8 -*-
"""ml_interpret.py

Global feature-importance utilities for ML models:
- Linear models (e.g. Logistic Regression)
- Tree-based models (Random Forest, LightGBM)
- Permutation-based importance (e.g. nonlinear SVM)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from scipy.sparse import issparse


# -------------------------------------------------------------------
# 1. Linear models: global importance via absolute coefficients
# -------------------------------------------------------------------

def linear_model_global_importance(
    model,
    feature_names: Sequence[str],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Compute global feature importance for a linear model (e.g. LogisticRegression)
    using the absolute value of coefficients.

    For multiclass (coef_ shape = [n_classes, n_features]):
      - take mean absolute coefficient across classes.

    Returns
    -------
    importance_df : pd.DataFrame with columns:
        - feature
        - importance       (raw absolute coefficient or mean across classes)
        - relative_importance (0–100 scaled)
        - rank
    """
    coef = getattr(model, "coef_", None)
    if coef is None:
        raise ValueError("Model has no .coef_ attribute (not a linear model?)")

    coef = np.asarray(coef, dtype=float)
    if coef.ndim == 1:
        abs_imp = np.abs(coef)
    else:
        # multiclass: average |coef| across classes
        abs_imp = np.mean(np.abs(coef), axis=0)

    if abs_imp.shape[0] != len(feature_names):
        raise ValueError(
            f"Length of coefficients ({abs_imp.shape[0]}) does not match "
            f"number of feature names ({len(feature_names)})."
        )

    importance = abs_imp.copy()
    if normalize:
        max_val = importance.max()
        if max_val > 0:
            rel = importance / max_val * 100.0
        else:
            rel = np.zeros_like(importance)
    else:
        rel = importance

    df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": importance,
            "relative_importance": rel,
        }
    ).sort_values("importance", ascending=False)

    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def plot_linear_importance_barh(
    importance_df: pd.DataFrame,
    model_name: str = "Linear Model",
    top_n: int = 20,
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Pretty horizontal bar plot of top-N linear feature importances.

    Expects importance_df from linear_model_global_importance().
    Uses 'relative_importance' (0–100) for the x-axis.
    """
    df_plot = importance_df.head(top_n).copy()
    # Reverse order so most important at top
    df_plot = df_plot.iloc[::-1]

    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    sns.barplot(
        x="relative_importance",
        y="feature",
        data=df_plot,
        palette="mako",
    )
    plt.xlabel("Relative Importance (%)")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Features - {model_name}")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved linear importance plot to: {save_path}")

    plt.close()
    plt.close()

# -------------------------------------------------------------------
# 2. Tree-based models: feature_importances_
# -------------------------------------------------------------------

def compute_tree_feature_importance(
    model,
    feature_names: Sequence[str],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Compute global feature importance for tree-based models
    (RandomForest, LightGBM, etc.) using .feature_importances_.

    Returns
    -------
    importance_df : pd.DataFrame with columns:
        - feature
        - importance           (raw feature_importances_)
        - relative_importance  (0–100 scaled)
        - rank
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("Model has no .feature_importances_ attribute.")

    importances = np.asarray(importances, dtype=float)
    if importances.shape[0] != len(feature_names):
        raise ValueError(
            f"Length of feature_importances_ ({importances.shape[0]}) does not "
            f"match number of feature names ({len(feature_names)})."
        )

    importance = importances.copy()
    if normalize:
        max_val = importance.max()
        if max_val > 0:
            rel = importance / max_val * 100.0
        else:
            rel = np.zeros_like(importance)
    else:
        rel = importance

    df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": importance,
            "relative_importance": rel,
        }
    ).sort_values("importance", ascending=False)

    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def plot_feature_importance_pretty(
    importance_df: pd.DataFrame,
    model_name: str = "Model",
    y_label: str = "Feature",
    top_n: int = 20,
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Pretty horizontal bar plot for any importance DataFrame that has:
      - 'feature'
      - 'relative_importance' (0–100)

    Typically used with compute_tree_feature_importance()
    or compute_permutation_feature_importance().
    """
    df_plot = importance_df.head(top_n).copy()
    df_plot = df_plot.iloc[::-1]

    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    sns.barplot(
        x="relative_importance",
        y="feature",
        data=df_plot,
        palette="mako",
    )
    plt.xlabel("Relative Importance (%)")
    plt.ylabel(y_label)
    plt.title(f"Top {top_n} Features - {model_name}")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved feature-importance plot to: {save_path}")

    plt.close()
    plt.close()

# -------------------------------------------------------------------
# 3. Permutation-based importance (e.g. for SVM)
# -------------------------------------------------------------------

def compute_permutation_feature_importance(
    model,
    X,
    y: Sequence[int],
    feature_names: Sequence[str],
    max_samples: int = 1000,
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str = "f1_weighted",
    normalize: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Permutation-based feature importance, e.g. for nonlinear SVM.

    Steps:
      - Optionally subsample up to max_samples from X, y for efficiency.
      - (Optionally) convert to dense if X is sparse.
      - Run sklearn.inspection.permutation_importance.
      - Clip small negative means to 0.
      - Scale to [0, 100] as 'relative_importance'.

    Parameters
    ----------
    model : fitted classifier (e.g., SVC with probability=True or decision_function)
    X : array-like or sparse matrix (validation features)
    y : labels
    feature_names : list of feature names (e.g. tfidf.get_feature_names_out())
    max_samples : int
        Max number of samples to use from X for permutation importance.
    n_repeats : int
        Number of permutations per feature.
    random_state : int
    scoring : str
        E.g. 'f1_weighted', 'accuracy', etc.
    normalize : bool
        If True, 'relative_importance' column is 0–100 scaled.
    verbose : bool

    Returns
    -------
    importance_df : pd.DataFrame with columns:
        - feature
        - importance_mean
        - importance_std
        - relative_importance (0–100)
        - rank
    """
    X_arr = X
    y = np.asarray(y)

    n_val = X_arr.shape[0]
    if n_val > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_val, size=max_samples, replace=False)
        X_arr = X_arr[idx]
        y = y[idx]

    if verbose:
        print(
            f"[Permutation] Running permutation_importance on "
            f"{X_arr.shape[0]} samples, {X_arr.shape[1]} features"
        )

    # If sparse, permutation_importance can handle it, but for safety you can densify.
    if issparse(X_arr):
        X_perm = X_arr.toarray()
    else:
        X_perm = np.asarray(X_arr)

    result = permutation_importance(
        model,
        X_perm,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring=scoring,
    )

    mean_imp = np.asarray(result.importances_mean, dtype=float)
    std_imp = np.asarray(result.importances_std, dtype=float)

    # Clip small negatives to zero for nicer interpretation
    mean_imp = np.clip(mean_imp, a_min=0.0, a_max=None)

    if mean_imp.shape[0] != len(feature_names):
        raise ValueError(
            f"Permutation importance length ({mean_imp.shape[0]}) does not match "
            f"number of feature names ({len(feature_names)})."
        )

    if normalize:
        max_val = mean_imp.max()
        if max_val > 0:
            rel = mean_imp / max_val * 100.0
        else:
            rel = np.zeros_like(mean_imp)
    else:
        rel = mean_imp

    df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance_mean": mean_imp,
            "importance_std": std_imp,
            "relative_importance": rel,
        }
    ).sort_values("importance_mean", ascending=False)

    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1

    if verbose:
        print("[Permutation] Finished computing permutation importance.")

    return df