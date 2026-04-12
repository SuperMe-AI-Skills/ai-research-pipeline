# -*- coding: utf-8 -*-
"""ml_interpret.py

Global feature-importance utilities for structured data ML models:
- Linear models (Logistic Regression, Ridge) via coefficients
- Tree-based models (Random Forest, XGBoost, LightGBM, CatBoost) via
  feature_importances_
- Permutation-based importance (SVM and other nonlinear models)
- Horizontal bar plot for any importance DataFrame
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance


# -------------------------------------------------------------------
# 1. Linear models: global importance via absolute coefficients
# -------------------------------------------------------------------

def linear_model_importance(
    model,
    feature_names: Sequence[str],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Compute global feature importance for a linear model (e.g. LogisticRegression,
    Ridge) using the absolute value of coefficients.

    For multiclass (coef_ shape = [n_classes, n_features]):
      - take mean absolute coefficient across classes.

    Returns
    -------
    importance_df : pd.DataFrame with columns:
        - feature
        - importance       (raw absolute coefficient or mean across classes)
        - relative_importance (0-100 scaled)
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


# -------------------------------------------------------------------
# 2. Tree-based models: feature_importances_
# -------------------------------------------------------------------

def tree_feature_importance(
    model,
    feature_names: Sequence[str],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Compute global feature importance for tree-based models
    (RandomForest, XGBoost, LightGBM, CatBoost) using .feature_importances_.

    Returns
    -------
    importance_df : pd.DataFrame with columns:
        - feature
        - importance           (raw feature_importances_)
        - relative_importance  (0-100 scaled)
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


# -------------------------------------------------------------------
# 3. Permutation-based importance (e.g. for SVM)
# -------------------------------------------------------------------

def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: Sequence,
    feature_names: Sequence[str],
    scoring: str = "f1_weighted",
    n_repeats: int = 5,
    max_samples: int = 1000,
    random_state: int = 42,
    normalize: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Permutation-based feature importance, e.g. for nonlinear SVM.

    Steps:
      - Optionally subsample up to max_samples from X, y for efficiency.
      - Run sklearn.inspection.permutation_importance.
      - Clip small negative means to 0.
      - Scale to [0, 100] as 'relative_importance'.

    Parameters
    ----------
    model : fitted classifier or regressor
    X : np.ndarray (validation/test features)
    y : labels or targets
    feature_names : list of feature names
    scoring : e.g. 'f1_weighted', 'accuracy', 'r2', 'neg_mean_squared_error'
    n_repeats : number of permutations per feature
    max_samples : max number of samples to use
    random_state : int
    normalize : bool
    verbose : bool

    Returns
    -------
    importance_df : pd.DataFrame with columns:
        - feature
        - importance_mean
        - importance_std
        - relative_importance (0-100)
        - rank
    """
    X_arr = np.asarray(X)
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

    result = permutation_importance(
        model,
        X_arr,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring=scoring,
    )

    mean_imp = np.asarray(result.importances_mean, dtype=float)
    std_imp = np.asarray(result.importances_std, dtype=float)

    # Clip small negatives to zero
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


# -------------------------------------------------------------------
# 4. Horizontal bar plot for any importance DataFrame
# -------------------------------------------------------------------

def plot_importance_barh(
    importance_df: pd.DataFrame,
    model_name: str = "Model",
    top_n: int = 20,
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Pretty horizontal bar plot of top-N feature importances.

    Expects importance_df with columns 'feature' and 'relative_importance' (0-100).
    Works with output from linear_model_importance(), tree_feature_importance(),
    or compute_permutation_importance().
    """
    df_plot = importance_df.head(top_n).copy()
    # Reverse order so most important is at top
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
        print(f"Saved feature-importance plot to: {save_path}")

    plt.show()
    plt.close()
