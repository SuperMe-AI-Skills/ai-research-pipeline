# -*- coding: utf-8 -*-
"""ml_logreg.py

Logistic Regression / Ridge Regression for structured data.

Implements:
  - train_logreg: manual grid search with itertools.product
  - predict_logreg: returns (y_pred, y_proba)

Supports both classification (LogisticRegression) and regression (Ridge).
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error


# -------------------------------------------------------------------
# 1. Core trainer
# -------------------------------------------------------------------

def train_logreg(
    X_train: np.ndarray,
    y_train: Sequence,
    X_val: np.ndarray,
    y_val: Sequence,
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    task: str = "classification",
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Logistic Regression (classification) or Ridge (regression) training
    with manual grid search.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Feature matrices.
    y_train, y_val : array-like
        Target arrays.
    param_grid : dict of hyperparameters -> list of values
    task : 'classification' or 'regression'
    random_state : int
    verbose : bool

    Returns
    -------
    best_model : fitted model
    best_params : dict of best hyperparameters
    best_score : best validation score (F1 for classification, neg RMSE for regression)
    """
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    if task == "classification":
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs", "saga"],
                "penalty": ["l2"],
                "class_weight": ["balanced", None],
            }
    else:
        # Regression: Ridge
        if param_grid is None:
            param_grid = {
                "alpha": [0.1, 1, 10],
            }

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    all_param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = -np.inf
    best_model = None
    best_params: Dict[str, Any] = {}

    tag = "LogReg" if task == "classification" else "Ridge"

    if verbose:
        print(f"\n[{tag}] Starting hyperparameter search over {len(all_param_combos)} combinations...")

    start_time = time.time()

    for i, params in enumerate(all_param_combos, start=1):
        if verbose:
            print(f"\n[{tag}] Combination {i}/{len(all_param_combos)}: {params}")

        try:
            if task == "classification":
                model = LogisticRegression(
                    **params,
                    random_state=random_state,
                    n_jobs=-1,
                    max_iter=1000,
                )
                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)
                score = f1_score(y_val, y_val_pred, average="weighted")
                metric_name = "F1 (weighted)"
            else:
                model = Ridge(**params, random_state=random_state)
                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                score = -rmse  # negate so higher is better
                metric_name = "neg RMSE"
        except Exception as e:
            if verbose:
                print(f"[{tag}] Failed with parameters {params} due to: {e}")
            continue

        if verbose:
            print(f"[{tag}] Validation {metric_name}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    total_time = time.time() - start_time

    if best_model is None:
        raise RuntimeError(f"No valid {tag} model was found during tuning.")

    if verbose:
        print(f"\n[{tag}] Hyperparameter search completed.")
        print(f"[{tag}] Best params: {best_params}")
        print(f"[{tag}] Best validation score: {best_score:.4f}")
        print(f"[{tag}] Total tuning time: {total_time:.2f} seconds")

    return best_model, best_params, float(best_score)


# -------------------------------------------------------------------
# 2. Prediction helper
# -------------------------------------------------------------------

def predict_logreg(
    model,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict labels & probabilities for test data.

    Returns
    -------
    y_pred : 1D array of predicted labels/values
    y_proba : 2D array [n_samples, n_classes] for classification, None for regression
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    return y_pred, y_proba
