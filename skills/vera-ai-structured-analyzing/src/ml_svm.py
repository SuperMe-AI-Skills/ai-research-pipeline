# -*- coding: utf-8 -*-
"""ml_svm.py

Support Vector Machine for structured data.

Implements:
  - train_svm: manual grid search with itertools.product
  - predict_svm: returns (y_pred, y_proba)

Supports both classification (SVC) and regression (SVR).
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.metrics import f1_score, mean_squared_error


# -------------------------------------------------------------------
# 1. Core trainer
# -------------------------------------------------------------------

def train_svm(
    X_train: np.ndarray,
    y_train: Sequence,
    X_val: np.ndarray,
    y_val: Sequence,
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    task: str = "classification",
    probability: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    SVM training with manual grid search.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Feature matrices.
    y_train, y_val : array-like
        Target arrays.
    param_grid : dict of hyperparameters -> list of values
    task : 'classification' or 'regression'
    probability : bool
        Enable predict_proba for SVC (Platt scaling).
    random_state : int
    verbose : bool

    Returns
    -------
    best_model : fitted model
    best_params : dict of best hyperparameters
    best_score : best validation score
    """
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    if task == "classification":
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
                "class_weight": ["balanced", None],
            }
    else:
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
                "epsilon": [0.1, 0.2],
            }

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    all_param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = -np.inf
    best_model = None
    best_params: Dict[str, Any] = {}

    if verbose:
        print(f"\n[SVM] Starting hyperparameter search over {len(all_param_combos)} combinations...")

    start_time = time.time()

    for i, params in enumerate(all_param_combos, start=1):
        if verbose:
            print(f"\n[SVM] Combination {i}/{len(all_param_combos)}: {params}")

        try:
            if task == "classification":
                model = SVC(
                    **params,
                    probability=probability,
                    random_state=random_state,
                )
                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)
                score = f1_score(y_val, y_val_pred, average="weighted")
                metric_name = "F1 (weighted)"
            else:
                model = SVR(**params)
                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                score = -rmse
                metric_name = "neg RMSE"
        except Exception as e:
            if verbose:
                print(f"[SVM] Failed with parameters {params} due to: {e}")
            continue

        if verbose:
            print(f"[SVM] Validation {metric_name}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    total_time = time.time() - start_time

    if best_model is None:
        raise RuntimeError("No valid SVM model was found during tuning.")

    if verbose:
        print("\n[SVM] Hyperparameter search completed.")
        print(f"[SVM] Best params: {best_params}")
        print(f"[SVM] Best validation score: {best_score:.4f}")
        print(f"[SVM] Total tuning time: {total_time:.2f} seconds")
        if task == "classification":
            print(f"[SVM] probability={probability}")

    return best_model, best_params, float(best_score)


# -------------------------------------------------------------------
# 2. Prediction helper
# -------------------------------------------------------------------

def predict_svm(
    model,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict labels & probabilities for test data.

    Returns
    -------
    y_pred : 1D array of predicted labels/values
    y_proba : 2D array [n_samples, n_classes] if available, else None
    """
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None

    return y_pred, y_proba
