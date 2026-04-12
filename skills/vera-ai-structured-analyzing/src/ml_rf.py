# -*- coding: utf-8 -*-
"""ml_rf.py

Random Forest for structured data.

Implements:
  - train_rf: manual grid search with itertools.product
  - predict_rf: returns (y_pred, y_proba)

Supports both classification (RandomForestClassifier) and regression
(RandomForestRegressor).
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error


# -------------------------------------------------------------------
# 1. Core trainer
# -------------------------------------------------------------------

def train_rf(
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
    Random Forest training with manual grid search.

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
    best_score : best validation score
    """
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    if task == "classification":
        if param_grid is None:
            param_grid = {
                "n_estimators": [200, 500],
                "max_depth": [None, 20, 50],
                "min_samples_split": [2, 5],
                "class_weight": ["balanced", None],
            }
    else:
        if param_grid is None:
            param_grid = {
                "n_estimators": [200, 500],
                "max_depth": [None, 20, 50],
                "min_samples_split": [2, 5],
            }

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    all_param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = -np.inf
    best_model = None
    best_params: Dict[str, Any] = {}

    if verbose:
        print(f"\n[RF] Starting hyperparameter search over {len(all_param_combos)} combinations...")

    start_time = time.time()

    for i, params in enumerate(all_param_combos, start=1):
        if verbose:
            print(f"\n[RF] Combination {i}/{len(all_param_combos)}: {params}")

        try:
            if task == "classification":
                model = RandomForestClassifier(
                    **params,
                    random_state=random_state,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)
                score = f1_score(y_val, y_val_pred, average="weighted")
                metric_name = "F1 (weighted)"
            else:
                model = RandomForestRegressor(
                    **params,
                    random_state=random_state,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                score = -rmse
                metric_name = "neg RMSE"
        except Exception as e:
            if verbose:
                print(f"[RF] Failed with parameters {params} due to: {e}")
            continue

        if verbose:
            print(f"[RF] Validation {metric_name}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    total_time = time.time() - start_time

    if best_model is None:
        raise RuntimeError("No valid RandomForest model was found during tuning.")

    if verbose:
        print("\n[RF] Hyperparameter search completed.")
        print(f"[RF] Best params: {best_params}")
        print(f"[RF] Best validation score: {best_score:.4f}")
        print(f"[RF] Total tuning time: {total_time:.2f} seconds")

    return best_model, best_params, float(best_score)


# -------------------------------------------------------------------
# 2. Prediction helper
# -------------------------------------------------------------------

def predict_rf(
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
