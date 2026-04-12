# -*- coding: utf-8 -*-
"""ml_catboost.py

CatBoost for structured data.

Implements:
  - train_catboost: manual grid search with itertools.product
  - predict_catboost: returns (y_pred, y_proba)

Supports both classification (CatBoostClassifier) and regression
(CatBoostRegressor). Native categorical feature handling via cat_features.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import f1_score, mean_squared_error


# -------------------------------------------------------------------
# 1. Core trainer
# -------------------------------------------------------------------

def train_catboost(
    X_train: np.ndarray,
    y_train: Sequence,
    X_val: np.ndarray,
    y_val: Sequence,
    cat_features: Optional[List[int]] = None,
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    task: str = "classification",
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    CatBoost training with manual grid search.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Feature matrices.
    y_train, y_val : array-like
        Target arrays.
    cat_features : list of int or None
        Indices of categorical features for native CatBoost handling.
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

    if param_grid is None:
        param_grid = {
            "iterations": [200, 500],
            "learning_rate": [0.01, 0.1],
            "depth": [4, 6, 8],
        }

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    all_param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = -np.inf
    best_model = None
    best_params: Dict[str, Any] = {}

    if verbose:
        print(f"\n[CatBoost] Starting hyperparameter search over {len(all_param_combos)} combinations...")
        if cat_features:
            print(f"[CatBoost] Categorical feature indices: {cat_features}")

    start_time = time.time()

    for i, params in enumerate(all_param_combos, start=1):
        if verbose:
            print(f"\n[CatBoost] Combination {i}/{len(all_param_combos)}: {params}")

        try:
            if task == "classification":
                n_classes = len(np.unique(y_train))
                loss_function = "Logloss" if n_classes == 2 else "MultiClass"
                model = CatBoostClassifier(
                    **params,
                    loss_function=loss_function,
                    random_seed=random_state,
                    cat_features=cat_features,
                    verbose=0,
                    allow_writing_files=False,
                )
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
                y_val_pred = model.predict(X_val).flatten()
                # CatBoost predict returns strings for classification; cast to int
                y_val_pred = y_val_pred.astype(y_val.dtype)
                score = f1_score(y_val, y_val_pred, average="weighted")
                metric_name = "F1 (weighted)"
            else:
                model = CatBoostRegressor(
                    **params,
                    loss_function="RMSE",
                    random_seed=random_state,
                    cat_features=cat_features,
                    verbose=0,
                    allow_writing_files=False,
                )
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
                y_val_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                score = -rmse
                metric_name = "neg RMSE"
        except Exception as e:
            if verbose:
                print(f"[CatBoost] Failed with parameters {params} due to: {e}")
            continue

        if verbose:
            print(f"[CatBoost] Validation {metric_name}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    total_time = time.time() - start_time

    if best_model is None:
        raise RuntimeError("No valid CatBoost model was found during tuning.")

    if verbose:
        print("\n[CatBoost] Hyperparameter search completed.")
        print(f"[CatBoost] Best params: {best_params}")
        print(f"[CatBoost] Best validation score: {best_score:.4f}")
        print(f"[CatBoost] Total tuning time: {total_time:.2f} seconds")

    return best_model, best_params, float(best_score)


# -------------------------------------------------------------------
# 2. Prediction helper
# -------------------------------------------------------------------

def predict_catboost(
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

    # Flatten if needed (CatBoost can return 2D for predict)
    if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
        y_pred = y_pred.flatten()

    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None

    return y_pred, y_proba
