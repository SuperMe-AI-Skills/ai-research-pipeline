# -*- coding: utf-8 -*-
"""Minimal LightGBM utilities.

Implements:
  - train_lgbm_text_only
  - train_lgbm_text_extra
  - predict_lgbm

Logic matches the original notebook-style manual tuning:
  manual grid over n_estimators, learning_rate, num_leaves, max_depth, class_weight
  metric = F1 (weighted) on validation set.
"""

import time
from itertools import product

import numpy as np
from scipy.sparse import hstack, csr_matrix, issparse
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score


# -------------------------------------------------------------------
# Helper: combine text + extra features
# -------------------------------------------------------------------

def _build_augmented_matrix(X_text, X_extra):
    """Combine text TF-IDF matrix with optional extra numeric features."""
    if X_extra is None:
        return X_text

    # Ensure both are sparse before hstack
    if not issparse(X_text):
        X_text = csr_matrix(X_text)
    if not issparse(X_extra):
        X_extra = csr_matrix(X_extra)

    return hstack([X_text, X_extra])


# -------------------------------------------------------------------
# Core: training logic (works on any already-combined X_train / X_val)
# -------------------------------------------------------------------

def _train_lgbm_core(
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid=None,
    random_state=42,
    verbose=True,
):
    """
    Core LightGBM training with manual grid search.

    Parameters
    ----------
    X_train, X_val : array or sparse matrix
    y_train, y_val : arrays of labels
    param_grid : dict of lists, or None to use default grid
    random_state : int
    verbose : bool

    Returns
    -------
    best_model : LGBMClassifier
    best_params : dict
    best_f1 : float
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "num_leaves": [31, 63],
            "max_depth": [None, 10],
            "class_weight": ["balanced", None],
        }

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    # Pick LightGBM objective from the actual label cardinality
    # ("multiclass" with 2 classes trains one tree per class per iteration,
    # which is wasteful and produces miscalibrated probabilities).
    n_classes = int(np.unique(y_train).size)
    lgbm_objective = "binary" if n_classes == 2 else "multiclass"

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    all_combos = [dict(zip(keys, v)) for v in product(*values)]

    best_f1 = -1.0
    best_model = None
    best_params = None

    if verbose:
        print("\n[LGBM] Starting manual LightGBM tuning...")
        print(f"[LGBM] Objective: {lgbm_objective} (n_classes={n_classes})")
        print(f"[LGBM] Total combinations: {len(all_combos)}")
        print(f"[LGBM] Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    start_time = time.time()

    for i, params in enumerate(all_combos, start=1):
        if verbose:
            print(f"\n[LGBM] Comb {i}/{len(all_combos)}: {params}")

        # LightGBM expects max_depth=-1 for "no limit"; None is invalid.
        max_depth = params["max_depth"]
        if max_depth is None:
            max_depth = -1

        model = LGBMClassifier(
            objective=lgbm_objective,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=max_depth,
            class_weight=params["class_weight"],
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,  # suppress internal LGBM logs
        )

        try:
            # IMPORTANT: no 'verbose' kw in fit()
            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)
            f1_val = f1_score(y_val, y_val_pred, average="weighted")

            if verbose:
                print(f"[LGBM] F1 (weighted): {f1_val:.4f}")

            if f1_val > best_f1:
                best_f1 = f1_val
                best_model = model
                best_params = params
                if verbose:
                    print(f"[LGBM] New best F1: {f1_val:.4f} with params: {best_params}")
        except Exception as e:
            if verbose:
                print(f"[LGBM] Skipping combination {i} due to error: {e}")
            continue

    total_time = time.time() - start_time

    if best_model is None:
        raise RuntimeError("No valid LightGBM model was found during tuning.")

    if verbose:
        print("\n[LGBM] Hyperparameter search completed.")
        print(f"[LGBM] Best params: {best_params}")
        print(f"[LGBM] Best validation weighted F1: {best_f1:.4f}")
        print(f"[LGBM] Total tuning time: {total_time:.2f} seconds")

    return best_model, best_params, float(best_f1)


# -------------------------------------------------------------------
# Public: text-only wrapper
# -------------------------------------------------------------------

def train_lgbm_text_only(
    X_train_text,
    y_train,
    X_val_text,
    y_val,
    param_grid=None,
    random_state=42,
    verbose=True,
):
    """
    LightGBM training with text-only TF-IDF features.

    This mirrors the original notebook tuner, but packaged as a function.
    """
    return _train_lgbm_core(
        X_train=X_train_text,
        y_train=y_train,
        X_val=X_val_text,
        y_val=y_val,
        param_grid=param_grid,
        random_state=random_state,
        verbose=verbose,
    )


# -------------------------------------------------------------------
# Public: text + extra wrapper
# -------------------------------------------------------------------

def train_lgbm_text_extra(
    X_train_text,
    X_train_extra,
    y_train,
    X_val_text,
    X_val_extra,
    y_val,
    param_grid=None,
    random_state=42,
    verbose=True,
):
    """
    LightGBM training with text (TF-IDF) + extra numeric features.
    """
    X_train = _build_augmented_matrix(X_train_text, X_train_extra)
    X_val = _build_augmented_matrix(X_val_text, X_val_extra)

    return _train_lgbm_core(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_grid=param_grid,
        random_state=random_state,
        verbose=verbose,
    )


# -------------------------------------------------------------------
# Public: prediction helper
# -------------------------------------------------------------------

def predict_lgbm(model, X_test_text, X_test_extra=None):
    """
    Predict labels & probabilities for test data, handling text-only or text+extra.

    Returns
    -------
    y_pred : array of predicted labels
    y_proba : 2D array [n_samples, n_classes] or None
    """
    X_test = _build_augmented_matrix(X_test_text, X_test_extra)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    return y_pred, y_proba