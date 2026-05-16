from __future__ import annotations

import itertools
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import hstack, csr_matrix, issparse
from sklearn.metrics import f1_score
from sklearn.svm import SVC


# -------------------------------------------------------------------
# 0. Internal helper: combine text + extra features
# -------------------------------------------------------------------

def _build_augmented_matrix(X_text, X_extra):
    """
    Combine text features (typically TF-IDF) with extra numeric features.
    If X_extra is None, returns X_text as-is.
    """
    if X_extra is None:
        return X_text

    if not issparse(X_text):
        raise ValueError("X_text should be a sparse matrix (e.g., TF-IDF).")

    if not issparse(X_extra):
        X_extra = csr_matrix(X_extra)

    return hstack([X_text, X_extra])


# -------------------------------------------------------------------
# 1. Core trainer (feature-agnostic)
# -------------------------------------------------------------------

def train_svm(
    X_train_text,
    y_train: Sequence[int],
    X_val_text,
    y_val: Sequence[int],
    X_train_extra=None,
    X_val_extra=None,
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    random_state: int = 42,
    verbose: bool = True,
    probability: bool = True,   # <-- NEW
) -> Tuple[SVC, Dict[str, Any], float]:
    """
    Core SVM (SVC) training with manual grid search.

    probability:
      - True  -> enables predict_proba (slower: Platt scaling CV internally)
      - False -> faster training; predict_proba unavailable
    """
    if param_grid is None:
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
            "class_weight": ["balanced", None],
        }

    # Combine features
    X_train = _build_augmented_matrix(X_train_text, X_train_extra)
    X_val = _build_augmented_matrix(X_val_text, X_val_extra)

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    all_param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_f1 = -1.0
    best_model: Optional[SVC] = None
    best_params: Dict[str, Any] = {}

    if verbose:
        print(f"\n[SVM] Starting hyperparameter search over {len(all_param_combos)} combinations...")

    start_time = time.time()

    for i, params in enumerate(all_param_combos, start=1):
        if verbose:
            print(f"\n[SVM] Combination {i}/{len(all_param_combos)}: {params}")

        svc = SVC(
            **params,
            probability=probability,     # <-- NEW
            random_state=random_state,
        )

        try:
            svc.fit(X_train, y_train)
        except Exception as e:  # noqa
            if verbose:
                print(f"[SVM] Failed with parameters {params} due to: {e}")
            continue

        y_val_pred = svc.predict(X_val)
        f1_val = f1_score(y_val, y_val_pred, average="weighted")

        if verbose:
            print(f"[SVM] Validation weighted F1: {f1_val:.4f}")

        if f1_val > best_f1:
            best_f1 = f1_val
            best_model = svc
            best_params = params

    total_time = time.time() - start_time

    if best_model is None:
        raise RuntimeError("No valid SVM model was found during tuning.")

    if verbose:
        print("\n[SVM] Hyperparameter search completed.")
        print(f"[SVM] Best params: {best_params}")
        print(f"[SVM] Best validation weighted F1: {best_f1:.4f}")
        print(f"[SVM] Total tuning time: {total_time:.2f} seconds")
        print(f"[SVM] probability={probability}")

    return best_model, best_params, float(best_f1)


# -------------------------------------------------------------------
# 2. Convenience wrappers: text-only vs text+extra
# -------------------------------------------------------------------

def train_svm_text_only(
    X_train_text,
    y_train: Sequence[int],
    X_val_text,
    y_val: Sequence[int],
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    random_state: int = 42,
    verbose: bool = True,
    probability: bool = True,   # <-- NEW
) -> Tuple[SVC, Dict[str, Any], float]:
    return train_svm(
        X_train_text=X_train_text,
        y_train=y_train,
        X_val_text=X_val_text,
        y_val=y_val,
        X_train_extra=None,
        X_val_extra=None,
        param_grid=param_grid,
        random_state=random_state,
        verbose=verbose,
        probability=probability,  # <-- NEW
    )


def train_svm_text_extra(
    X_train_text,
    X_train_extra,
    y_train: Sequence[int],
    X_val_text,
    X_val_extra,
    y_val: Sequence[int],
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    random_state: int = 42,
    verbose: bool = True,
    probability: bool = True,   # <-- NEW
) -> Tuple[SVC, Dict[str, Any], float]:
    return train_svm(
        X_train_text=X_train_text,
        y_train=y_train,
        X_val_text=X_val_text,
        y_val=y_val,
        X_train_extra=X_train_extra,
        X_val_extra=X_val_extra,
        param_grid=param_grid,
        random_state=random_state,
        verbose=verbose,
        probability=probability,  # <-- NEW
    )


# -------------------------------------------------------------------
# 3. Prediction helper
# -------------------------------------------------------------------

def predict_svm(
    model: SVC,
    X_test_text,
    X_test_extra=None,
):
    """
    Returns:
      y_pred  : predicted labels
      y_proba : predicted probabilities if available, else None
    """
    X_test = _build_augmented_matrix(X_test_text, X_test_extra)
    y_pred = model.predict(X_test)

    # predict_proba only works if fitted with probability=True
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None

    return y_pred, y_proba