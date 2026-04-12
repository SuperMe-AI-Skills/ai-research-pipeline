# -*- coding: utf-8 -*-
"""ensemble.py

Ensemble methods for structured data:
- Stacking (with cross-validated OOF predictions + meta-learner)
- Voting (weighted average of base model predictions)

Supports both classification and regression tasks.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, mean_squared_error
from scipy.optimize import minimize


# -------------------------------------------------------------------
# 1. Stacking ensemble
# -------------------------------------------------------------------

def train_stacking(
    base_models: List[Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta_learner: str = "logreg",
    n_folds: int = 5,
    task: str = "classification",
    random_state: int = 42,
) -> Tuple[Any, List[Any], np.ndarray]:
    """
    Train a stacking ensemble.

    Generates out-of-fold (OOF) predictions from base models using
    cross-validation, then trains a meta-learner on the OOF predictions.

    Parameters
    ----------
    base_models : list of fitted sklearn/xgboost/lightgbm models
        Each must support .predict() and optionally .predict_proba().
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    X_val : np.ndarray
        Validation features (used to evaluate the meta-learner).
    y_val : np.ndarray
        Validation targets.
    meta_learner : str
        'logreg' for LogisticRegression (classification) or Ridge (regression).
    n_folds : int
        Number of CV folds for generating OOF predictions.
    task : 'classification' or 'regression'
    random_state : int

    Returns
    -------
    meta_model : fitted meta-learner
    base_models : list of base models (unchanged, for use in predict_stacking)
    oof_preds : np.ndarray of OOF predictions used to train meta-learner
    """
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    n_samples = X_train.shape[0]
    n_base = len(base_models)

    print(f"\n[Stacking] Building OOF predictions with {n_folds} folds, {n_base} base models...")
    start_time = time.time()

    if task == "classification":
        # Determine number of classes
        n_classes = len(np.unique(y_train))
        use_proba = all(hasattr(m, "predict_proba") for m in base_models)

        if use_proba:
            # OOF: (n_samples, n_base * n_classes)
            oof_train = np.zeros((n_samples, n_base * n_classes))
            val_meta_features = np.zeros((X_val.shape[0], n_base * n_classes))
        else:
            oof_train = np.zeros((n_samples, n_base))
            val_meta_features = np.zeros((X_val.shape[0], n_base))

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        for fold_idx, (tr_idx, oof_idx) in enumerate(kf.split(X_train, y_train), 1):
            print(f"[Stacking] Fold {fold_idx}/{n_folds}")
            X_tr, X_oof = X_train[tr_idx], X_train[oof_idx]
            y_tr = y_train[tr_idx]

            for m_idx, model in enumerate(base_models):
                # Clone and refit the model on this fold
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_tr, y_tr)

                if use_proba:
                    start_col = m_idx * n_classes
                    end_col = start_col + n_classes
                    oof_train[oof_idx, start_col:end_col] = fold_model.predict_proba(X_oof)
                    val_meta_features[:, start_col:end_col] += fold_model.predict_proba(X_val) / n_folds
                else:
                    oof_train[oof_idx, m_idx] = fold_model.predict(X_oof)
                    val_meta_features[:, m_idx] += fold_model.predict(X_val) / n_folds

        # Train meta-learner
        print("[Stacking] Training meta-learner...")
        if meta_learner == "logreg":
            meta_model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            meta_model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1,
            )

        meta_model.fit(oof_train, y_train)

        # Evaluate on validation
        y_val_pred = meta_model.predict(val_meta_features)
        val_f1 = f1_score(y_val, y_val_pred, average="weighted")
        print(f"[Stacking] Meta-learner validation F1 (weighted): {val_f1:.4f}")

    else:
        # Regression
        oof_train = np.zeros((n_samples, n_base))
        val_meta_features = np.zeros((X_val.shape[0], n_base))

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        for fold_idx, (tr_idx, oof_idx) in enumerate(kf.split(X_train), 1):
            print(f"[Stacking] Fold {fold_idx}/{n_folds}")
            X_tr, X_oof = X_train[tr_idx], X_train[oof_idx]
            y_tr = y_train[tr_idx]

            for m_idx, model in enumerate(base_models):
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_tr, y_tr)
                oof_train[oof_idx, m_idx] = fold_model.predict(X_oof)
                val_meta_features[:, m_idx] += fold_model.predict(X_val) / n_folds

        # Train meta-learner (Ridge for regression)
        print("[Stacking] Training meta-learner (Ridge)...")
        meta_model = Ridge(alpha=1.0, random_state=random_state)
        meta_model.fit(oof_train, y_train)

        y_val_pred = meta_model.predict(val_meta_features)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        print(f"[Stacking] Meta-learner validation RMSE: {val_rmse:.4f}")

    total_time = time.time() - start_time
    print(f"[Stacking] Total time: {total_time:.2f}s")

    return meta_model, base_models, oof_train


# -------------------------------------------------------------------
# 2. Predict with stacking
# -------------------------------------------------------------------

def predict_stacking(
    meta_model,
    base_models: List[Any],
    X_test: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict using a stacking ensemble.

    Parameters
    ----------
    meta_model : fitted meta-learner
    base_models : list of fitted base models
    X_test : np.ndarray

    Returns
    -------
    y_pred : 1D array of predictions
    y_proba : 2D array of probabilities (classification) or None (regression)
    """
    n_base = len(base_models)
    use_proba = all(hasattr(m, "predict_proba") for m in base_models)

    if use_proba:
        # Get number of classes from first model
        proba_sample = base_models[0].predict_proba(X_test[:1])
        n_classes = proba_sample.shape[1]
        meta_features = np.zeros((X_test.shape[0], n_base * n_classes))

        for m_idx, model in enumerate(base_models):
            start_col = m_idx * n_classes
            end_col = start_col + n_classes
            meta_features[:, start_col:end_col] = model.predict_proba(X_test)
    else:
        meta_features = np.zeros((X_test.shape[0], n_base))
        for m_idx, model in enumerate(base_models):
            meta_features[:, m_idx] = model.predict(X_test)

    y_pred = meta_model.predict(meta_features)

    if hasattr(meta_model, "predict_proba"):
        y_proba = meta_model.predict_proba(meta_features)
    else:
        y_proba = None

    return y_pred, y_proba


# -------------------------------------------------------------------
# 3. Voting ensemble: optimize weights on validation
# -------------------------------------------------------------------

def train_voting(
    models: List[Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "classification",
) -> Tuple[np.ndarray]:
    """
    Optimize voting weights on validation set.

    For classification: optimize weights to maximize weighted F1.
    For regression: optimize weights to minimize RMSE.

    Parameters
    ----------
    models : list of fitted models
    X_val : np.ndarray
    y_val : np.ndarray
    task : 'classification' or 'regression'

    Returns
    -------
    weights : np.ndarray of shape (n_models,) summing to 1
    """
    y_val = np.asarray(y_val)
    n_models = len(models)

    print(f"\n[Voting] Optimizing weights for {n_models} models...")

    if task == "classification":
        # Collect probability predictions from all models
        all_probas = []
        for model in models:
            if hasattr(model, "predict_proba"):
                all_probas.append(model.predict_proba(X_val))
            else:
                # Fall back to hard predictions as one-hot
                preds = model.predict(X_val)
                n_classes = len(np.unique(y_val))
                one_hot = np.zeros((len(preds), n_classes))
                one_hot[np.arange(len(preds)), preds.astype(int)] = 1.0
                all_probas.append(one_hot)

        def neg_f1(w):
            """Negative weighted F1 to minimize."""
            w = np.abs(w)
            w = w / w.sum()
            blended = sum(w[i] * all_probas[i] for i in range(n_models))
            y_pred = np.argmax(blended, axis=1)
            return -f1_score(y_val, y_pred, average="weighted")

        # Initial uniform weights
        w0 = np.ones(n_models) / n_models
        result = minimize(neg_f1, w0, method="Nelder-Mead",
                          options={"maxiter": 1000, "xatol": 1e-6})

        weights = np.abs(result.x)
        weights = weights / weights.sum()

        # Report
        blended = sum(weights[i] * all_probas[i] for i in range(n_models))
        y_pred = np.argmax(blended, axis=1)
        final_f1 = f1_score(y_val, y_pred, average="weighted")
        print(f"[Voting] Optimized weights: {weights}")
        print(f"[Voting] Validation F1 (weighted): {final_f1:.4f}")

    else:
        # Regression
        all_preds = np.column_stack([model.predict(X_val) for model in models])

        def rmse_obj(w):
            """RMSE to minimize."""
            w = np.abs(w)
            w = w / w.sum()
            blended = all_preds @ w
            return np.sqrt(mean_squared_error(y_val, blended))

        w0 = np.ones(n_models) / n_models
        result = minimize(rmse_obj, w0, method="Nelder-Mead",
                          options={"maxiter": 1000, "xatol": 1e-6})

        weights = np.abs(result.x)
        weights = weights / weights.sum()

        blended = all_preds @ weights
        final_rmse = np.sqrt(mean_squared_error(y_val, blended))
        print(f"[Voting] Optimized weights: {weights}")
        print(f"[Voting] Validation RMSE: {final_rmse:.4f}")

    return weights


# -------------------------------------------------------------------
# 4. Predict with voting
# -------------------------------------------------------------------

def predict_voting(
    models: List[Any],
    weights: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict using a weighted voting ensemble.

    Parameters
    ----------
    models : list of fitted models
    weights : np.ndarray of shape (n_models,)
    X_test : np.ndarray

    Returns
    -------
    y_pred : 1D array of predictions
    y_proba : 2D array of blended probabilities (classification) or None
    """
    n_models = len(models)
    use_proba = all(hasattr(m, "predict_proba") for m in models)

    if use_proba:
        # Classification with probabilities
        all_probas = [model.predict_proba(X_test) for model in models]
        blended = sum(weights[i] * all_probas[i] for i in range(n_models))
        y_pred = np.argmax(blended, axis=1)
        y_proba = blended
    else:
        # Check if this is regression (no predict_proba at all)
        has_any_proba = any(hasattr(m, "predict_proba") for m in models)
        if not has_any_proba:
            # Regression
            all_preds = np.column_stack([model.predict(X_test) for model in models])
            y_pred = all_preds @ weights
            y_proba = None
        else:
            # Mixed: some have proba, some don't
            all_probas = []
            for model in models:
                if hasattr(model, "predict_proba"):
                    all_probas.append(model.predict_proba(X_test))
                else:
                    preds = model.predict(X_test)
                    n_classes = all_probas[0].shape[1] if all_probas else 2
                    one_hot = np.zeros((len(preds), n_classes))
                    one_hot[np.arange(len(preds)), preds.astype(int)] = 1.0
                    all_probas.append(one_hot)

            blended = sum(weights[i] * all_probas[i] for i in range(n_models))
            y_pred = np.argmax(blended, axis=1)
            y_proba = blended

    return y_pred, y_proba
