"""
ensemble.py
===========
Ensemble methods for combining multiple image classification models.

Provides:
  - :func:`predict_all_models` — batch inference across several models.
  - :func:`train_soft_voting` — optimise blending weights on validation data.
  - :func:`predict_soft_voting` — weighted probability averaging.
  - :func:`train_stacking` — train a logistic-regression meta-learner.
  - :func:`predict_stacking` — predict with the stacking ensemble.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .dl_train_utils import get_device


# ===================================================================== #
#  Helpers                                                               #
# ===================================================================== #
def _logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


# ===================================================================== #
#  Predict with all models                                               #
# ===================================================================== #
@torch.no_grad()
def predict_all_models(
    models: Dict[str, nn.Module],
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Run every model on the data and collect logits.

    Parameters
    ----------
    models : dict
        ``{model_name: nn.Module}`` mapping.
    data_loader : DataLoader
    device : optional torch.device

    Returns
    -------
    dict
        ``{model_name: logits}`` where each ``logits`` array has
        shape ``(N, C)``.
    """
    if device is None:
        device = get_device()

    results: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        model = model.to(device)
        model.eval()

        all_logits: list[np.ndarray] = []
        for images, _ in data_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            all_logits.append(outputs.cpu().numpy())

        results[name] = np.concatenate(all_logits, axis=0)

    return results


# ===================================================================== #
#  Soft voting — optimise weights                                        #
# ===================================================================== #
def train_soft_voting(
    model_logits_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    method: str = "optimize",
) -> Dict[str, float]:
    """Find optimal blending weights for soft voting.

    Parameters
    ----------
    model_logits_dict : dict
        ``{model_name: logits (N, C)}``.
    y_true : np.ndarray  (N,)
        True integer labels.
    method : str
        ``'optimize'`` uses ``scipy.optimize.minimize`` to maximise
        weighted-F1.  ``'uniform'`` returns equal weights.

    Returns
    -------
    dict
        ``{model_name: weight}`` with weights summing to 1.
    """
    names = list(model_logits_dict.keys())
    n_models = len(names)

    # Convert logits to probabilities for blending
    proba_list = [
        _logits_to_probs(model_logits_dict[n]) for n in names
    ]
    y_true = np.asarray(y_true)

    if method == "uniform" or n_models == 1:
        return {n: 1.0 / n_models for n in names}

    # Objective: minimise negative weighted F1
    def _objective(w: np.ndarray) -> float:
        # Normalise weights to sum to 1
        w_norm = np.abs(w) / (np.abs(w).sum() + 1e-12)
        blended = sum(w_norm[i] * proba_list[i] for i in range(n_models))
        y_pred = blended.argmax(axis=1)
        return -f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

    x0 = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models

    result = minimize(
        _objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200},
    )

    w_opt = np.abs(result.x)
    w_opt = w_opt / (w_opt.sum() + 1e-12)

    return {n: float(w_opt[i]) for i, n in enumerate(names)}


# ===================================================================== #
#  Soft voting — predict                                                 #
# ===================================================================== #
def predict_soft_voting(
    model_logits_dict: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted average of class probabilities.

    Parameters
    ----------
    model_logits_dict : dict
        ``{model_name: logits (N, C)}``.
    weights : dict
        ``{model_name: weight}``.

    Returns
    -------
    y_pred : np.ndarray (N,)
    y_proba : np.ndarray (N, C)
    """
    names = list(model_logits_dict.keys())
    proba_list = [
        _logits_to_probs(model_logits_dict[n]) for n in names
    ]

    blended = np.zeros_like(proba_list[0])
    for i, n in enumerate(names):
        blended += weights.get(n, 0.0) * proba_list[i]

    y_pred = blended.argmax(axis=1)
    return y_pred, blended


# ===================================================================== #
#  Stacking — train meta-learner                                         #
# ===================================================================== #
def train_stacking(
    model_logits_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    meta_learner: str = "logreg",
) -> Tuple[Any, List[str]]:
    """Train a stacking meta-learner on concatenated probabilities.

    Parameters
    ----------
    model_logits_dict : dict
        ``{model_name: logits (N, C)}``.
    y_true : np.ndarray  (N,)
    meta_learner : str
        Currently only ``'logreg'`` (logistic regression) is supported.

    Returns
    -------
    meta_model : fitted sklearn estimator
    feature_order : list of str
        Ordered model names used to build the feature matrix (needed
        for consistent prediction).
    """
    feature_order = sorted(model_logits_dict.keys())
    y_true = np.asarray(y_true)

    # Build meta-feature matrix: concatenate probabilities from each model
    meta_features = np.concatenate(
        [_logits_to_probs(model_logits_dict[n]) for n in feature_order],
        axis=1,
    )

    if meta_learner == "logreg":
        clf = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
        )
    else:
        raise ValueError(
            f"Unsupported meta_learner '{meta_learner}'. Use 'logreg'."
        )

    clf.fit(meta_features, y_true)
    return clf, feature_order


# ===================================================================== #
#  Stacking — predict                                                    #
# ===================================================================== #
def predict_stacking(
    meta_model: Any,
    model_logits_dict: Dict[str, np.ndarray],
    feature_order: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with a trained stacking ensemble.

    Parameters
    ----------
    meta_model : fitted sklearn estimator
    model_logits_dict : dict
        ``{model_name: logits (N, C)}``.
    feature_order : list of str
        Same order returned by :func:`train_stacking`.

    Returns
    -------
    y_pred : np.ndarray (N,)
    y_proba : np.ndarray (N, C)
    """
    meta_features = np.concatenate(
        [_logits_to_probs(model_logits_dict[n]) for n in feature_order],
        axis=1,
    )

    y_pred = meta_model.predict(meta_features)
    y_proba = meta_model.predict_proba(meta_features)
    return y_pred, y_proba
