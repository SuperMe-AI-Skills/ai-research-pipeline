"""
dl_train_utils.py
=================
Shared deep-learning training utilities for all image classification models.

Provides early stopping, single-epoch train/evaluate helpers, a full
training loop, and device selection.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


# ===================================================================== #
#  Device helper                                                         #
# ===================================================================== #
def get_device() -> torch.device:
    """Return ``cuda`` device when available, otherwise ``cpu``."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ===================================================================== #
#  Early Stopping                                                        #
# ===================================================================== #
class EarlyStopping:
    """Track validation performance and stop when it stops improving.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs without improvement before stopping.
    mode : str
        ``'min'`` (e.g. loss) or ``'max'`` (e.g. accuracy / F1).
    delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 3,
        mode: str = "min",
        delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.delta = delta

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_model_state: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.delta
        return score > self.best_score + self.delta

    # ------------------------------------------------------------------ #
    def __call__(self, score: float, model: nn.Module) -> None:
        """Update state with the latest validation score.

        Parameters
        ----------
        score : float
            Current validation metric value.
        model : nn.Module
            Model whose state dict will be saved on improvement.
        """
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    # ------------------------------------------------------------------ #
    def restore_best(self, model: nn.Module) -> nn.Module:
        """Load the best model state back into *model* (in-place)."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model


# ===================================================================== #
#  Single-epoch helpers                                                  #
# ===================================================================== #
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run one training epoch.

    Returns
    -------
    avg_loss : float
    preds : np.ndarray  (N,)
    labels : np.ndarray (N,)
    """
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(targets.cpu().tolist())

    avg_loss = running_loss / len(train_loader.dataset)  # type: ignore[arg-type]
    return avg_loss, np.array(all_preds), np.array(all_labels)


# ------------------------------------------------------------------ #
@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Run one evaluation pass (no gradient computation).

    Returns
    -------
    avg_loss : float
    preds : np.ndarray  (N,)
    labels : np.ndarray (N,)
    logits : np.ndarray (N, C)
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_logits: List[np.ndarray] = []

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(targets.cpu().tolist())
        all_logits.append(outputs.cpu().numpy())

    avg_loss = running_loss / len(data_loader.dataset)  # type: ignore[arg-type]
    logits = np.concatenate(all_logits, axis=0)
    return avg_loss, np.array(all_preds), np.array(all_labels), logits


# ===================================================================== #
#  Full training loop                                                    #
# ===================================================================== #
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    num_epochs: int = 10,
    patience: int = 3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Full training loop with early stopping.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    criterion : nn.Module
    optimizer : Optimizer
    scheduler : optional LR scheduler (stepped after each epoch)
    num_epochs : int
    patience : int
        Early stopping patience (based on validation loss).
    device : optional torch.device
    verbose : bool
        Print per-epoch summary when ``True``.

    Returns
    -------
    model : nn.Module
        Model with best validation weights restored.
    history : dict
        Keys: ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``.
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    early = EarlyStopping(patience=patience, mode="min")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_preds, val_labels, _ = evaluate_one_epoch(
            model, val_loader, criterion, device
        )

        train_acc = float((train_preds == train_labels).mean())
        val_acc = float((val_preds == val_labels).mean())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            scheduler.step()

        if verbose:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            )

        early(val_loss, model)
        if early.early_stop:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}.")
            break

    model = early.restore_best(model)
    return model, history
