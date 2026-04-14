"""
baseline_cnn.py
===============
Simple three-layer CNN baseline for image classification.

Architecture:
    Conv2d(3,32,3) -> ReLU -> MaxPool
    Conv2d(32,64,3) -> ReLU -> MaxPool
    Conv2d(64,128,3) -> ReLU -> AdaptiveAvgPool2d(1)
    Flatten -> Dropout -> Linear(128, num_classes)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dl_train_utils import (
    evaluate_one_epoch,
    get_device,
    train_model,
)


# ===================================================================== #
#  Model                                                                 #
# ===================================================================== #
class SimpleCNN(nn.Module):
    """A lightweight three-block convolutional network.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout probability before the final linear layer.
    """

    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===================================================================== #
#  Builder                                                               #
# ===================================================================== #
def build_simple_cnn(
    num_classes: int,
    dropout: float = 0.5,
) -> SimpleCNN:
    """Instantiate a :class:`SimpleCNN`.

    Parameters
    ----------
    num_classes : int
    dropout : float

    Returns
    -------
    SimpleCNN
    """
    return SimpleCNN(num_classes=num_classes, dropout=dropout)


# ===================================================================== #
#  Training wrapper                                                      #
# ===================================================================== #
def train_simple_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    num_epochs: int = 10,
    lr: float = 0.001,
    patience: int = 3,
    device: Optional[torch.device] = None,
) -> Tuple[SimpleCNN, Dict[str, List[float]]]:
    """Build and train a :class:`SimpleCNN`.

    Uses Adam optimiser with cross-entropy loss.

    Parameters
    ----------
    train_loader, val_loader : DataLoader
    num_classes : int
    num_epochs : int
    lr : float
    patience : int
    device : optional torch.device

    Returns
    -------
    model : SimpleCNN
        Trained model with best validation weights restored.
    history : dict
        Training history (losses, accuracies per epoch).
    """
    if device is None:
        device = get_device()

    model = build_simple_cnn(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        patience=patience,
        device=device,
        verbose=True,
    )
    return model, history


# ===================================================================== #
#  Prediction                                                            #
# ===================================================================== #
@torch.no_grad()
def predict_simple_cnn(
    model: SimpleCNN,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference with a trained :class:`SimpleCNN`.

    Parameters
    ----------
    model : SimpleCNN
    data_loader : DataLoader
    device : optional torch.device

    Returns
    -------
    logits : np.ndarray  shape (N, C)
    preds : np.ndarray   shape (N,)
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    all_logits: list[np.ndarray] = []
    for images, _ in data_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        all_logits.append(outputs.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    preds = logits.argmax(axis=1)
    return logits, preds
