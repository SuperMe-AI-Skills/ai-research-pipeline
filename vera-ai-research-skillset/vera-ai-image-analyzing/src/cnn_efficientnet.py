"""
cnn_efficientnet.py
===================
EfficientNet-B0 transfer learning for image classification.

Supports feature extraction (frozen backbone) and full fine-tuning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from .dl_train_utils import (
    get_device,
    train_model,
)


# ===================================================================== #
#  Builder                                                               #
# ===================================================================== #
def build_efficientnet(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    pretrained_weights: str = "IMAGENET1K_V1",
) -> nn.Module:
    """Load EfficientNet-B0 and replace the classifier head.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
    pretrained : bool
        Load ImageNet-pretrained weights.
    freeze_backbone : bool
        If ``True``, freeze all parameters except the new classifier head.
    pretrained_weights : str
        Which ImageNet weights checkpoint to load when ``pretrained`` is
        True.  Default ``"IMAGENET1K_V1"`` matches config.

    Returns
    -------
    nn.Module
    """
    if pretrained:
        try:
            weights = getattr(models.EfficientNet_B0_Weights, pretrained_weights)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown EfficientNet_B0 weights variant: {pretrained_weights!r}."
            ) from exc
    else:
        weights = None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier: Sequential(Dropout, Linear)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    # Ensure new head is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


# ===================================================================== #
#  Training wrapper                                                      #
# ===================================================================== #
def train_efficientnet(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    mode: str = "fine_tune",
    num_epochs: int = 10,
    lr: float = 1e-4,
    patience: int = 3,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Build and train an EfficientNet-B0.

    Parameters
    ----------
    train_loader, val_loader : DataLoader
    num_classes : int
    mode : str
        ``'feature_extraction'`` — freeze backbone, lr=1e-3.
        ``'fine_tune'`` — unfreeze all, lr=1e-4.
    num_epochs, lr, patience : training hyper-parameters.
    device : optional torch.device

    Returns
    -------
    model, history
    """
    if device is None:
        device = get_device()

    if mode == "feature_extraction":
        model = build_efficientnet(
            num_classes=num_classes, pretrained=True, freeze_backbone=True
        )
        effective_lr = 1e-3
    else:
        model = build_efficientnet(
            num_classes=num_classes, pretrained=True, freeze_backbone=False
        )
        effective_lr = lr

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=effective_lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
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
def predict_efficientnet(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference with a trained EfficientNet-B0.

    Returns
    -------
    logits : np.ndarray  (N, C)
    preds : np.ndarray   (N,)
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
