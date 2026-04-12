"""
dl_vit.py
=========
Vision Transformer (ViT-B/16) transfer learning for image classification.

Requires 224x224 input images. Supports feature extraction, fine-tuning,
and attention weight extraction for interpretability.
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
def build_vit(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    model_name: str = "vit_b_16",
) -> nn.Module:
    """Load a Vision Transformer and replace the classification head.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
    pretrained : bool
        Load ImageNet-pretrained weights.
    freeze_backbone : bool
        If ``True``, freeze all encoder parameters; only the new head
        is trained.
    model_name : str
        ViT variant.  Currently only ViT-B/16 is supported.  Both the
        torchvision-style name ``'vit_b_16'`` and the timm-style name
        ``'vit_base_patch16_224'`` are accepted.

    Returns
    -------
    nn.Module
    """
    # Accept both torchvision and timm naming conventions for ViT-B/16
    # so the config (timm style) and code default (torchvision style)
    # don't drift.
    _VIT_B_16_ALIASES = {"vit_b_16", "vit_base_patch16_224"}

    if model_name in _VIT_B_16_ALIASES:
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
    else:
        raise ValueError(
            f"Unsupported ViT model: {model_name!r}. "
            f"Supported aliases: {sorted(_VIT_B_16_ALIASES)}"
        )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classification head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    # Ensure new head is trainable
    for param in model.heads.head.parameters():
        param.requires_grad = True

    return model


# ===================================================================== #
#  Training wrapper                                                      #
# ===================================================================== #
def train_vit(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    mode: str = "fine_tune",
    num_epochs: int = 10,
    lr: float = 1e-5,
    patience: int = 3,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Build and train a Vision Transformer.

    Parameters
    ----------
    train_loader, val_loader : DataLoader
    num_classes : int
    mode : str
        ``'feature_extraction'`` — freeze backbone, lr=1e-3.
        ``'fine_tune'`` — unfreeze all, lr=1e-5 (lower for transformers).
    num_epochs, lr, patience : training hyper-parameters.
    device : optional torch.device

    Returns
    -------
    model, history
    """
    if device is None:
        device = get_device()

    if mode == "feature_extraction":
        model = build_vit(
            num_classes=num_classes, pretrained=True, freeze_backbone=True
        )
        effective_lr = 1e-3
    else:
        model = build_vit(
            num_classes=num_classes, pretrained=True, freeze_backbone=False
        )
        effective_lr = lr

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=effective_lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
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
def predict_vit(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference with a trained Vision Transformer.

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


# ===================================================================== #
#  Attention extraction                                                  #
# ===================================================================== #
def extract_attention_weights(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract attention maps from the last transformer block.

    Uses forward hooks to capture the attention weights produced by the
    multi-head self-attention layer of the final encoder block.

    Parameters
    ----------
    model : nn.Module
        A ViT-B/16 model (from :func:`build_vit` or torchvision).
    image_tensor : torch.Tensor
        Single image tensor of shape ``(1, 3, 224, 224)`` or
        ``(3, 224, 224)`` (batch dim added automatically).
    device : optional torch.device

    Returns
    -------
    attention : np.ndarray
        Attention map of shape ``(num_heads, seq_len, seq_len)`` from
        the last encoder block.  ``seq_len = 1 + num_patches`` where
        the first token is the [CLS] token.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    attention_store: dict[str, torch.Tensor] = {}

    # Locate the last encoder block's self-attention module
    last_block = model.encoder.layers[-1]
    attn_module = last_block.self_attention

    def _hook_fn(
        module: nn.Module,
        input_args: tuple,
        output: tuple,
    ) -> None:
        # nn.MultiheadAttention returns (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) >= 2:
            attention_store["attn"] = output[1].detach().cpu()

    # Register hook
    handle = attn_module.register_forward_hook(_hook_fn)

    # We need attn weights returned -- temporarily enable
    original_need_weights = True
    # torchvision ViT calls self_attention with need_weights=False by
    # default.  We monkey-patch the forward of the encoder layer to
    # pass need_weights=True so the hook can capture them.
    _orig_forward = last_block.forward

    def _patched_forward(input_tensor: torch.Tensor) -> torch.Tensor:
        """Patched forward that forces attention weight output."""
        if input_tensor.dim() != 3:
            raise ValueError(
                f"Expected (batch, seq, hidden) got {tuple(input_tensor.shape)}"
            )
        x = last_block.ln_1(input_tensor)
        # Call self_attention with need_weights=True
        attn_out, attn_weights = last_block.self_attention(
            x, x, x, need_weights=True
        )
        attention_store["attn"] = attn_weights.detach().cpu()
        x = last_block.dropout(attn_out)
        x = x + input_tensor
        y = last_block.ln_2(x)
        y = last_block.mlp(y)
        return x + y

    last_block.forward = _patched_forward  # type: ignore[assignment]

    try:
        with torch.no_grad():
            _ = model(image_tensor)
    finally:
        handle.remove()
        last_block.forward = _orig_forward  # type: ignore[assignment]

    if "attn" not in attention_store:
        raise RuntimeError(
            "Failed to capture attention weights. "
            "Ensure the model is a torchvision ViT-B/16."
        )

    # Shape: (batch, num_heads, seq_len, seq_len) or (batch, seq, seq)
    attn = attention_store["attn"]
    if attn.dim() == 4:
        attn = attn[0]  # drop batch dim -> (heads, seq, seq)
    elif attn.dim() == 3:
        attn = attn[0].unsqueeze(0)  # (1, seq, seq)

    return attn.numpy()
