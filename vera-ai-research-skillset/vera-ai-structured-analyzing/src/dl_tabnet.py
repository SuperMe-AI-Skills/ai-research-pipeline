# -*- coding: utf-8 -*-
"""dl_tabnet.py

Simplified TabNet implementation in PyTorch (self-contained, no external dependency).

Implements:
  - TabNetModel: simplified TabNet with attention mechanism
  - train_tabnet: training loop with early stopping
  - predict_tabnet: inference helper
  - get_tabnet_feature_importance: attention-based importance scores
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------------------------------------------
# 0. Sparsemax activation (entmax-1.5 simplified)
# -------------------------------------------------------------------

def _sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax activation function.
    Projects input onto the probability simplex with sparse output.
    """
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)

    # Reshape k for broadcasting
    shape = [1] * z.ndim
    shape[dim] = -1
    k = k.view(shape)

    support = (1 + k * z_sorted - z_cumsum) > 0
    k_max = support.sum(dim=dim, keepdim=True).float()
    tau = (z_cumsum.gather(dim, (k_max - 1).long().clamp(min=0)) - 1) / k_max

    output = torch.clamp(z - tau, min=0)
    return output


# -------------------------------------------------------------------
# 1. GLU Block (shared/step-specific)
# -------------------------------------------------------------------

class _GLUBlock(nn.Module):
    """Gated Linear Unit block: FC -> BN -> GLU."""

    def __init__(self, input_dim: int, output_dim: int, virtual_batch_size: int = None):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2, bias=False)
        self.bn = nn.BatchNorm1d(output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


# -------------------------------------------------------------------
# 2. TabNet Model
# -------------------------------------------------------------------

class TabNetModel(nn.Module):
    """
    Simplified TabNet with attention mechanism.

    Architecture per step:
      - Shared layer (across all steps)
      - Step-specific layer
      - Sparsemax attention mask for feature selection
      - Aggregate selected features via attention

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output units (n_classes for classification, 1 for regression).
    n_steps : int
        Number of sequential attention steps.
    n_d : int
        Dimension of the decision layer at each step.
    n_a : int
        Dimension of the attention layer at each step.
    gamma : float
        Coefficient for attention entropy penalty.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_steps: int = 3,
        n_d: int = 8,
        n_a: int = 8,
        gamma: float = 1.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.n_d = n_d
        self.n_a = n_a
        self.gamma = gamma

        # Initial batch normalization
        self.initial_bn = nn.BatchNorm1d(input_dim)

        # Shared layer across all steps
        self.shared_layer = _GLUBlock(input_dim, n_d + n_a)

        # Step-specific layers
        self.step_layers = nn.ModuleList([
            _GLUBlock(n_d + n_a, n_d + n_a) for _ in range(n_steps)
        ])

        # Attention layers (one per step)
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_a, input_dim, bias=False),
                nn.BatchNorm1d(input_dim),
            )
            for _ in range(n_steps)
        ])

        # Final output layer
        self.final_fc = nn.Linear(n_d, output_dim)

        # Store attention masks for interpretability
        self._attention_masks: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)

        Returns
        -------
        logits : torch.Tensor of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)
        x = self.initial_bn(x)

        # Initialize
        prior_scales = torch.ones(batch_size, self.input_dim, device=x.device)
        aggregated_output = torch.zeros(batch_size, self.n_d, device=x.device)
        complementary_aggregate = torch.zeros(
            batch_size, self.input_dim, device=x.device
        )

        self._attention_masks = []

        for step in range(self.n_steps):
            # Shared transformation
            h = self.shared_layer(x)

            # Step-specific transformation
            h = self.step_layers[step](h)

            # Split into decision (n_d) and attention (n_a) parts
            h_d = h[:, : self.n_d]
            h_a = h[:, self.n_d :]

            # Compute attention mask
            a = self.attention_layers[step](h_a)
            a = a * prior_scales
            a = _sparsemax(a, dim=-1)

            self._attention_masks.append(a.detach())

            # Update prior scales
            prior_scales = prior_scales * (self.gamma - a)

            # Masked features -> aggregate
            masked_x = a * x
            aggregated_output = aggregated_output + h_d

            # Track complementary aggregate for interpretability
            complementary_aggregate = complementary_aggregate + a

        # Final output
        logits = self.final_fc(aggregated_output)
        return logits

    def get_attention_masks(self) -> List[torch.Tensor]:
        """Return stored attention masks from the last forward pass."""
        return self._attention_masks


# -------------------------------------------------------------------
# 3. Training loop
# -------------------------------------------------------------------

def train_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "classification",
    n_steps: int = 3,
    n_d: int = 8,
    n_a: int = 8,
    gamma: float = 1.3,
    lr: float = 0.02,
    batch_size: int = 256,
    num_epochs: int = 50,
    patience: int = 5,
    device: Optional[str] = None,
) -> Tuple[TabNetModel, Dict[str, Any], Dict[str, list]]:
    """
    Train a TabNet model on tabular data with early stopping.

    Parameters
    ----------
    X_train, X_val : np.ndarray
    y_train, y_val : np.ndarray
    task : 'classification' or 'regression'
    n_steps, n_d, n_a, gamma : TabNet hyperparameters
    lr : learning rate
    batch_size : mini-batch size
    num_epochs : maximum epochs
    patience : early stopping patience
    device : 'cuda', 'mps', 'cpu', or None (auto-detect)

    Returns
    -------
    model : trained TabNetModel
    metrics : dict with best epoch info
    history : dict with lists of train_loss, val_loss per epoch
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    print(f"[TabNet] Using device: {device}")

    # Prepare data
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    if task == "classification":
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        num_classes = len(np.unique(y_train))
        output_dim = num_classes
        criterion = nn.CrossEntropyLoss()
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        output_dim = 1
        criterion = nn.MSELoss()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build model
    input_dim = X_train.shape[1]
    model = TabNetModel(
        input_dim=input_dim,
        output_dim=output_dim,
        n_steps=n_steps,
        n_d=n_d,
        n_a=n_a,
        gamma=gamma,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)

            if task == "regression":
                output = output.squeeze(-1)

            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(X_batch)
                if task == "regression":
                    output = output.squeeze(-1)
                loss = criterion(output, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"[TabNet] Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[TabNet] Early stopping at epoch {epoch} (patience={patience})")
                break

    total_time = time.time() - start_time

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_time": total_time,
        "n_steps": n_steps,
        "n_d": n_d,
        "n_a": n_a,
        "gamma": gamma,
        "lr": lr,
        "batch_size": batch_size,
    }

    print(f"\n[TabNet] Training complete.")
    print(f"[TabNet] Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    print(f"[TabNet] Total time: {total_time:.2f}s")

    return model, metrics, history


# -------------------------------------------------------------------
# 4. Prediction helper
# -------------------------------------------------------------------

def predict_tabnet(
    model: TabNetModel,
    X_test: np.ndarray,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on test data.

    Returns
    -------
    For classification:
        logits : np.ndarray of shape (n_samples, n_classes)
        preds : np.ndarray of shape (n_samples,) -- argmax labels
    For regression:
        predictions : np.ndarray of shape (n_samples,)
        predictions : same array repeated
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_outputs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            output = model(batch)
            all_outputs.append(output.cpu().numpy())

    outputs = np.concatenate(all_outputs, axis=0)

    if outputs.ndim == 2 and outputs.shape[1] > 1:
        # Classification
        preds = np.argmax(outputs, axis=1)
        return outputs, preds
    else:
        # Regression
        if outputs.ndim == 2:
            outputs = outputs.squeeze(-1)
        return outputs, outputs


# -------------------------------------------------------------------
# 5. Feature importance from attention masks
# -------------------------------------------------------------------

def get_tabnet_feature_importance(
    model: TabNetModel,
    X: np.ndarray,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute feature importance from TabNet attention masks.

    Runs a forward pass to collect attention masks, then aggregates
    across steps and samples.

    Parameters
    ----------
    model : trained TabNetModel
    X : np.ndarray of shape (n_samples, n_features)
    batch_size : batch size for inference
    device : device string or None

    Returns
    -------
    importance : np.ndarray of shape (n_features,)
        Normalized importance scores (sum to 1).
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_masks: List[np.ndarray] = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            _ = model(batch)  # trigger forward to populate attention masks
            masks = model.get_attention_masks()
            # Sum across steps for each sample
            step_sum = sum(m.cpu().numpy() for m in masks)  # (batch, n_features)
            all_masks.append(step_sum)

    all_masks_arr = np.concatenate(all_masks, axis=0)  # (n_samples, n_features)

    # Average across samples
    importance = np.mean(all_masks_arr, axis=0)

    # Normalize to sum to 1
    total = importance.sum()
    if total > 0:
        importance = importance / total

    return importance
