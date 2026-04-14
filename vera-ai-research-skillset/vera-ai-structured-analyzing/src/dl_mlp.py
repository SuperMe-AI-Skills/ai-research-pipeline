# -*- coding: utf-8 -*-
"""dl_mlp.py

PyTorch MLP for tabular data.

Implements:
  - TabularDataset: PyTorch Dataset for features + labels
  - MLPClassifier: configurable MLP with BatchNorm + Dropout
  - MLPRegressor: single-output variant
  - train_mlp: training loop with early stopping
  - predict_mlp: inference helper
  - random_search_mlp: random hyperparameter search
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------------------------
# 1. Dataset
# -------------------------------------------------------------------

class TabularDataset(Dataset):
    """
    Simple PyTorch Dataset for tabular features + labels.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    task : 'classification' or 'regression'
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, task: str = "classification"):
        self.X = torch.tensor(X, dtype=torch.float32)
        if task == "classification":
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)
        self.task = task

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# -------------------------------------------------------------------
# 2. MLP Classifier
# -------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for classification.

    Architecture: Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Output

    Parameters
    ----------
    input_dim : int
        Number of input features.
    num_classes : int
        Number of output classes.
    hidden_dims : list of int
        Hidden layer sizes (default: [256, 128, 64]).
    dropout : float
        Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (pre-softmax)."""
        return self.network(x)


# -------------------------------------------------------------------
# 3. MLP Regressor
# -------------------------------------------------------------------

class MLPRegressor(nn.Module):
    """
    Multi-layer perceptron for regression.

    Architecture: Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Linear(1)

    Parameters
    ----------
    input_dim : int
    hidden_dims : list of int
    dropout : float
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scalar prediction."""
        return self.network(x).squeeze(-1)


# -------------------------------------------------------------------
# 4. Training loop
# -------------------------------------------------------------------

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "classification",
    hidden_dims: List[int] = None,
    dropout: float = 0.3,
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    device: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, list]]:
    """
    Train an MLP on tabular data with early stopping.

    Parameters
    ----------
    X_train, X_val : np.ndarray
    y_train, y_val : np.ndarray
    task : 'classification' or 'regression'
    hidden_dims : list of hidden layer sizes
    dropout : dropout probability
    batch_size : mini-batch size
    num_epochs : maximum epochs
    lr : learning rate
    patience : early stopping patience
    device : 'cuda', 'mps', 'cpu', or None (auto-detect)

    Returns
    -------
    model : trained nn.Module
    metrics : dict with best epoch info
    history : dict with lists of train_loss, val_loss per epoch
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)
    print(f"[MLP] Using device: {device}")

    # Build datasets and loaders
    train_ds = TabularDataset(X_train, y_train, task=task)
    val_ds = TabularDataset(X_val, y_val, task=task)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build model
    input_dim = X_train.shape[1]
    if task == "classification":
        num_classes = len(np.unique(y_train))
        model = MLPClassifier(input_dim, num_classes, hidden_dims, dropout).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = MLPRegressor(input_dim, hidden_dims, dropout).to(device)
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
                loss = criterion(output, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"[MLP] Epoch {epoch}/{num_epochs} | "
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
                print(f"[MLP] Early stopping at epoch {epoch} (patience={patience})")
                break

    total_time = time.time() - start_time

    # Restore best model. best_state was saved as CPU clones (line above)
    # so we explicitly map back to the training device before loading to
    # avoid relying on PyTorch's implicit cross-device copy semantics.
    if best_state is not None:
        best_state_on_device = {k: v.to(device) for k, v in best_state.items()}
        model.load_state_dict(best_state_on_device)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_time": total_time,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
    }

    print(f"\n[MLP] Training complete.")
    print(f"[MLP] Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    print(f"[MLP] Total time: {total_time:.2f}s")

    return model, metrics, history


# -------------------------------------------------------------------
# 5. Prediction helper
# -------------------------------------------------------------------

def predict_mlp(
    model: nn.Module,
    X_test: np.ndarray,
    batch_size: int = 64,
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
        predictions : same array repeated (for uniform API)
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
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_outputs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            output = model(batch)
            all_outputs.append(output.cpu().numpy())

    outputs = np.concatenate(all_outputs, axis=0)

    if outputs.ndim == 2 and outputs.shape[1] > 1:
        # Classification: logits
        preds = np.argmax(outputs, axis=1)
        return outputs, preds
    else:
        # Regression: scalar predictions
        if outputs.ndim == 2:
            outputs = outputs.squeeze(-1)
        return outputs, outputs


# -------------------------------------------------------------------
# 6. Random search
# -------------------------------------------------------------------

def random_search_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "classification",
    n_trials: int = 10,
    hidden_dims_options: Optional[List[List[int]]] = None,
    dropout_options: Optional[List[float]] = None,
    lr_options: Optional[List[float]] = None,
    batch_size_options: Optional[List[int]] = None,
    num_epochs: int = 50,
    patience: int = 5,
    device: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, list]]:
    """
    Random search over MLP hyperparameters.

    Parameters
    ----------
    n_trials : number of random configurations to try
    hidden_dims_options : list of hidden_dims configs
    dropout_options : list of dropout values
    lr_options : list of learning rates
    batch_size_options : list of batch sizes

    Returns
    -------
    best_model, best_metrics, best_history
    """
    if hidden_dims_options is None:
        hidden_dims_options = [
            [256, 128, 64],
            [512, 256, 128],
            [128, 64],
            [256, 128],
            [512, 256, 128, 64],
        ]
    if dropout_options is None:
        dropout_options = [0.1, 0.2, 0.3, 0.4, 0.5]
    if lr_options is None:
        lr_options = [1e-4, 5e-4, 1e-3, 5e-3]
    if batch_size_options is None:
        batch_size_options = [32, 64, 128, 256]

    rng = random.Random(random_state)

    best_val_loss = float("inf")
    best_model = None
    best_metrics: Dict[str, Any] = {}
    best_history: Dict[str, list] = {}

    print(f"\n[MLP RandomSearch] Running {n_trials} trials...")

    for trial in range(1, n_trials + 1):
        hd = rng.choice(hidden_dims_options)
        dp = rng.choice(dropout_options)
        lr = rng.choice(lr_options)
        bs = rng.choice(batch_size_options)

        print(f"\n[MLP RandomSearch] Trial {trial}/{n_trials}: "
              f"hidden_dims={hd}, dropout={dp}, lr={lr}, batch_size={bs}")

        try:
            model, metrics, history = train_mlp(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                task=task,
                hidden_dims=hd,
                dropout=dp,
                batch_size=bs,
                num_epochs=num_epochs,
                lr=lr,
                patience=patience,
                device=device,
            )

            if metrics["best_val_loss"] < best_val_loss:
                best_val_loss = metrics["best_val_loss"]
                best_model = model
                best_metrics = metrics
                best_history = history
                print(f"[MLP RandomSearch] New best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"[MLP RandomSearch] Trial {trial} failed: {e}")
            continue

    if best_model is None:
        raise RuntimeError("No valid MLP model was found during random search.")

    print(f"\n[MLP RandomSearch] Best config: {best_metrics}")
    return best_model, best_metrics, best_history
