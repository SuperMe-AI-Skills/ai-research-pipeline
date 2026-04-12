# dl_albert.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import itertools
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AlbertModel,
    AlbertConfig,
)


# --------------------------------------------------------------------
# 1. Dataset for ALBERT (+ optional tabular features)
# --------------------------------------------------------------------

class AlbertTabulateDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]],
        tokenizer: AutoTokenizer,
        max_len: int,
        tabular: Optional[np.ndarray] = None,
    ):
        self.texts = list(texts)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len

        if tabular is not None:
            self.tabular = np.asarray(tabular, dtype=np.float32)
        else:
            self.tabular = None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.tabular is not None:
            item["tabular"] = torch.tensor(self.tabular[idx], dtype=torch.float32)

        return item


# --------------------------------------------------------------------
# 2. ALBERTTabulate model
# --------------------------------------------------------------------

class AlbertTabulate(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
        tabular_dim: int = 0,
        freeze_base: bool = False,
    ):
        super().__init__()

        self.base_model = AlbertModel.from_pretrained(model_name_or_path)
        hidden_size = self.base_model.config.hidden_size

        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.tabular_dim = tabular_dim

        if hidden_dim is None:
            # Simple: [CLS] + tabular → num_classes
            self.fc = nn.Linear(hidden_size + tabular_dim, num_classes)
            self.use_hidden = False
        else:
            # [CLS] + tabular → hidden_dim → num_classes
            self.fc_hidden = nn.Linear(hidden_size + tabular_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, num_classes)
            self.activation = nn.ReLU()
            self.use_hidden = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = outputs.last_hidden_state[:, 0, :]  # [B, H]
        h = self.dropout(cls)

        if self.tabular_dim > 0 and tabular is not None:
            h = torch.cat([h, tabular], dim=1)

        if self.use_hidden:
            h = self.fc_hidden(h)
            h = self.activation(h)
            h = self.dropout(h)
            logits = self.fc_out(h)
        else:
            logits = self.fc(h)

        return logits


# --------------------------------------------------------------------
# 3. Training loop
# --------------------------------------------------------------------

def _get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_albert_tabulate(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    num_classes: int,
    max_len: int,
    model_name_or_path: str = "albert-base-v2",
    batch_size: int = 16,
    num_epochs: int = 3,
    lr: float = 2e-5,
    dropout: float = 0.1,
    hidden_dim: Optional[int] = None,
    train_tabular: Optional[np.ndarray] = None,
    val_tabular: Optional[np.ndarray] = None,
    random_state: int = 42,
    patience: Optional[int] = None,
    freeze_base: bool = False,
) -> Tuple[AlbertTabulate, Dict[str, float], AutoTokenizer, Dict[str, List[float]]]:
    """
    Train an ALBERT-based classifier with optional tabular features.
    """
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)

    device = _get_device(None)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    train_dataset = AlbertTabulateDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_len=max_len,
        tabular=train_tabular,
    )
    val_dataset = AlbertTabulateDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_len=max_len,
        tabular=val_tabular,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    tab_dim = 0 if train_tabular is None else train_tabular.shape[1]

    model = AlbertTabulate(
        model_name_or_path=model_name_or_path,
        num_classes=num_classes,
        dropout=dropout,
        hidden_dim=hidden_dim,
        tabular_dim=tab_dim,
        freeze_base=freeze_base,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    epochs_without_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_losses = []
        all_train_preds = []
        all_train_labels = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            tab = batch.get("tabular")
            if tab is not None:
                tab = tab.to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, tabular=tab)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            all_train_preds.append(preds.detach().cpu().numpy())
            all_train_labels.append(labels.detach().cpu().numpy())

        train_loss = float(np.mean(epoch_train_losses))
        all_train_preds = np.concatenate(all_train_preds)
        all_train_labels = np.concatenate(all_train_labels)
        train_f1 = f1_score(all_train_labels, all_train_preds, average="weighted")

        # Validation
        model.eval()
        epoch_val_losses = []
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                tab = batch.get("tabular")
                if tab is not None:
                    tab = tab.to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask, tabular=tab)
                loss = criterion(logits, labels)
                epoch_val_losses.append(loss.item())

                preds = torch.argmax(logits, dim=1)
                all_val_preds.append(preds.detach().cpu().numpy())
                all_val_labels.append(labels.detach().cpu().numpy())

        val_loss = float(np.mean(epoch_val_losses))
        all_val_preds = np.concatenate(all_val_preds)
        all_val_labels = np.concatenate(all_val_labels)
        val_f1 = f1_score(all_val_labels, all_val_preds, average="weighted")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if patience is not None and epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    metrics = {
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
    }

    return model, metrics, tokenizer, history


# --------------------------------------------------------------------
# 4. Prediction helpers
# --------------------------------------------------------------------

def predict_albert_tabulate(
    model: AlbertTabulate,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    max_len: int,
    batch_size: int = 32,
    tabular: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference with ALBERTTabulate and return (logits, preds).
    """
    device = _get_device(device)
    model = model.to(device)
    model.eval()

    dataset = AlbertTabulateDataset(
        texts=texts,
        labels=None,
        tokenizer=tokenizer,
        max_len=max_len,
        tabular=tabular,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            tab = batch.get("tabular")
            if tab is not None:
                tab = tab.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, tabular=tab)
            preds = torch.argmax(logits, dim=1)

            all_logits.append(logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    return all_logits, all_preds


def predict_albert(
    model: AlbertTabulate,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    max_len: int,
    batch_size: int = 32,
    tabular: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper so the notebook can import `predict_albert`.
    """
    return predict_albert_tabulate(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_len=max_len,
        batch_size=batch_size,
        tabular=tabular,
        device=device,
    )


# --------------------------------------------------------------------
# 5. Random search
# --------------------------------------------------------------------

def random_search_albert(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    num_classes: int,
    max_len: int,
    # canonical tabular args
    train_tabular: Optional[np.ndarray] = None,
    val_tabular: Optional[np.ndarray] = None,
    model_name_or_path: str = "albert-base-v2",
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    n_trials: int = 5,
    random_state: int = 42,
    # NEW: aliases so notebook can call train_extra / val_extra
    train_extra: Optional[np.ndarray] = None,
    val_extra: Optional[np.ndarray] = None,
) -> Tuple[
    AlbertTabulate,
    AutoTokenizer,
    Dict[str, Any],
    Dict[str, float],
    Dict[str, List[float]],
]:
    """
    Random search wrapper around train_albert_tabulate.

    Supports both:
      - train_tabular / val_tabular
      - train_extra / val_extra (aliases, for consistency with GRU/CNN)

    Returns
    -------
    best_model, best_tokenizer, best_params, best_metrics, best_history
    """
    # --- resolve aliases for tabular features ---
    # If train_tabular is not supplied but train_extra is, use train_extra
    if train_tabular is None and train_extra is not None:
        train_tabular = train_extra
    if val_tabular is None and val_extra is not None:
        val_tabular = val_extra

    if param_grid is None:
        param_grid = {
            "batch_size":  [16, 32],
            "num_epochs":  [3, 4],
            "lr":          [2e-5, 3e-5],
            "dropout":     [0.1, 0.2],
            "hidden_dim":  [None, 128],
            "freeze_base": [False],
            "patience":    [1, 2],
        }

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    all_combos = list(itertools.product(*values))

    rng = random.Random(random_state)
    if n_trials is not None and n_trials < len(all_combos):
        sampled_combos = rng.sample(all_combos, n_trials)
    else:
        sampled_combos = all_combos

    best_f1 = -1.0
    best_model: Optional[AlbertTabulate] = None
    best_tokenizer: Optional[AutoTokenizer] = None
    best_params: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_history: Optional[Dict[str, List[float]]] = None

    print(f"[ALBERT] Random search over {len(sampled_combos)} combinations.")

    for trial_idx, combo in enumerate(sampled_combos, 1):
        params = dict(zip(keys, combo))
        print(f"\n[ALBERT] Trial {trial_idx}/{len(sampled_combos)} params={params}")

        model, metrics, tokenizer, history = train_albert_tabulate(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            num_classes=num_classes,
            max_len=max_len,
            model_name_or_path=model_name_or_path,
            batch_size=params["batch_size"],
            num_epochs=params["num_epochs"],
            lr=params["lr"],
            dropout=params["dropout"],
            hidden_dim=params["hidden_dim"],
            train_tabular=train_tabular,
            val_tabular=val_tabular,
            random_state=random_state,
            patience=params["patience"],
            freeze_base=params["freeze_base"],
        )

        val_f1 = metrics.get("best_val_f1", None)
        if val_f1 is None:
            continue

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model
            best_tokenizer = tokenizer
            best_params = params
            best_metrics = metrics
            best_history = history
            print(f"[ALBERT] 🔹 New best val F1 = {best_f1:.4f}")

    if best_model is None:
        raise RuntimeError("random_search_albert did not find any successful configuration.")

    print(f"\n[ALBERT] ✅ Best val F1 = {best_f1:.4f} with params: {best_params}")

    return best_model, best_tokenizer, best_params, best_metrics, best_history