# dl_cnn.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import itertools
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# --------------------------------------------------------------------
# 1. Vocab & text → indices (same logic as GRU)
# --------------------------------------------------------------------

def build_vocab_cnn(
    texts: Sequence[str],
    vocab_size: int,
) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        if not isinstance(t, str):
            continue
        tokens = t.split()
        counter.update(tokens)

    most_common = [w for w, _ in counter.most_common(max(vocab_size - 2, 0))]

    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    idx = 2
    for w in most_common:
        if w not in word_to_idx:
            word_to_idx[w] = idx
            idx += 1

    return word_to_idx


def texts_to_padded_indices(
    texts: Sequence[str],
    word_to_idx: Dict[str, int],
    max_len: int,
) -> np.ndarray:
    n = len(texts)
    arr = np.zeros((n, max_len), dtype=np.int64)

    unk_idx = word_to_idx.get(UNK_TOKEN, 1)

    for i, t in enumerate(texts):
        if not isinstance(t, str):
            tokens = []
        else:
            tokens = t.split()

        ids = [word_to_idx.get(tok, unk_idx) for tok in tokens[:max_len]]
        arr[i, : len(ids)] = ids

    return arr


# --------------------------------------------------------------------
# 2. Dataset (same pattern as GRU)
# --------------------------------------------------------------------

class CNNDataset(Dataset):
    def __init__(
        self,
        input_ids: np.ndarray,
        labels: np.ndarray,
        extra_features: Optional[np.ndarray] = None,
    ):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

        if extra_features is not None:
            self.extra = torch.tensor(extra_features, dtype=torch.float32)
        else:
            self.extra = None

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        item = {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }
        if self.extra is not None:
            item["extra"] = self.extra[idx]
        return item


# --------------------------------------------------------------------
# 3. TextCNN model
# --------------------------------------------------------------------

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: Sequence[int],
        num_classes: int,
        dropout: float = 0.5,
        extra_feature_dim: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filters,
                    kernel_size=(fs, embedding_dim),
                )
                for fs in filter_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.extra_feature_dim = extra_feature_dim

        cnn_output_dim = num_filters * len(filter_sizes)
        fc_input_dim = cnn_output_dim + extra_feature_dim

        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        extra_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
        extra_features: [B, F] or None
        """
        emb = self.embedding(input_ids)      # [B, T, E]
        emb = emb.unsqueeze(1)               # [B, 1, T, E]

        conv_outputs = []
        for conv in self.convs:
            x = torch.relu(conv(emb))        # [B, num_filters, T - fs + 1, 1]
            x = x.squeeze(3)                # [B, num_filters, T - fs + 1]
            x = torch.max(x, dim=2)[0]      # [B, num_filters]
            conv_outputs.append(x)

        h = torch.cat(conv_outputs, dim=1)   # [B, num_filters * len(filter_sizes)]
        h = self.dropout(h)

        if self.extra_feature_dim > 0 and extra_features is not None:
            h = torch.cat([h, extra_features], dim=1)

        logits = self.fc(h)
        return logits


# --------------------------------------------------------------------
# 4. Training
# --------------------------------------------------------------------

def _get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn_model(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    vocab_size: int,
    max_len: int,
    num_classes: int,
    embedding_dim: int = 128,
    num_filters: int = 128,
    filter_sizes: Sequence[int] = (3, 4, 5),
    dropout: float = 0.5,
    batch_size: int = 32,
    num_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    train_extra: Optional[np.ndarray] = None,
    val_extra: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    random_state: int = 42,
    patience: Optional[int] = None,
) -> Tuple[TextCNN, Dict[str, float], Dict[str, int], Dict[str, List[float]]]:
    """
    Train a CNN-based text classifier (optionally + extra features).
    """
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)

    device = _get_device(device)

    word_to_idx = build_vocab_cnn(train_texts, vocab_size=vocab_size)
    vocab_len = len(word_to_idx)

    X_train_ids = texts_to_padded_indices(train_texts, word_to_idx, max_len=max_len)
    X_val_ids   = texts_to_padded_indices(val_texts,   word_to_idx, max_len=max_len)

    y_train = np.asarray(train_labels, dtype=np.int64)
    y_val   = np.asarray(val_labels, dtype=np.int64)

    train_dataset = CNNDataset(X_train_ids, y_train, extra_features=train_extra)
    val_dataset   = CNNDataset(X_val_ids,   y_val,   extra_features=val_extra)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    extra_dim = 0 if train_extra is None else train_extra.shape[1]

    model = TextCNN(
        vocab_size=vocab_len,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        filter_sizes=filter_sizes,
        num_classes=num_classes,
        dropout=dropout,
        extra_feature_dim=extra_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

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
            labels = batch["labels"].to(device)

            extra = batch.get("extra")
            if extra is not None:
                extra = extra.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, extra_features=extra)
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
                labels = batch["labels"].to(device)

                extra = batch.get("extra")
                if extra is not None:
                    extra = extra.to(device)

                logits = model(input_ids, extra_features=extra)
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

    return model, metrics, word_to_idx, history


# --------------------------------------------------------------------
# 5. Prediction helper
# --------------------------------------------------------------------

def predict_cnn(
    model: TextCNN,
    texts: Sequence[str],
    word_to_idx: Dict[str, int],
    max_len: int,
    batch_size: int = 32,
    extra_features: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference and return (logits, predicted_labels).
    """
    device = _get_device(device)
    model = model.to(device)
    model.eval()

    X_ids = texts_to_padded_indices(texts, word_to_idx, max_len=max_len)
    dummy_labels = np.zeros(len(texts), dtype=np.int64)

    dataset = CNNDataset(X_ids, dummy_labels, extra_features=extra_features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            extra = batch.get("extra")
            if extra is not None:
                extra = extra.to(device)

            logits = model(input_ids, extra_features=extra)
            preds = torch.argmax(logits, dim=1)

            all_logits.append(logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    return all_logits, all_preds


# --------------------------------------------------------------------
# 6. Random search
# --------------------------------------------------------------------

def random_search_cnn(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    num_classes: int,
    vocab_size: int,
    max_len: int,
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    n_trials: int = 5,
    train_extra: Optional[np.ndarray] = None,
    val_extra: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    random_state: int = 42,
) -> Tuple[TextCNN, Dict[str, int], Dict[str, Any], Dict[str, float], Dict[str, List[float]]]:
    """
    Random search wrapper around train_cnn_model.
    """
    if param_grid is None:
        param_grid = {
            "embedding_dim": [128],
            "num_filters": [100, 200],
            "filter_sizes": [(3, 4, 5)],
            "dropout": [0.5],
            "batch_size": [32],
            "num_epochs": [5],
            "lr": [1e-3],
            "weight_decay": [0.0],
            "patience": [2],
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
    best_model: Optional[TextCNN] = None
    best_vocab: Optional[Dict[str, int]] = None
    best_history: Optional[Dict[str, List[float]]] = None
    best_params: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, float]] = None

    print(f"[CNN] Random search over {len(sampled_combos)} combinations.")

    for trial_idx, combo in enumerate(sampled_combos, 1):
        params = dict(zip(keys, combo))
        print(f"\n[CNN] Trial {trial_idx}/{len(sampled_combos)} params={params}")

        model, metrics, vocab, history = train_cnn_model(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            vocab_size=vocab_size,
            max_len=max_len,
            num_classes=num_classes,
            embedding_dim=params["embedding_dim"],
            num_filters=params["num_filters"],
            filter_sizes=params["filter_sizes"],
            dropout=params["dropout"],
            batch_size=params["batch_size"],
            num_epochs=params["num_epochs"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            train_extra=train_extra,
            val_extra=val_extra,
            device=device,
            random_state=random_state,
            patience=params["patience"],
        )

        val_f1 = metrics.get("best_val_f1", None)
        if val_f1 is None:
            continue

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model
            best_vocab = vocab
            best_history = history
            best_params = params
            best_metrics = metrics
            print(f"[CNN] 🔹 New best val F1 = {best_f1:.4f}")

    if best_model is None:
        raise RuntimeError("random_search_cnn did not find any successful configuration.")

    print(f"\n[CNN] ✅ Best val F1 = {best_f1:.4f} with params: {best_params}")

    return best_model, best_vocab, best_params, best_metrics, best_history