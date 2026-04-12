# dl_gru_extra.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import random
from collections import Counter
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

ExtraArray = Optional[np.ndarray]
ExtraDict  = Optional[Dict[str, np.ndarray]]
ExtraFeatures = Union[ExtraArray, ExtraDict, None]


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_extra_to_device(extra: Any, device: torch.device):
    if extra is None:
        return None
    if isinstance(extra, dict):
        return {k: v.to(device) for k, v in extra.items()}
    return extra.to(device)


def _infer_num_cats_from_extra(train_extra: Dict[str, np.ndarray]) -> Tuple[int, int]:
    num_drugs = int(np.nanmax(train_extra["drug_id"])) + 1
    num_conds = int(np.nanmax(train_extra["cond_id"])) + 1
    return num_drugs, num_conds


# -------------------------
# 1) vocab + tokenize
# -------------------------

def build_vocab_gru(texts: Sequence[str], vocab_size: int) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        if isinstance(t, str):
            counter.update(t.split())
    most_common = [w for w, _ in counter.most_common(max(vocab_size - 2, 0))]
    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    idx = 2
    for w in most_common:
        if w not in word_to_idx:
            word_to_idx[w] = idx
            idx += 1
    return word_to_idx


def texts_to_padded_indices(
    texts: Sequence[str], word_to_idx: Dict[str, int], max_len: int
) -> np.ndarray:
    n = len(texts)
    arr = np.zeros((n, max_len), dtype=np.int64)
    unk = word_to_idx.get(UNK_TOKEN, 1)
    for i, t in enumerate(texts):
        toks = t.split() if isinstance(t, str) else []
        ids = [word_to_idx.get(tok, unk) for tok in toks[:max_len]]
        arr[i, : len(ids)] = ids
    return arr


# -------------------------
# 2) dataset
# -------------------------

class GRUDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, labels: np.ndarray, extra_features: ExtraFeatures = None):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels    = torch.tensor(labels, dtype=torch.long)

        if extra_features is None:
            self.extra = None
        elif isinstance(extra_features, dict):
            needed = {"useful_z", "drug_id", "cond_id"}
            missing = needed - set(extra_features.keys())
            if missing:
                raise KeyError(f"extra_features dict missing keys: {missing}")

            # store as tensors; __getitem__ will pick idx and DataLoader will collate into batch tensors
            self.extra = {
                "useful_z": torch.tensor(np.asarray(extra_features["useful_z"]), dtype=torch.float32),
                "drug_id":  torch.tensor(np.asarray(extra_features["drug_id"]),  dtype=torch.long),
                "cond_id":  torch.tensor(np.asarray(extra_features["cond_id"]),  dtype=torch.long),
            }
        else:
            self.extra = torch.tensor(np.asarray(extra_features), dtype=torch.float32)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        item = {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}
        if self.extra is not None:
            if isinstance(self.extra, dict):
                item["extra"] = {k: v[idx] for k, v in self.extra.items()}
            else:
                item["extra"] = self.extra[idx]
        return item


# -------------------------
# 3) model
# -------------------------

class GRUSentimentModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.5,
        bidirectional: bool = True,
        # array-mode extras
        extra_feature_dim: int = 0,
        # dict-mode extras
        use_id_embeddings: bool = False,
        num_drugs: Optional[int] = None,
        num_conds: Optional[int] = None,
        cat_emb_dim: int = 16,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        self.use_id_embeddings = use_id_embeddings
        self.cat_emb_dim = cat_emb_dim

        if self.use_id_embeddings:
            if num_drugs is None or num_conds is None:
                raise ValueError("num_drugs/num_conds must be provided when use_id_embeddings=True")
            self.drug_emb = nn.Embedding(num_drugs, cat_emb_dim)
            self.cond_emb = nn.Embedding(num_conds, cat_emb_dim)
            self.extra_dim = 1 + 2 * cat_emb_dim
        else:
            self.extra_dim = int(extra_feature_dim)

        gru_out = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_out + self.extra_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, extra_features: Any = None) -> torch.Tensor:
        emb = self.embedding(input_ids)          # [B,T,E]
        _, h_n = self.gru(emb)                   # [L*D,B,H]

        if self.gru.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B,2H]
        else:
            h = h_n[-1]                                # [B,H]

        h = self.dropout(h)

        if self.extra_dim > 0 and extra_features is not None:
            if self.use_id_embeddings:
                if not isinstance(extra_features, dict):
                    raise TypeError("Expected dict extras, got non-dict.")
                useful = extra_features["useful_z"].float()
                # robustly force to [B,1]
                useful = useful.reshape(useful.shape[0], -1)
                useful = useful[:, :1]

                drug_e = self.drug_emb(extra_features["drug_id"].long())  # [B,E]
                cond_e = self.cond_emb(extra_features["cond_id"].long())  # [B,E]
                extra_vec = torch.cat([useful, drug_e, cond_e], dim=1)    # [B, 1+2E]
            else:
                extra_vec = extra_features.float()
                extra_vec = extra_vec.reshape(extra_vec.shape[0], -1)

            h = torch.cat([h, extra_vec], dim=1)

        return self.fc(h)


# -------------------------
# 4) training
# -------------------------

def train_gru_model(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    vocab_size: int,
    max_len: int,
    num_classes: int,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.5,
    batch_size: int = 32,
    num_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    train_extra: ExtraFeatures = None,
    val_extra: ExtraFeatures = None,
    device: Optional[torch.device] = None,
    random_state: int = 42,
    patience: Optional[int] = None,
    cat_emb_dim: int = 16,
) -> Tuple[GRUSentimentModel, Dict[str, float], Dict[str, int], Dict[str, List[float]]]:

    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)

    device = _get_device(device)

    word_to_idx = build_vocab_gru(train_texts, vocab_size=vocab_size)
    X_tr = texts_to_padded_indices(train_texts, word_to_idx, max_len=max_len)
    X_va = texts_to_padded_indices(val_texts,   word_to_idx, max_len=max_len)

    y_tr = np.asarray(train_labels, dtype=np.int64)
    y_va = np.asarray(val_labels,   dtype=np.int64)

    train_ds = GRUDataset(X_tr, y_tr, extra_features=train_extra)
    val_ds   = GRUDataset(X_va, y_va, extra_features=val_extra)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    use_id_embeddings = isinstance(train_extra, dict)

    if use_id_embeddings:
        num_drugs, num_conds = _infer_num_cats_from_extra(train_extra)  # type: ignore[arg-type]
        extra_dim = 0
    else:
        extra_dim = 0 if train_extra is None else int(train_extra.shape[1])  # type: ignore[union-attr]
        num_drugs, num_conds = None, None

    model = GRUSentimentModel(
        vocab_size=len(word_to_idx),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=True,
        extra_feature_dim=extra_dim,
        use_id_embeddings=use_id_embeddings,
        num_drugs=num_drugs,
        num_conds=num_conds,
        cat_emb_dim=cat_emb_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_f1, best_epoch, best_state = -1.0, -1, None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        tr_losses, tr_preds, tr_true = [], [], []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            extra     = move_extra_to_device(batch.get("extra"), device)

            optimizer.zero_grad()
            logits = model(input_ids, extra_features=extra)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            tr_losses.append(loss.item())
            tr_preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
            tr_true.append(labels.detach().cpu().numpy())

        tr_loss = float(np.mean(tr_losses))
        tr_preds = np.concatenate(tr_preds)
        tr_true  = np.concatenate(tr_true)
        tr_f1 = f1_score(tr_true, tr_preds, average="weighted")

        model.eval()
        va_losses, va_preds, va_true = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels    = batch["labels"].to(device)
                extra     = move_extra_to_device(batch.get("extra"), device)

                logits = model(input_ids, extra_features=extra)
                loss = criterion(logits, labels)

                va_losses.append(loss.item())
                va_preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
                va_true.append(labels.detach().cpu().numpy())

        va_loss = float(np.mean(va_losses))
        va_preds = np.concatenate(va_preds)
        va_true  = np.concatenate(va_true)
        va_f1 = f1_score(va_true, va_preds, average="weighted")

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        print(f"Epoch {epoch}/{num_epochs} | Train loss {tr_loss:.4f} F1 {tr_f1:.4f} | Val loss {va_loss:.4f} F1 {va_f1:.4f}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if patience is not None and no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {"best_val_f1": float(best_val_f1), "best_epoch": int(best_epoch)}
    return model, metrics, word_to_idx, history


def predict_gru(
    model: GRUSentimentModel,
    texts: Sequence[str],
    word_to_idx: Dict[str, int],
    max_len: int,
    batch_size: int = 32,
    extra_features: ExtraFeatures = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    device = _get_device(device)
    model = model.to(device).eval()

    X = texts_to_padded_indices(texts, word_to_idx, max_len=max_len)
    dummy = np.zeros(len(texts), dtype=np.int64)
    ds = GRUDataset(X, dummy, extra_features=extra_features)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_logits, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            extra = move_extra_to_device(batch.get("extra"), device)
            logits = model(input_ids, extra_features=extra)
            preds = torch.argmax(logits, dim=1)
            all_logits.append(logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_logits, 0), np.concatenate(all_preds, 0)


def random_search_gru(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    num_classes: int,
    vocab_size: int,
    max_len: int,
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    n_trials: int = 5,
    train_extra: ExtraFeatures = None,
    val_extra: ExtraFeatures = None,
    device: Optional[torch.device] = None,
    random_state: int = 42,
) -> Tuple[GRUSentimentModel, Dict[str, int], Dict[str, Any], Dict[str, float], Dict[str, List[float]]]:

    if param_grid is None:
        param_grid = {
            "embedding_dim": [150, 300],
            "hidden_dim":    [128, 256],
            "num_layers":    [1],
            "dropout":       [0.3, 0.5],
            "weight_decay":  [0.0, 1e-4],
            "batch_size":    [32],
            "num_epochs":    [12, 20],
            "lr":            [3e-4, 1e-3],
            "patience":      [3, 5],
            "cat_emb_dim":   [8, 16, 32],
        }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    rng = random.Random(random_state)
    combos = rng.sample(combos, n_trials) if (n_trials is not None and n_trials < len(combos)) else combos

    best_f1 = -1.0
    best = None

    print(f"[GRU] Random search over {len(combos)} combinations")
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"\n[GRU] Trial {i}/{len(combos)} params={params}")

        model, metrics, vocab, history = train_gru_model(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            vocab_size=vocab_size,
            max_len=max_len,
            num_classes=num_classes,
            embedding_dim=params["embedding_dim"],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            batch_size=params["batch_size"],
            num_epochs=params["num_epochs"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            patience=params["patience"],
            cat_emb_dim=params["cat_emb_dim"],
            train_extra=train_extra,
            val_extra=val_extra,
            device=device,
            random_state=random_state,
        )

        val_f1 = metrics["best_val_f1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best = (model, vocab, params, metrics, history)
            print(f"[GRU] 🔹 New best val F1 = {best_f1:.4f}")

    if best is None:
        raise RuntimeError("random_search_gru failed.")
    print(f"\n[GRU] ✅ Best val F1 = {best_f1:.4f}")
    return best