# dl_albert_extra.py
# Drop-in full file; supports dict extras: useful_z + drug_id + cond_id
# Adds GRU/CNN-style grid control for cat_emb_dim (and weight_decay, device).

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import itertools
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, AlbertModel


# ----------------------------
# Helpers
# ----------------------------

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


ExtraArray = Optional[np.ndarray]
ExtraDict = Optional[Dict[str, np.ndarray]]
ExtraFeatures = Union[ExtraArray, ExtraDict, None]


def _infer_num_cats_from_extra(train_extra: Dict[str, np.ndarray]) -> Tuple[int, int]:
    num_drugs = int(np.nanmax(train_extra["drug_id"])) + 1
    num_conds = int(np.nanmax(train_extra["cond_id"])) + 1
    return num_drugs, num_conds


def _ensure_1d_long(x: torch.Tensor) -> torch.Tensor:
    # Accept shapes: (B,), (B,1), scalar; return (B,)
    if x.dim() == 0:
        x = x.view(1)
    if x.dim() == 2 and x.size(1) == 1:
        x = x.squeeze(1)
    return x.long()


def _ensure_useful_colvec(x: torch.Tensor) -> torch.Tensor:
    # Accept shapes: (B,), (B,1), scalar; return (B,1)
    x = x.float()
    if x.dim() == 0:
        x = x.view(1)
    if x.dim() == 2 and x.size(1) == 1:
        x = x.squeeze(1)
    return x.unsqueeze(1)  # (B,1)


# --------------------------------------------------------------------
# 1) Dataset
# --------------------------------------------------------------------

class AlbertTabulateDataset(Dataset):
    """
    Supports:
      - tabular = None
      - tabular = np.ndarray  [N, F]
      - tabular = dict with keys {"useful_z","drug_id","cond_id"}
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]],
        tokenizer: AutoTokenizer,
        max_len: int,
        tabular: ExtraFeatures = None,
    ):
        self.texts = list(texts)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len

        if tabular is None:
            self.tabular = None

        elif isinstance(tabular, dict):
            needed = {"useful_z", "drug_id", "cond_id"}
            missing = needed - set(tabular.keys())
            if missing:
                raise KeyError(f"tabular dict missing keys: {missing}")

            # store as 1D tensors so batching -> (B,), not (B,1)
            useful = np.asarray(tabular["useful_z"], dtype=np.float32).reshape(-1)
            drug   = np.asarray(tabular["drug_id"],  dtype=np.int64).reshape(-1)
            cond   = np.asarray(tabular["cond_id"],  dtype=np.int64).reshape(-1)

            self.tabular = {
                "useful_z": torch.tensor(useful, dtype=torch.float32),  # (N,)
                "drug_id":  torch.tensor(drug,   dtype=torch.long),     # (N,)
                "cond_id":  torch.tensor(cond,   dtype=torch.long),     # (N,)
            }

        else:
            self.tabular = np.asarray(tabular, dtype=np.float32)  # [N, F]

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
            if isinstance(self.tabular, dict):
                item["tabular"] = {k: v[idx] for k, v in self.tabular.items()}
            else:
                item["tabular"] = torch.tensor(self.tabular[idx], dtype=torch.float32)  # (F,)

        return item


# --------------------------------------------------------------------
# 2) Model
# --------------------------------------------------------------------

class AlbertTabulate(nn.Module):
    """
    Two modes:
      (A) array mode: tabular_dim > 0, pass tabular as float tensor [B,F]
      (B) dict mode: initialize with num_drugs & num_conds; pass tabular dict with:
            useful_z: (B,) or (B,1)
            drug_id : (B,) or (B,1)
            cond_id : (B,) or (B,1)
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
        tabular_dim: int = 0,                 # array-mode dim
        freeze_base: bool = False,
        # dict-mode config
        num_drugs: Optional[int] = None,
        num_conds: Optional[int] = None,
        cat_emb_dim: int = 16,
    ):
        super().__init__()

        self.base_model = AlbertModel.from_pretrained(model_name_or_path)
        hidden_size = self.base_model.config.hidden_size

        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        self.use_id_embeddings = (num_drugs is not None) and (num_conds is not None)
        self.cat_emb_dim = cat_emb_dim

        if self.use_id_embeddings:
            self.drug_emb = nn.Embedding(num_drugs, cat_emb_dim)
            self.cond_emb = nn.Embedding(num_conds, cat_emb_dim)
            self.extra_dim = 1 + 2 * cat_emb_dim  # useful_z + drug_emb + cond_emb
        else:
            self.extra_dim = int(tabular_dim)

        in_dim = hidden_size + self.extra_dim

        if hidden_dim is None:
            self.fc = nn.Linear(in_dim, num_classes)
            self.use_hidden = False
        else:
            self.fc_hidden = nn.Linear(in_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, num_classes)
            self.activation = nn.ReLU()
            self.use_hidden = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular: Any = None,
    ) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # [B,H]
        h = self.dropout(cls)

        extra_vec = None
        if self.extra_dim > 0 and tabular is not None:
            if self.use_id_embeddings:
                if not isinstance(tabular, dict):
                    raise TypeError("Model initialized for dict extras, but got non-dict tabular.")

                useful = _ensure_useful_colvec(tabular["useful_z"])  # (B,1)
                drug_id = _ensure_1d_long(tabular["drug_id"])        # (B,)
                cond_id = _ensure_1d_long(tabular["cond_id"])        # (B,)

                drug_e = self.drug_emb(drug_id)  # (B,E)
                cond_e = self.cond_emb(cond_id)  # (B,E)

                extra_vec = torch.cat([useful, drug_e, cond_e], dim=1)  # (B, 1+2E)
            else:
                extra_vec = tabular.float()  # expected (B,F)

        if extra_vec is not None:
            h = torch.cat([h, extra_vec], dim=1)

        if self.use_hidden:
            h = self.fc_hidden(h)
            h = self.activation(h)
            h = self.dropout(h)
            logits = self.fc_out(h)
        else:
            logits = self.fc(h)

        return logits


# --------------------------------------------------------------------
# 3) Training
# --------------------------------------------------------------------

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
    weight_decay: float = 0.0,
    dropout: float = 0.1,
    hidden_dim: Optional[int] = None,
    train_tabular: ExtraFeatures = None,
    val_tabular: ExtraFeatures = None,
    # dict-mode config
    num_drugs: Optional[int] = None,
    num_conds: Optional[int] = None,
    cat_emb_dim: int = 16,
    random_state: int = 42,
    patience: Optional[int] = None,
    freeze_base: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[AlbertTabulate, Dict[str, float], AutoTokenizer, Dict[str, List[float]]]:
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)

    device = _get_device(device)
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

    use_id_embeddings = isinstance(train_tabular, dict)

    if use_id_embeddings:
        if (num_drugs is None) or (num_conds is None):
            num_drugs, num_conds = _infer_num_cats_from_extra(train_tabular)  # type: ignore[arg-type]
        tab_dim = 1 + 2 * cat_emb_dim
    else:
        tab_dim = 0 if train_tabular is None else int(train_tabular.shape[1])  # type: ignore[union-attr]

    model = AlbertTabulate(
        model_name_or_path=model_name_or_path,
        num_classes=num_classes,
        dropout=dropout,
        hidden_dim=hidden_dim,
        tabular_dim=(0 if use_id_embeddings else tab_dim),
        freeze_base=freeze_base,
        num_drugs=(num_drugs if use_id_embeddings else None),
        num_conds=(num_conds if use_id_embeddings else None),
        cat_emb_dim=cat_emb_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

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

            tab = move_extra_to_device(batch.get("tabular"), device)

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

                tab = move_extra_to_device(batch.get("tabular"), device)

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

    metrics = {"best_val_f1": float(best_val_f1), "best_epoch": int(best_epoch)}
    return model, metrics, tokenizer, history


# --------------------------------------------------------------------
# 4) Prediction
# --------------------------------------------------------------------

def predict_albert_tabulate(
    model: AlbertTabulate,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    max_len: int,
    batch_size: int = 32,
    tabular: ExtraFeatures = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    device = _get_device(device)
    model = model.to(device).eval()

    dataset = AlbertTabulateDataset(
        texts=texts,
        labels=None,
        tokenizer=tokenizer,
        max_len=max_len,
        tabular=tabular,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tab = move_extra_to_device(batch.get("tabular"), device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, tabular=tab)
            preds = torch.argmax(logits, dim=1)

            all_logits.append(logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_preds, axis=0)


def predict_albert(
    model: AlbertTabulate,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    max_len: int,
    batch_size: int = 32,
    tabular: ExtraFeatures = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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
# 5) Random search (GRU/CNN style; cat_emb_dim inside param_grid)
# --------------------------------------------------------------------

def random_search_albert(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    num_classes: int,
    max_len: int,
    # canonical args
    train_tabular: ExtraFeatures = None,
    val_tabular: ExtraFeatures = None,
    model_name_or_path: str = "albert-base-v2",
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    n_trials: int = 5,
    random_state: int = 42,
    # aliases (notebook consistency)
    train_extra: ExtraFeatures = None,
    val_extra: ExtraFeatures = None,
    # dict-mode config (optional overrides)
    num_drugs: Optional[int] = None,
    num_conds: Optional[int] = None,
    # default if not in grid
    cat_emb_dim: int = 16,
    device: Optional[torch.device] = None,
) -> Tuple[AlbertTabulate, AutoTokenizer, Dict[str, Any], Dict[str, float], Dict[str, List[float]]]:
    # resolve aliases
    if train_tabular is None and train_extra is not None:
        train_tabular = train_extra
    if val_tabular is None and val_extra is not None:
        val_tabular = val_extra

    if param_grid is None:
        param_grid = {
            "batch_size":   [16, 32],
            "num_epochs":   [3, 4],
            "lr":           [2e-5, 3e-5],
            "weight_decay": [0.0, 0.01],
            "dropout":      [0.1, 0.2],
            "hidden_dim":   [None, 128],
            "freeze_base":  [False],
            "patience":     [1, 2],
            "cat_emb_dim":  [16],
        }

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    all_combos = list(itertools.product(*values))

    rng = random.Random(random_state)
    sampled = rng.sample(all_combos, n_trials) if (n_trials is not None and n_trials < len(all_combos)) else all_combos

    best_f1 = -1.0
    best_model = None
    best_tokenizer = None
    best_params = None
    best_metrics = None
    best_history = None

    print(f"[ALBERT] Random search over {len(sampled)} combinations.")

    for trial_idx, combo in enumerate(sampled, 1):
        params = dict(zip(keys, combo))
        print(f"\n[ALBERT] Trial {trial_idx}/{len(sampled)} params={params}")

        trial_cat_emb_dim = params.get("cat_emb_dim", cat_emb_dim)

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
            weight_decay=params.get("weight_decay", 0.0),
            dropout=params["dropout"],
            hidden_dim=params["hidden_dim"],
            train_tabular=train_tabular,
            val_tabular=val_tabular,
            num_drugs=num_drugs,
            num_conds=num_conds,
            cat_emb_dim=trial_cat_emb_dim,
            random_state=random_state,
            patience=params["patience"],
            freeze_base=params["freeze_base"],
            device=device,
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