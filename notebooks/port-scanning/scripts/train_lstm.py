"""
train_lstm.py — Train a basic LSTM baseline for port-scan detection.

Key design decisions:
  • Predicts per-timestep label from a sequence of features.
  • Uses Focal Loss + class weights to handle severe class imbalance.
  • Per-node evaluation to check generalisation.
  • Monitors macro-F1 on val set for early stopping.

Usage:
  .venv/bin/python preprocess.py   # run once
  .venv/bin/python train_lstm.py
"""

import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
)
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings("ignore")

# ── config ──────────────────────────────────────────────────────────
DATA_DIR = Path("processed_data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# hyperparameters
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.4
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
FOCAL_GAMMA = 2.0

LABEL_NAMES = {0: "Normal", 1: "Horizontal", 2: "Vertical", 3: "Distributed/Spillover"}


# ══════════════════════════════════════════════════════════════════════
class LSTMClassifier(nn.Module):
    """Simple 2-layer LSTM → MLP classifier."""

    def __init__(self, n_feats: int, n_hidden: int, n_layers: int,
                 n_classes: int, dropout: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feats,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        lstm_out = n_hidden * 2  # bidirectional

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, lstm_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out // 2, n_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, (h_n, _) = self.lstm(x)
        # use last timestep output
        last_out = out[:, -1, :]  # (B, lstm_out)
        logits = self.classifier(last_out)
        return logits


# ══════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples, focuses on hard / rare classes."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight,
                                              reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# ══════════════════════════════════════════════════════════════════════
def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights."""
    classes, counts = np.unique(y, return_counts=True)
    weights = counts.sum() / (len(classes) * counts.astype(float))
    w = torch.ones(int(classes.max()) + 1, dtype=torch.float32)
    for c, v in zip(classes, weights):
        w[int(c)] = v
    print(f"  Class weights: {dict(zip(classes, weights.round(2)))}")
    return w.to(DEVICE)


# ══════════════════════════════════════════════════════════════════════
def oversample_minority(X: np.ndarray, y: np.ndarray):
    """Oversample minority classes at the sequence level (before batching)."""
    B, T, F = X.shape
    X_2d = X.reshape(B, -1)
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_2d, y)
    return X_res.reshape(-1, T, F), y_res


# ══════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for Xb, yb in loader:
        Xb = Xb.to(DEVICE)
        logits = model(Xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  LSTM TRAINING — Port-Scan Detection")
    print("=" * 60)

    # ── Load preprocessed data ───────────────────────────────────
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    y_val   = np.load(DATA_DIR / "y_val.npy")
    X_test  = np.load(DATA_DIR / "X_test.npy")
    y_test  = np.load(DATA_DIR / "y_test.npy")

    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)

    n_feats = meta["n_feats"]
    n_classes = len(meta["label_map"])
    seq_len = meta["seq_len"]

    print(f"\n  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  Features: {n_feats}  Classes: {n_classes}  Seq len: {seq_len}")

    # ── Oversample minority classes ────────────────────────────────
    print("\n── Class distribution before oversampling ──")
    for c in range(n_classes):
        print(f"  Class {c} ({LABEL_NAMES[c]}): {(y_train == c).sum():,}")
    X_train, y_train = oversample_minority(X_train, y_train)
    print(f"\n── After oversampling ──")
    for c in range(n_classes):
        print(f"  Class {c} ({LABEL_NAMES[c]}): {(y_train == c).sum():,}")

    # ── Dataloaders ───────────────────────────────────────────────
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val).long())
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=0)

    # ── Model setup ───────────────────────────────────────────────
    model = LSTMClassifier(
        n_feats=n_feats,
        n_hidden=HIDDEN_DIM,
        n_layers=NUM_LAYERS,
        n_classes=n_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    class_weights = compute_class_weights(y_train)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    print(f"\n── Model ──")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Training loop ─────────────────────────────────────────────
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n── Training ({EPOCHS} max epochs, patience={PATIENCE}) ──")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)

        y_val_pred, y_val_true = evaluate(model, val_loader)
        val_f1_macro = f1_score(y_val_true, y_val_pred, average="macro")
        val_f1_weighted = f1_score(y_val_true, y_val_pred, average="weighted")
        val_acc = accuracy_score(y_val_true, y_val_pred)

        scheduler.step(val_f1_macro)

        if (epoch - 1) % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | loss: {train_loss:.4f} | "
                f"val acc: {val_acc:.4f} | val macro-F1: {val_f1_macro:.4f}"
            )

        # early stopping on val macro-F1
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), DATA_DIR / "best_lstm.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}. "
                      f"Best val macro-F1: {best_val_f1:.4f} @ epoch {best_epoch}")
                break

    # ── Load best model & evaluate on test ────────────────────────
    print(f"\n── Final Evaluation (best epoch {best_epoch}) ──")
    model.load_state_dict(torch.load(DATA_DIR / "best_lstm.pt", weights_only=True))

    y_test_pred, y_test_true = evaluate(model, test_loader)

    # overall metrics
    test_f1_macro = f1_score(y_test_true, y_test_pred, average="macro")
    test_f1_weighted = f1_score(y_test_true, y_test_pred, average="weighted")
    test_acc = accuracy_score(y_test_true, y_test_pred)
    test_mcc = matthews_corrcoef(y_test_true, y_test_pred)

    print(f"\n── Test Set Performance ──")
    print(f"  Accuracy:           {test_acc:.4f}")
    print(f"  Macro F1:           {test_f1_macro:.4f}")
    print(f"  Weighted F1:        {test_f1_weighted:.4f}")
    print(f"  MCC:                {test_mcc:.4f}")

    # map labels back to original
    inv_map = meta["inv_label_map"]
    y_test_orig = np.array([inv_map[str(v)] for v in y_test_true])
    y_pred_orig = np.array([inv_map[str(v)] for v in y_test_pred])

    print(f"\n  Classification Report:")
    target_names = [f"{k} ({LABEL_NAMES[k]})" for k in sorted(LABEL_NAMES)]
    print(classification_report(
        y_test_true, y_test_pred,
        target_names=target_names,
        digits=4,
    ))

    cm = confusion_matrix(y_test_true, y_test_pred)
    print(f"\n  Confusion Matrix:")
    print(pd.DataFrame(cm,
        index=[f"True {k}" for k in sorted(LABEL_NAMES)],
        columns=[f"Pred {k}" for k in sorted(LABEL_NAMES)],
    ))


if __name__ == "__main__":
    import pandas as pd
    main()
