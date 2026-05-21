"""
ablation.py — What happens when we remove the top-2 features?

Drops `syn_burst_ratio` and `lookback_acceleration` from the feature set,
then trains both XGBoost and LSTM to see if temporal modeling adds value
when instantaneous burst indicators are unavailable.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("processed_data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = {0: "Normal", 1: "Horizontal", 2: "Vertical", 3: "Distributed/Spillover"}

# drop these two features (they trivially solve per-timestep classification)
DROP_FEATS = ["syn_burst_ratio", "lookback_acceleration"]

# LSTM hyperparams
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.4
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS = 80
PATIENCE = 10
FOCAL_GAMMA = 2.0
XGB_AVAILABLE = True


class LSTMClassifier(nn.Module):
    def __init__(self, n_feats, n_hidden, n_layers, n_classes, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, n_hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0, bidirectional=True)
        lstm_out = n_hidden * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, lstm_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out // 2, n_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    weights = counts.sum() / (len(classes) * counts.astype(float))
    w = torch.ones(int(classes.max()) + 1, dtype=torch.float32)
    for c, v in zip(classes, weights):
        w[int(c)] = v
    return w.to(DEVICE)


def oversample(X, y):
    B, T, F = X.shape
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X.reshape(B, -1), y)
    return X_res.reshape(-1, T, F), y_res


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for Xb, yb in loader:
        logits = model(Xb.to(DEVICE))
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def remove_features(X, cols, drop_names):
    """Drop named features from the last axis of X."""
    drop_idx = [i for i, c in enumerate(cols) if c in drop_names]
    keep_idx = [i for i in range(len(cols)) if i not in drop_idx]
    return X[:, :, keep_idx], [c for i, c in enumerate(cols) if i not in drop_idx]


def main():
    print("=" * 60)
    print("  ABLATION: Remove syn_burst_ratio & lookback_acceleration")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    y_val   = np.load(DATA_DIR / "y_val.npy")
    X_test  = np.load(DATA_DIR / "X_test.npy")
    y_test  = np.load(DATA_DIR / "y_test.npy")

    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)

    cols = meta["feature_cols"]
    print(f"\nOriginal features: {len(cols)}  →  dropping {DROP_FEATS}")

    X_train, cols_keep = remove_features(X_train, cols, DROP_FEATS)
    X_val,   _         = remove_features(X_val,   cols, DROP_FEATS)
    X_test,  _         = remove_features(X_test,  cols, DROP_FEATS)

    n_feats = X_train.shape[2]
    n_classes = len(meta["label_map"])
    print(f"Reduced features: {n_feats}  (dropped {len(DROP_FEATS)})")
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── 1. XGBoost (flattened) ─────────────────────────────────
    print("\n─── XGBoost on flattened windows ───")
    try:
        import xgboost as xgb
        B, T, F = X_train.shape
        X_train_flat = X_train.reshape(B, -1)
        X_val_flat   = X_val.reshape(len(X_val), -1)
        X_test_flat  = X_test.reshape(len(X_test), -1)

        model_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=42, n_jobs=-1,
        )
        model_xgb.fit(X_train_flat, y_train,
                      eval_set=[(X_train_flat, y_train), (X_val_flat, y_val)],
                      verbose=False)

        y_pred_xgb = model_xgb.predict(X_test_flat)
        xgb_acc = accuracy_score(y_test, y_pred_xgb)
        xgb_f1 = f1_score(y_test, y_pred_xgb, average="macro")
        print(f"  XGBoost Test:  Accuracy={xgb_acc:.4f}  Macro-F1={xgb_f1:.4f}")
        print(classification_report(y_test, y_pred_xgb, target_names=[
            f"{k} ({LABEL_NAMES[k]})" for k in sorted(LABEL_NAMES)], digits=4))
    except ImportError:
        print("  XGBoost not available")
        xgb_acc, xgb_f1 = None, None

    # ── 2. LSTM ────────────────────────────────────────────────
    print("\n─── LSTM (temporal) ───")
    X_train_os, y_train_os = oversample(X_train, y_train)

    train_ds = TensorDataset(torch.from_numpy(X_train_os), torch.from_numpy(y_train_os).long())
    val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long())
    test_ds  = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = LSTMClassifier(n_feats, HIDDEN_DIM, NUM_LAYERS, n_classes, DROPOUT).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=compute_class_weights(y_train_os))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_f1 = 0.0
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)

        y_val_pred, y_val_true = evaluate(model, val_loader)
        val_f1 = f1_score(y_val_true, y_val_pred, average="macro")
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), DATA_DIR / "best_lstm_ablation.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}  (best val macro-F1: {best_val_f1:.4f})")
                break
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss: {total_loss/len(train_loader.dataset):.4f} | val macro-F1: {val_f1:.4f}")

    model.load_state_dict(torch.load(DATA_DIR / "best_lstm_ablation.pt", weights_only=True))
    y_pred_lstm, y_test_true = evaluate(model, test_loader)
    lstm_acc = accuracy_score(y_test_true, y_pred_lstm)
    lstm_f1 = f1_score(y_test_true, y_pred_lstm, average="macro")

    print(f"\n  LSTM Test:  Accuracy={lstm_acc:.4f}  Macro-F1={lstm_f1:.4f}")
    print(classification_report(y_test_true, y_pred_lstm, target_names=[
        f"{k} ({LABEL_NAMES[k]})" for k in sorted(LABEL_NAMES)], digits=4))

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ABLATION SUMMARY")
    print("=" * 60)
    print(f"""
  Without syn_burst_ratio and lookback_acceleration:
  ┌──────────────────────┬──────────┬──────────┐
  │ Model                │ Accuracy │ Macro-F1 │
  ├──────────────────────┼──────────┼──────────┤
  │ XGBoost (flattened)  │ {xgb_acc or 0:.4f}  │ {xgb_f1 or 0:.4f}  │
  │ LSTM (temporal)      │ {lstm_acc:.4f}  │ {lstm_f1:.4f}  │
  └──────────────────────┴──────────┴──────────┘

  Interpretation:
  - If LSTM significantly outperforms XGBoost here, it means temporal
    context genuinely helps when instantaneous burst indicators are removed.
  - If both drop together, the scan signal is spread across other features
    and temporal modeling adds limited value.
""")

    cm_lstm = confusion_matrix(y_test_true, y_pred_lstm)
    print(" LSTM Confusion Matrix:")
    print(pd.DataFrame(cm_lstm,
        index=[f"True {k}" for k in sorted(LABEL_NAMES)],
        columns=[f"Pred {k}" for k in sorted(LABEL_NAMES)]))
    if XGB_AVAILABLE:
        cm_xgb = confusion_matrix(y_test, y_pred_xgb)
        print("\n XGBoost Confusion Matrix:")
        print(pd.DataFrame(cm_xgb,
            index=[f"True {k}" for k in sorted(LABEL_NAMES)],
            columns=[f"Pred {k}" for k in sorted(LABEL_NAMES)]))


if __name__ == "__main__":
    main()
