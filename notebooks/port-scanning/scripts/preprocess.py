"""
preprocess.py — Proper preprocessing for port-scanning dataset.

Fixes:
  1. Drops 9 constant/near-constant features (zero discriminative value)
  2. Prunes 38 → ~20 features by keeping 1 per highly correlated group
  3. Chronological split per node (70/15/15) — eliminates temporal leakage
  4. Resets sliding windows at session gaps > 100s — no cross-session windows
  5. Drops `scenario` column (data-source label, not a predictive feature)
  6. One-hot encodes `observer_node`
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ── config ──────────────────────────────────────────────────────────
FILE_ID = "1lYT9J7ZAo4MdTqN-RHqBja8OVins-Nv8"
SEQ_LEN = 12           # 120-second windows (10 s per timestep)
STRIDE = 2             # 50 % overlap between consecutive windows
GAP_THRESHOLD = 100    # session-gap threshold in seconds
SPLIT = (0.70, 0.15, 0.15)   # train / val / test proportions

# label mapping: original → contiguous
LABEL_MAP = {0: 0, 1: 1, 2: 2, 5: 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ── constant / near-constant features (zero variance) ──────────────
CONSTANT_FEATURES = [
    "total_synack", "total_ack", "unique_src_ips", "unique_dst_ips",
    "unique_dst_ports", "payload_bytes", "scanning_src_count",
    "scan_breadth", "scan_entropy_proxy",
]

# ── features to keep (1 per high-correlation group + important ones) ─
# Based on EDA correlation analysis (|r| > 0.95 groups):
#   G1: total_syn (keep), conn_attempts, syn_rate_per_sec, syn_ack_ratio
#   G2: failed_conns (keep), total_rst, scan_intensity
#   G3: conn_failure_rate (keep), rst_per_syn, failed_per_syn, conn_success_pct
#   G4: lb_syn_60s (keep), lb_dst_ip_60s, lb_dst_port_60s
#   G5: syn_iat_mean_ms (keep), syn_iat_std_ms
#   G6: proto_overhead_pct (keep), ctrl_overhead_pct
FEATURES_TO_KEEP = [
    # TCP flags
    "total_syn",           # G1 representative
    "total_fin",
    "syn_rst_ratio",
    "syn_to_fin_ratio",
    # Connection
    "failed_conns",        # G2 representative
    "conn_failure_rate",   # G3 representative
    "conn_completion_pct",
    # Dispersion
    "fan_out_ratio",
    "port_scan_ratio",
    "temporal_scan_density",
    # Lookback
    "lb_syn_60s",          # G4 representative
    "lb_syn_300s",
    # Timing
    "syn_iat_mean_ms",     # G5 representative
    "syn_iat_cv",
    "syn_regularity",
    # Important burst indicators (LSTM should detect onset before these peak)
    "syn_burst_ratio",
    "lookback_acceleration",
    # Protocol
    "proto_overhead_pct",  # G6 representative
    "tcp_ctrl_bytes",
    "total_packets",
]

NODE_COL = "observer_node"
LABEL_COL = "scan_label"
TIME_COL = "t"
CAT_COLS = [NODE_COL, "scenario"]  # scenario is excluded from features


# ══════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """Load the dataset from Google Drive."""
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    df = pd.read_csv(url)
    df = df.sort_values([NODE_COL, TIME_COL]).reset_index(drop=True)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def drop_useless_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant features and the `scenario` column."""
    to_drop = [c for c in CONSTANT_FEATURES if c in df.columns] + ["scenario"]
    df = df.drop(columns=to_drop, errors="ignore")
    print(f"Dropped {len(to_drop)} useless columns (constants + scenario).  Shape: {df.shape}")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the chosen feature set plus metadata columns."""
    keep = list(set(FEATURES_TO_KEEP) & set(df.columns))
    meta = [TIME_COL, NODE_COL, LABEL_COL]
    missing = [c for c in FEATURES_TO_KEEP if c not in df.columns]
    if missing:
        print(f"  Warning: features not found in data: {missing}")
    print(f"Selected {len(keep)} features + meta columns.  Shape: {len(keep) + len(meta)} cols")
    return df[meta + keep].copy()


def one_hot_encode_node(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode observer_node and drop the original column."""
    node_dummies = pd.get_dummies(df[NODE_COL], prefix="node")
    df = pd.concat([df.drop(columns=[NODE_COL]), node_dummies], axis=1)
    print(f"One-hot encoded node → {list(node_dummies.columns)}")
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map original labels {0,1,2,5} → contiguous {0,1,2,3}."""
    df["label"] = df[LABEL_COL].map(LABEL_MAP)
    df = df.drop(columns=[LABEL_COL])
    return df


def create_sequences_per_episode(
    df_node: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = SEQ_LEN,
    stride: int = STRIDE,
):
    """
    Create sliding windows within each contiguous episode (gaps ≤ GAP_THRESHOLD).
    Returns arrays X (N, seq_len, n_feats) and y (N,) with per-timestep labels.
    """
    X_list, y_list = [], []

    # detect episode boundaries
    time_diffs = df_node[TIME_COL].diff()
    episode_starts = [0] + list(np.where(time_diffs > GAP_THRESHOLD)[0])
    episode_ends = episode_starts[1:] + [len(df_node)]

    total_windows = 0
    for start, end in zip(episode_starts, episode_ends):
        ep = df_node.iloc[start:end]
        if len(ep) < seq_len:
            continue
        # sliding windows within this episode
        for i in range(0, len(ep) - seq_len + 1, stride):
            window = ep.iloc[i : i + seq_len]
            X_list.append(window[feature_cols].values.astype(np.float32))
            # use the LAST timestep's label as the target
            y_list.append(window["label"].iloc[-1])
            total_windows += 1

    if not X_list:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty(0, dtype=np.int64)

    return np.stack(X_list), np.array(y_list, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════
def preprocess():
    """End-to-end preprocessing pipeline."""
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_data()
    df = drop_useless_features(df)
    df = select_features(df)
    df = one_hot_encode_node(df)
    df = map_labels(df)

    # identify feature columns (everything except time, label)
    feature_cols = [c for c in df.columns if c not in (TIME_COL, "label")]
    print(f"Feature dimension: {len(feature_cols)}  ({feature_cols})")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

    # ── per-node chronological split ──────────────────────────────
    node_prefixes = sorted(df.columns[df.columns.str.startswith("node_")])
    # Reconstruct node name from one-hot columns
    node_names = [p.replace("node_", "") for p in node_prefixes]

    train_X, train_y = [], []
    val_X, val_y = [], []
    test_X, test_y = [], []

    for node_prefix in node_prefixes:
        # filter rows belonging to this node
        mask = df[node_prefix] == 1
        df_node = df[mask].sort_values(TIME_COL).reset_index(drop=True)
        n = len(df_node)
        if n < SEQ_LEN:
            continue

        t1 = int(n * SPLIT[0])
        t2 = int(n * (SPLIT[0] + SPLIT[1]))

        df_train = df_node.iloc[:t1]
        df_val   = df_node.iloc[t1:t2]
        df_test  = df_node.iloc[t2:]

        for split_name, split_df, X_store, y_store in [
            ("train", df_train, train_X, train_y),
            ("val",   df_val,   val_X,   val_y),
            ("test",  df_test,  test_X,  test_y),
        ]:
            Xs, ys = create_sequences_per_episode(split_df, feature_cols)
            if len(Xs) > 0:
                X_store.append(Xs)
                y_store.append(ys)

    def _cat(arrs):
        return np.concatenate(arrs, axis=0) if arrs else np.empty((0,))

    X_train, y_train = _cat(train_X), _cat(train_y)
    X_val,   y_val   = _cat(val_X),   _cat(val_y)
    X_test,  y_test  = _cat(test_X),  _cat(test_y)

    print(f"\n── Sequence counts ──")
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")
    print(f"  Sequence shape: (seq_len={SEQ_LEN}, n_feats={X_train.shape[2]})")

    # ── Standardise features (fit on train only) ──────────────────
    scaler = StandardScaler()
    B, T, F = X_train.shape
    X_train_2d = X_train.reshape(-1, F)
    scaler.fit(X_train_2d)

    def _scale(X):
        b, t, f = X.shape
        return scaler.transform(X.reshape(-1, f)).reshape(b, t, f)

    X_train = _scale(X_train)
    X_val   = _scale(X_val)
    X_test  = _scale(X_test)

    print(f"\n  Feature means (train):  {scaler.mean_[:5]}...")
    print(f"  Feature  stds (train):  {np.sqrt(scaler.var_[:5])}...")

    # ── Save arrays ────────────────────────────────────────────────
    out_dir = Path("processed_data")
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy",   X_val)
    np.save(out_dir / "y_val.npy",   y_val)
    np.save(out_dir / "X_test.npy",  X_test)
    np.save(out_dir / "y_test.npy",  y_test)

    # save metadata
    meta = {
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "gap_threshold": GAP_THRESHOLD,
        "label_map": LABEL_MAP,
        "inv_label_map": INV_LABEL_MAP,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_feats": F,
    }
    import json
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n✓ Saved to '{out_dir}/'")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    preprocess()
