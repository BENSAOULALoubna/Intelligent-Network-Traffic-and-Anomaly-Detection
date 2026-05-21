"""
Train XGBoost congestion classifier and export as .pt file.
"""

import os

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "congestion_training.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "congestion_xgb_model.pt")

LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}
LABEL_REMAP_INV = {v: k for k, v in LABEL_REMAP.items()}

df = pd.read_csv("https://drive.google.com/uc?id=1uNd8iiTHT9Xp_aeMeKJ4ie-aQZIS64VD")
print(f"Loaded: {df.shape}")

df.drop(columns=["tcp_synack"], inplace=True)
df["is_portscan"] = (df["scenario"] == "training_portscan").astype(int)
df.drop(columns=["scenario"], inplace=True)

le_node = LabelEncoder()
df["node_id_enc"] = le_node.fit_transform(df["node_id"])
df.sort_values(["node_id", "t"], inplace=True)
df.reset_index(drop=True, inplace=True)

CORE_SIGNALS = [
    "bw_avg_mbps",
    "bw_in_mbps",
    "bw_out_mbps",
    "util_in_pct",
    "util_out_pct",
    "pkt_loss_pct",
    "pkt_dropped",
    "jitter_ms",
    "rtt_avg_ms",
    "rtt_std_ms",
    "tcp_retransmit_pct",
    "syn_rate_pps",
    "rst_rate_pps",
    "flow_churn_per_sec",
    "new_flows",
    "peak_kbps_1s_window",
]
WINDOWS = [5, 10, 30]
LAGS = [1, 3, 5, 10]

fe = df.copy()

for sig in [
    "bw_avg_mbps",
    "pkt_loss_pct",
    "jitter_ms",
    "tcp_retransmit_pct",
    "syn_rate_pps",
]:
    for lag in LAGS:
        fe[f"{sig}_lag{lag}"] = fe.groupby("node_id")[sig].shift(lag)

for sig in [
    "bw_avg_mbps",
    "pkt_loss_pct",
    "jitter_ms",
    "tcp_retransmit_pct",
    "flow_churn_per_sec",
    "peak_kbps_1s_window",
]:
    grp = fe.groupby("node_id")[sig]
    for w in WINDOWS:
        fe[f"{sig}_roll_mean_{w}"] = grp.transform(
            lambda x: x.rolling(w, min_periods=2).mean()
        )
        fe[f"{sig}_roll_std_{w}"] = grp.transform(
            lambda x: x.rolling(w, min_periods=2).std()
        )

for sig in ["bw_avg_mbps", "syn_rate_pps", "pkt_dropped"]:
    grp = fe.groupby("node_id")[sig]
    for w in [5, 10]:
        fe[f"{sig}_roll_max_{w}"] = grp.transform(
            lambda x: x.rolling(w, min_periods=2).max()
        )

for sig in ["bw_avg_mbps", "pkt_loss_pct", "jitter_ms", "util_in_pct", "util_out_pct"]:
    grp = fe.groupby("node_id")[sig]
    fe[f"{sig}_delta1"] = grp.diff(1)
    fe[f"{sig}_delta5"] = grp.diff(5)
    fe[f"{sig}_pct_chg"] = grp.pct_change(fill_method=None)

for sig in ["bw_avg_mbps", "pkt_loss_pct"]:
    for w in [10, 30]:
        mu = fe.groupby("node_id")[sig].transform(
            lambda x: x.rolling(w, min_periods=3).mean()
        )
        std = fe.groupby("node_id")[sig].transform(
            lambda x: x.rolling(w, min_periods=3).std()
        )
        fe[f"{sig}_cv_{w}"] = (std / (mu.abs() + 1e-9)).clip(upper=10)

for sig in ["bw_avg_mbps", "syn_rate_pps", "flow_churn_per_sec"]:
    roll_mean = fe.groupby("node_id")[sig].transform(
        lambda x: x.rolling(10, min_periods=3).mean()
    )
    fe[f"{sig}_burst"] = ((fe[sig] - roll_mean) / (roll_mean.abs() + 1e-9)).clip(
        lower=0
    )

fe["syn_flood_indicator"] = (
    fe["syn_rate_pps"] > fe["syn_rate_pps"].quantile(0.90)
).astype(int)
fe["util_imbalance"] = (fe["util_in_pct"] - fe["util_out_pct"]).abs()
fe["util_max"] = fe[["util_in_pct", "util_out_pct"]].max(axis=1)
fe["bw_asym_abs"] = fe["bw_asymmetry"].abs()
fe["loss_per_mbps"] = fe["pkt_loss_pct"] / (fe["bw_avg_mbps"] + 1e-9)
fe["retransmit_load"] = fe["tcp_retransmit_pct"] * fe["bw_avg_mbps"]
fe["jitter_per_mbps"] = fe["jitter_ms"] / (fe["bw_avg_mbps"] + 1e-9)
fe["t_log"] = np.log1p(fe["t"])
node_t_max = fe.groupby("node_id")["t"].transform("max")
node_t_min = fe.groupby("node_id")["t"].transform("min")
fe["t_relative"] = (fe["t"] - node_t_min) / (node_t_max - node_t_min + 1e-9)

FORECAST_HORIZON = 5
fe["future_congestion_label"] = fe.groupby("node_id")["congestion_label"].shift(
    -FORECAST_HORIZON
)

node_id_saved = fe["node_id"].copy()
fe = fe.groupby("node_id", group_keys=False).apply(lambda g: g.ffill())
fe["node_id"] = node_id_saved
fe.fillna(0, inplace=True)

TARGET = "future_congestion_label"
COMPOSITE_LEAKAGE = [
    "traffic_intensity",
    "iat_pressure",
    "queue_stress",
    "bw_util_ratio",
    "drop_rate_trend",
    "ctrl_pkt_ratio",
]
EXCLUDE = (
    [TARGET, "congestion_label", "node_id", "t", "t_diff", "split"]
    + COMPOSITE_LEAKAGE
    + ["label_lag1", "label_roll_mean_5", "tcp_health_score"]
)
FEATURE_COLS = [
    c for c in fe.select_dtypes(include="number").columns if c not in EXCLUDE
]

SPLIT_PCT = 0.80


def temporal_split(group, pct=SPLIT_PCT):
    cutoff = group["t"].quantile(pct)
    group["split"] = np.where(group["t"] <= cutoff, "train", "val")
    return group


fe_filtered = fe.dropna(subset=[TARGET]).copy()
node_id_saved = fe_filtered["node_id"].copy()
fe_filtered = fe_filtered.groupby("node_id", group_keys=False).apply(temporal_split)
fe_filtered["node_id"] = node_id_saved

train = fe_filtered[fe_filtered["split"] == "train"]
val = fe_filtered[fe_filtered["split"] == "val"]

X_train = train[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = train[TARGET]
X_val = val[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
y_val = val[TARGET]

y_train_xgb = y_train.map(LABEL_REMAP).astype(int)
y_val_xgb = y_val.map(LABEL_REMAP).astype(int)

from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight("balanced", y_train_xgb)

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=30,
)
xgb_model.fit(
    X_train.values.astype(np.float32),
    y_train_xgb,
    sample_weight=sample_weights,
    eval_set=[(X_val.values.astype(np.float32), y_val_xgb)],
    verbose=50,
)

y_pred_mapped = xgb_model.predict(X_val.values.astype(np.float32))
y_pred = np.array([LABEL_REMAP_INV[p] for p in y_pred_mapped])

from sklearn.metrics import classification_report

all_labels = np.unique(np.concatenate((y_train.unique(), y_val.unique())))
all_labels.sort()
target_names = [f"label_{int(l)}" for l in all_labels]
print("\n=== XGBoost Classification Report (Validation) ===")
print(
    classification_report(y_val, y_pred, labels=all_labels, target_names=target_names)
)

model_data = {
    "model": xgb_model,
    "feature_columns": FEATURE_COLS,
    "node_encoder": le_node,
    "forecast_horizon": FORECAST_HORIZON,
    "label_remap": LABEL_REMAP,
    "label_remap_inv": LABEL_REMAP_INV,
    "metrics": {"train_rows": len(X_train), "val_rows": len(X_val)},
}
torch.save(model_data, MODEL_PATH)
print(f"Model exported to {MODEL_PATH}")
