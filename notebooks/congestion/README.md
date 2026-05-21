# Network Congestion Prediction — Feature Engineering & Model Training

Multi-class congestion classification (labels 0–4) on simulated network traffic data, using temporal feature engineering and a Random Forest baseline. The model forecasts congestion 5 time-steps (~3 seconds) ahead per network node.

## Project Structure

```
├── congestion_vf_forecasting_notebook.ipynb   # Main notebook (full pipeline)
├── train_and_export.py                        # Standalone training + .pt export script
├── feature_list (1).csv                       # Engineered feature columns (140 features)
├── congestion_rf_model.pt                     # Trained RandomForest model (PyTorch-serialized)
├── requirements.txt                           # Python dependencies
└── README.md
```

## Dataset

Download `congestion_training.csv` from [Google Drive](https://drive.google.com/drive/folders/16Keq9_UbZu3-9YVyX3OrdOad1YO4CK9_?usp=drive_link) and place it in this directory.

| Property | Value |
|---|---|
| Rows | 105,982 |
| Columns | 57 |
| Nodes | 6 (`boundary`, `core0–2`, `edge0–1`) |
| Scenarios | `training_congestion` (77K), `training_portscan` (28K) |
| Target | `congestion_label` ∈ {0, 1, 2, 4} (imbalanced: label 2 dominates at 62%) |

## Setup

```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook congestion_vf_forecasting_notebook.ipynb
```

Or train from the command line:
```bash
python train_and_export.py
```

## Notebook Walkthrough

### 1. Imports & Setup
Standard data science stack: `pandas`, `numpy`, `matplotlib`, `sklearn` (RandomForest, GradientBoosting, preprocessing, metrics).

### 2. Load & Initial Inspection
- **Shape:** 105,982 rows × 57 columns, no missing values
- **Target distribution:** label 0: 21,681 | label 1: 10,268 | label 2: 65,526 | label 4: 8,507
- **Node imbalance:** `boundary` node has 84,607 rows (80%), all other nodes have 4,275 each
- **Scenario split:** 77,772 `training_congestion`, 28,210 `training_portscan`

### 3. Preprocessing & Cleanup
- **Drop constant:** `tcp_synack` is all zeros → removed
- **Leakage columns identified:** 6 pre-engineered composites (`traffic_intensity`, `queue_stress`, `iat_pressure`, `bw_util_ratio`, `drop_rate_trend`, `ctrl_pkt_ratio`) are derived at simulation time — excluded from model features but kept for EDA
- **Port-scan rows:** Retained (not excluded), scenario encoded as binary `is_portscan` feature
- **Node encoding:** `LabelEncoder` maps 6 node IDs to 0–5
- **Temporal sort:** Data sorted per node by timestamp `t`; median step ~0.5s between observations

### 4. Exploratory Visualizations
- **Label distribution bar chart:** Shows strong class imbalance
- **Bandwidth scatter:** `bw_avg_mbps` over time for `boundary` node, colored by congestion label
- **Per-label metric boxplots:** 7 key metrics (`bw_avg_mbps`, `pkt_loss_pct`, `jitter_ms`, `tcp_retransmit_pct`, `rtt_avg_ms`, `syn_rate_pps`, `flow_churn_per_sec`) show separation across congestion levels
- **Correlation heatmap:** Top 15 raw features ranked by absolute correlation with the target

### 5. Feature Engineering (all per-node)
| Category | Features Generated | Count |
|---|---|---|
| **Lag features** | Lagged values of top 5 signals at {1, 3, 5, 10} steps | 20 |
| **Rolling statistics** | Mean, std of 6 signals at windows {5, 10, 30}; rolling max of 3 signals at {5, 10} | 42 |
| **Rate-of-change** | Delta1, delta5, pct_change for 5 signals | 15 |
| **Volatility (CV)** | Coeff. of variation for bandwidth & loss at {10, 30} | 4 |
| **Burst indicators** | Relative burst for bandwidth, SYN rate, flow churn; SYN flood flag | 4 |
| **Load ratios** | Util imbalance, util max, abs bw asymmetry, loss/mbps, retransmit×load, jitter/mbps | 6 |
| **Temporal** | `t_log`, `t_relative` (position within node timeline) | 2 |

**Total: 140 engineered features** (including 47 original non-leaking columns).

#### Forecasting Target
`future_congestion_label` created by shifting `congestion_label` **5 steps forward** per node — the model predicts congestion ~3 seconds into the future.

#### Causal Design
All rolling windows are right-aligned (trailing); lags and diffs look backward; the target is shifted forward. No future leakage.

### 6. Feature Set Definition
Excluded columns: `congestion_label`, `node_id`, `t`, `t_diff`, `split`, 6 leakage composites, `label_lag1`, `label_roll_mean_5`, `tcp_health_score`. Final count: **140 numeric features**.

### 7. Train / Validation Split (Temporal)
- 80/20 chronological split **per node** (no random shuffle)
- Validated chronologically after training data within each node
- **Result:** 84,761 train rows, 21,191 validation rows
- Validation set has no label 1 rows (distribution mismatch from temporal shift)

### 8. Baseline Model — Random Forest
- **Params:** 200 trees, unlimited depth, `min_samples_leaf=5`, `class_weight='balanced'`
- **Validation performance:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| label_0 | 1.00 | 1.00 | 1.00 | 4,270 |
| label_1 | 0.00 | 0.00 | 0.00 | 0* |
| label_2 | 0.96 | 0.92 | 0.94 | 15,605 |
| label_4 | 0.46 | 0.50 | 0.48 | 1,316 |

**Overall accuracy: 91%** (macro avg F1: 0.61, weighted avg F1: 0.92)

*Label 1 absent from validation set due to temporal split distribution shift.

- **Confusion matrix:** Classes 0 and 2 predicted well; class 4 has significant confusion with class 2
- **Top features:** `bw_avg_mbps_roll_mean` variants, `tcp_retransmit_pct_lag`, `flow_churn_per_sec_burst`, `syn_rate_pps_roll_max`, `jitter_ms_delta1`, `pkt_loss_pct_cv`, `retransmit_load`, `util_imbalance`, `loss_per_mbps`

### 9. Anomaly / Port-Scan Analysis
SYN and RST rate distributions heavily overlap between congestion and port-scan scenarios (mean SYN rate: 2.041 vs 2.008 pps). Confirms that port-scan rows are safe to keep — they do not form a distinct cluster.

### 10. Output
- `congestion_features_engineered.csv` — 105,952 rows × 144 columns (140 features + metadata)
- `feature_list.csv` — list of 140 feature column names

## Model Export

The trained RandomForest model is serialized to `congestion_rf_model.pt` using `torch.save()`. The saved dictionary contains:

```python
{
    'model': RandomForestClassifier,        # Trained sklearn model
    'feature_columns': list,                # 140 feature column names
    'node_encoder': LabelEncoder,           # Fitted node ID encoder
    'forecast_horizon': int,                # 5 steps ahead
    'metrics': dict,                        # Train/val row counts
}
```

To load for inference:

```python
import torch
data = torch.load('congestion_rf_model.pt', weights_only=False)
rf_model = data['model']
feature_cols = data['feature_columns']
node_encoder = data['node_encoder']
```

## Recommendations

| Area | Suggestion |
|---|---|
| **Model upgrade** | LightGBM / XGBoost for better handling of imbalance and speed |
| **Ordl regression** | Treat labels as ordinal (0 < 1 < 2 < 4) with thresholded outputs |
| **Deep learning** | Temporal Fusion Transformer or LSTM for multi-node sequence modeling |
| **Anomaly detection** | Isolation Forest sidecar on SYN/RST/flow burst features |
| **Forecast horizon** | 1–5 steps (~0.5–3s) for real-time operational use |
