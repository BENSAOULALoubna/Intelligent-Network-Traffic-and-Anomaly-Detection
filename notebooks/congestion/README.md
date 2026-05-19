# Network Congestion Prediction — Feature Engineering

Multi-class congestion classification on simulated network traffic data, using temporal feature engineering and a Random Forest baseline.

## Project Structure

```
├── congestion_forecasting_notebook.ipynb   # Main notebook
├── feature_list.csv                        # Engineered feature columns
├── requirements.txt                        # Python dependencies
└── README.md
```

## Dataset

The notebook expects `congestion_training.csv` in the same directory. The file is **not included in this repo** — download it from [(https://drive.google.com/drive/folders/16Keq9_UbZu3-9YVyX3OrdOad1YO4CK9_?usp=drive_link)] and place it in the project root.

| Property | Value |
|---|---|
| Rows | 105,982 |
| Columns | 57 |
| Nodes | 6 (`boundary`, `core0–2`, `edge0–1`) |
| Target | `congestion_label` ∈ {0, 1, 2, 4} |

## Setup

```bash
pip install -r requirements.txt
jupyter notebook congestion_forecasting_notebook.ipynb
```

## What the Notebook Does

1. **Inspection** — shape, dtypes, missing values, label distribution
2. **Preprocessing** — drops constant/leaking columns, encodes categoricals, sorts by node + timestamp
3. **EDA** — label distribution, per-label metric boxplots, correlation heatmap
4. **Feature Engineering** — lag features, rolling mean/std/max, rate-of-change, volatility (CV), burst indicators, load ratios, temporal features (all computed per node to avoid cross-node contamination)
5. **Forecasting target** — `future_congestion_label` via 5-step forward shift per node
6. **Train/val split** — chronological 80/20 split per node (no random shuffle)
7. **Baseline model** — Random Forest with `class_weight='balanced'`
8. **Output** — saves `congestion_features_engineered.csv` and `feature_list.csv`

## Key Design Decisions

- **Leakage columns excluded** — `traffic_intensity`, `queue_stress`, `iat_pressure`, `bw_util_ratio`, `drop_rate_trend`, `ctrl_pkt_ratio` are pre-computed simulation composites and are kept for EDA only, not used as model features.
- **Port-scan rows kept** — the `training_portscan` scenario shares similar feature/label distributions with congestion rows and is encoded as a binary `is_portscan` feature rather than dropped.
- **Temporal split** — validation data is always chronologically after training data within each node, preventing future leakage.
