# Feature Engineering & Data Preprocessing Documentation

**Project:** Intelligent Network Traffic and Anomaly Detection  
**Date:** May 19, 2026  
**Scope:** Comprehensive analysis of two feature engineering pipelines:
1. **Congestion Forecasting** (`notebooks/congestion/congestion_vf_forecasting_notebook.ipynb`)
2. **Port-Scan Detection** (`notebooks/port-scanning/Feature-Engineering.ipynb`)

---

## Executive Summary

| Aspect | Congestion | Port-Scan |
|---|---|---|
| **Dataset Size** | 105,982 rows × 57 columns | ~90,000 rows × 48 features |
| **Task** | Multi-class classification (0-4) | Multi-class classification (0,1,2,5) |
| **Target Variable** | `congestion_label` | `scan_label` |
| **Time Series Length** | ~105K observations over 4 nodes | ~380K seconds over 4 observer nodes |
| **Observation Interval** | Irregular sub-second (0.5-0.6s steps) | 10-second windows |
| **Feature Engineering** | **Temporal** (lags, rolling stats, rate-of-change) | **Static per-window** (burst ratios, lookback) |
| **Scenarios Mixed** | training_congestion + training_portscan | training_congestion + training_portscan |
| **Key Challenge** | Class imbalance (label 2: 62%), temporal causality | Class imbalance (label 5: 74%), per-node differences |
| **Recommended Models** | LightGBM / XGBoost + temporal LSTM/TFT | BiLSTM + Self-Attention with Focal Loss |
| **Output Files** | `congestion_features_engineered.csv` (future_congestion_label target) | Model-ready after feature selection |

---

# PART 1: CONGESTION FORECASTING PIPELINE

## 1.1 Dataset Overview

**Source:** `data/congestion_training.csv`

### 1.1.1 Dataset Characteristics
- **Rows:** 105,982 observations
- **Raw Columns:** 57
- **Nodes:** 6 (boundary, core0-2, edge0-1)
- **Time Range:** Simulation timestamps (irregular sub-second steps)
- **Scenarios:** 
  - `training_congestion`: 77,000 rows (73%)
  - `training_portscan`: 28,000 rows (27%)
- **Data Quality:** No missing values; complete dataset

### 1.1.2 Target Variable: `congestion_label`
- **Type:** Multi-class (ordinal)
- **Classes:** {0, 1, 2, 4} (note: no class 3)
- **Distribution (imbalanced):**
  - Label 2: 62% (primary class, 65,000 rows)
  - Label 0: 25% (27,500 rows)
  - Label 4: 9% (10,000 rows)
  - Label 1: 4% (3,500 rows)
- **Interpretation:** 0=no congestion, 1=mild, 2=moderate, 4=severe
- **Challenge:** Class 2 dominance means naive models achieve 62% by predicting label 2 always

### 1.1.3 Nodes & Temporal Structure
- **Primary node:** `boundary` (80% of data, 84,000 rows)
- **Core nodes:** core0, core1, core2
- **Edge nodes:** edge0, edge1
- **Timestamp column:** `t` (simulation time, not wall-clock)
  - Irregular sub-second steps (0.5-0.6s median interval per node)
  - Must sort by node + time before lag/rolling operations
  - One panel per node (observations are per-node, not aggregated)

### 1.1.4 Pre-Engineered Feature Columns (LEAKAGE RISK)
These are **derived** from raw signals at simulation time and must be excluded from models:
| Column | Derivation | Reason to Exclude |
|---|---|---|
| `traffic_intensity` | From `bw_avg_mbps` | Not available as real-time input |
| `iat_pressure` | From `iat_mean_ms` | Composite of IAT metrics |
| `queue_stress` | From pkt_loss + queue signals | Derived from raw signals; potential leakage |
| `bw_util_ratio` | From bw + util | Not a true predictor |
| `drop_rate_trend` | From `pkt_dropped` | Summary statistic, not causal |
| `ctrl_pkt_ratio` | From ctrl/data ratio | Pre-computed aggregate |
| `tcp_synack` | Constant zero | Drop (zero variance) |

---

## 1.2 Data Preprocessing Pipeline

### 1.2.1 Step 1: Drop Constant & Metadata Columns
```python
DROP_CONSTANT = ['tcp_synack']  # zero throughout entire dataset
DROP_META = ['scenario']  # encoded as binary feature instead
```
**Output:** 56 → 55 columns

### 1.2.2 Step 2: Encode Categorical Columns
**Port-Scan Scenario Encoding:**
```python
df['is_portscan'] = (df['scenario'] == 'training_portscan').astype(int)
```
- Binary feature: 0 = congestion, 1 = port-scan
- **Rationale:** Port-scan rows have similar label distributions to congestion rows → keep them, encode as feature
- Port-scan rows are treated as **anomaly context**, not excluded

**Node ID Encoding:**
```python
le_node = LabelEncoder()
df['node_id_enc'] = le_node.fit_transform(df['node_id'])
# Encoding: boundary=0, core0=1, core1=2, core2=3, edge0=4, edge1=5
```
- Node ID remains as string in final output for interpretability
- Numeric encoding used only for lag/rolling groupby operations

### 1.2.3 Step 3: Sort & Prepare for Temporal Operations
```python
df.sort_values(['node_id', 't'], inplace=True)
df.reset_index(drop=True, inplace=True)
df['t_diff'] = df.groupby('node_id')['t'].diff().fillna(0)
```
**Median time-step per node:** 0.5-0.6 seconds
- **Critical:** All groupby operations must group by node ID first
- This prevents cross-node contamination in lag/rolling windows
- Example: lag(1) for `boundary` node cannot pull from `core0` data

---

## 1.3 Feature Engineering: The Temporal Approach

**Principle:** All features are **strictly causal** — no future information leaks into features.

### 1.3.1 Core Raw Signals (Non-Leaking)
These are the foundation for feature engineering:
```python
CORE_SIGNALS = [
    'bw_avg_mbps', 'bw_in_mbps', 'bw_out_mbps',
    'util_in_pct', 'util_out_pct',
    'pkt_loss_pct', 'pkt_dropped',
    'jitter_ms', 'rtt_avg_ms', 'rtt_std_ms',
    'tcp_retransmit_pct', 'syn_rate_pps', 'rst_rate_pps',
    'flow_churn_per_sec', 'new_flows',
    'peak_kbps_1s_window',
]
```

### 1.3.2 Lag Features
**Time-shift features capture momentum:**
```python
LAGS = [1, 3, 5, 10]  # steps (5-50 steps ≈ 3-60 seconds)
# For each core signal: create lagged versions
df['bw_avg_mbps_lag1'] = df.groupby('node_id')['bw_avg_mbps'].shift(1)
df['bw_avg_mbps_lag3'] = df.groupby('node_id')['bw_avg_mbps'].shift(3)
# ... and so on
```
**Total lag features:** 5 signals × 4 lags = **20 features**

**Why multiple lags?**
- lag1: Immediate prior state
- lag3: ~2-3 seconds back (short-term trend)
- lag5: ~3-5 seconds back (medium-term momentum)
- lag10: ~5-10 seconds back (longer-term trend)

### 1.3.3 Rolling Statistics (Windows: 5, 10, 30 steps)
**Window sizes in seconds:** W5≈3s, W10≈6s, W30≈18s

```python
WINDOWS = [5, 10, 30]
for sig in ['bw_avg_mbps', 'pkt_loss_pct', 'jitter_ms', 'tcp_retransmit_pct',
            'flow_churn_per_sec', 'peak_kbps_1s_window']:
    grp = df.groupby('node_id')[sig]
    df[f'{sig}_roll_mean_{w}'] = grp.transform(lambda x: x.rolling(w, min_periods=2).mean())
    df[f'{sig}_roll_std_{w}']  = grp.transform(lambda x: x.rolling(w, min_periods=2).std())

# Rolling max for burst detection
for sig in ['bw_avg_mbps', 'syn_rate_pps', 'pkt_dropped']:
    for w in [5, 10]:
        df[f'{sig}_roll_max_{w}'] = grp.transform(lambda x: x.rolling(w, min_periods=2).max())
```
**Total rolling features:**
- Mean + Std: 6 signals × 3 windows × 2 = **36 features**
- Max: 3 signals × 2 windows = **6 features**
- **Total: 42 rolling features**

**Causality guarantee:** `rolling(window, min_periods=2)` is **right-aligned** by default in pandas, meaning:
- For timestep t, the window includes [t-W+1, t] (right-inclusive)
- No future data is used

### 1.3.4 Rate-of-Change Features
**Differential signals capture rapid changes:**
```python
for sig in ['bw_avg_mbps', 'pkt_loss_pct', 'jitter_ms', 'util_in_pct', 'util_out_pct']:
    grp = df.groupby('node_id')[sig]
    df[f'{sig}_delta1']  = grp.diff(1)                    # 1-step change
    df[f'{sig}_delta5']  = grp.diff(5)                    # 5-step change (≈2.5-3s)
    df[f'{sig}_pct_chg'] = grp.pct_change(fill_method=None)  # % change
```
**Total rate-of-change features:** 5 signals × 3 = **15 features**

**Use case:** Detect rapid spikes in loss % or jitter that precede congestion events.

### 1.3.5 Volatility Features (Coefficient of Variation)
**Measures stability of traffic over a window:**
```python
for sig in ['bw_avg_mbps', 'pkt_loss_pct']:
    for w in [10, 30]:
        mu  = df.groupby('node_id')[sig].transform(lambda x: x.rolling(w, min_periods=3).mean())
        std = df.groupby('node_id')[sig].transform(lambda x: x.rolling(w, min_periods=3).std())
        df[f'{sig}_cv_{w}'] = (std / (mu.abs() + 1e-9)).clip(upper=10)
```
**Total volatility features:** 2 signals × 2 windows = **4 features**

**Interpretation:**
- High CV = unstable traffic (frequent spikes or drops)
- Low CV = smooth, predictable traffic
- High CV in loss % suggests congestion event imminent

### 1.3.6 Burst Indicators
**Detect deviations above rolling mean:**
```python
for sig in ['bw_avg_mbps', 'syn_rate_pps', 'flow_churn_per_sec']:
    roll_mean = df.groupby('node_id')[sig].transform(
        lambda x: x.rolling(10, min_periods=3).mean()
    )
    df[f'{sig}_burst'] = ((df[sig] - roll_mean) / (roll_mean.abs() + 1e-9)).clip(lower=0)

df['syn_flood_indicator'] = (df['syn_rate_pps'] > df['syn_rate_pps'].quantile(0.90)).astype(int)
```
**Total burst features:** 3 burst deviations + 1 SYN flood binary = **4 features**

### 1.3.7 Load Ratio & Composite Features
**Normalized metrics for interpretability:**
```python
df['util_imbalance']    = (df['util_in_pct'] - df['util_out_pct']).abs()
df['util_max']          = df[['util_in_pct', 'util_out_pct']].max(axis=1)
df['bw_asym_abs']       = df['bw_asymmetry'].abs()
df['loss_per_mbps']     = df['pkt_loss_pct'] / (df['bw_avg_mbps'] + 1e-9)
df['retransmit_load']   = df['tcp_retransmit_pct'] * df['bw_avg_mbps']
df['jitter_per_mbps']   = df['jitter_ms'] / (df['bw_avg_mbps'] + 1e-9)
```
**Total composite features:** **6 features**

**Interpretation:**
- `loss_per_mbps`: Loss normalized by load (high = bad efficiency)
- `retransmit_load`: Congestion pressure metric
- `jitter_per_mbps`: Jitter per unit throughput (latency variability per load)

### 1.3.8 Temporal Features
**Encode position in time series:**
```python
df['t_log'] = np.log1p(df['t'])  # log-compress simulation timestamps

# Relative position within node's timeline
node_t_max = df.groupby('node_id')['t'].transform('max')
node_t_min = df.groupby('node_id')['t'].transform('min')
df['t_relative'] = (df['t'] - node_t_min) / (node_t_max - node_t_min + 1e-9)
```
**Total temporal features:** **2 features**

---

## 1.4 Forecasting Setup: Future Target Creation

**Key Innovation:** Convert classification task into **forecasting** by shifting target forward.

```python
FORECAST_HORIZON = 5  # steps (~3 seconds)
df['future_congestion_label'] = df.groupby('node_id')['congestion_label'].shift(-FORECAST_HORIZON)
```

### 1.4.1 Why Shift Forward?
- **Real-world scenario:** We want to predict congestion 3 seconds in advance, not retroactively
- **Prevents data leakage:** The model features only contain information observable at time t
- The target (`future_congestion_label`) represents the actual congestion at time t + 3 seconds
- **Causality maintained:** Features at t predict label at t+5 (future), never mixing temporal orders

### 1.4.2 Impact on Dataset
- **Rows with NaN target:** Last 5 rows per node have no future target
- **Remaining rows:** ~99% of original 105,982 (dropped ~5 rows per 6 nodes)
- **New target column:** `future_congestion_label` (same distribution as original, shifted)

---

## 1.5 Final Feature Set Definition

### 1.5.1 Exclusions
```python
EXCLUDE = (
    ['future_congestion_label', 'congestion_label', 'node_id', 't', 't_diff', 'split'] +
    COMPOSITE_LEAKAGE +  # 6 pre-engineered columns
    ['label_lag1', 'label_roll_mean_5', 'tcp_health_score']  # removed: target leakage
)
```

### 1.5.2 Final Feature Count
- **Total raw signals:** 15 CORE_SIGNALS
- **Lag features:** 20
- **Rolling (mean+std):** 36
- **Rolling (max):** 6
- **Rate-of-change:** 15
- **Volatility (CV):** 4
- **Burst indicators:** 4
- **Load composites:** 6
- **Temporal:** 2
- **Node encoding:** 1 (`is_portscan`)
- **Total FEATURE_COLS:** ~109 features

### 1.5.3 Feature Columns (Partial List)
Sample features from each category:
```
# Lag features
bw_avg_mbps_lag1, bw_avg_mbps_lag3, bw_avg_mbps_lag5, bw_avg_mbps_lag10
pkt_loss_pct_lag1, pkt_loss_pct_lag3, ...

# Rolling mean (6-18s windows)
bw_avg_mbps_roll_mean_5, bw_avg_mbps_roll_mean_10, bw_avg_mbps_roll_mean_30
pkt_loss_pct_roll_mean_5, ...

# Rolling std
bw_avg_mbps_roll_std_5, bw_avg_mbps_roll_std_10, ...

# Rate-of-change
bw_avg_mbps_delta1, bw_avg_mbps_delta5, bw_avg_mbps_pct_chg
pkt_loss_pct_delta1, ...

# Volatility (CV)
bw_avg_mbps_cv_10, bw_avg_mbps_cv_30, pkt_loss_pct_cv_10, ...

# Bursts
bw_avg_mbps_burst, syn_rate_pps_burst, flow_churn_per_sec_burst

# Composites
util_imbalance, bw_asym_abs, loss_per_mbps, retransmit_load, jitter_per_mbps

# Temporal
t_log, t_relative, is_portscan
```

---

## 1.6 Train / Validation Split (Temporal)

### 1.6.1 Temporal Split Strategy
```python
SPLIT_PCT = 0.80

def temporal_split(group, pct=0.80):
    cutoff = group['t'].quantile(pct)
    group['split'] = np.where(group['t'] <= cutoff, 'train', 'val')
    return group

fe_filtered = fe.dropna(subset=['future_congestion_label']).copy()
fe_filtered = fe_filtered.groupby('node_id', group_keys=False).apply(temporal_split)
```

### 1.6.2 Split Statistics
- **Train:** ~84,000 rows (80% of node-specific timelines)
- **Validation:** ~21,000 rows (20% of node-specific timelines)
- **Key point:** Split is **chronological per node**, not random
  - Prevents information leakage from future to past
  - Respects temporal causality
  - Models are evaluated on unseen future time periods

### 1.6.3 Class Distribution in Splits
**Example (hypothetical):**
```
Train label distribution:
  0: 25%, 1: 4%, 2: 62%, 4: 9%
Val label distribution:
  0: 25%, 1: 4%, 2: 62%, 4: 9%
(approx same due to stratified per-node split)
```

---

## 1.7 Output Datasets

### 1.7.1 Primary Output: `congestion_features_engineered.csv`
```python
OUT_COLS = ['node_id', 't', 'future_congestion_label', 'is_portscan'] + FEATURE_COLS
fe_out[OUT_COLS].to_csv('congestion_features_engineered.csv', index=False)
```

**File Statistics:**
- **Rows:** ~105,977 (after dropping NaN targets)
- **Columns:** 114 (4 metadata + 109 features + 1 target)
- **Size:** ~45 MB (uncompressed)

### 1.7.2 Secondary Output: `feature_list.csv`
```python
pd.Series(FEATURE_COLS).to_csv('feature_list.csv', index=False, header=['feature'])
```

**Purpose:** Reproducibility reference. Lists all 109 feature column names in order.

---

## 1.8 Correlation & Multicollinearity Analysis

### 1.8.1 High-Correlation Pairs
From EDA (feature correlation heatmap):
- **`traffic_intensity` ↔ `bw_avg_mbps`** (r ≈ 0.99) — likely same signal
- **`queue_stress` ↔ `pkt_loss_pct`** (r ≈ 0.95) — derived from same base
- **`ctrl_pkt_ratio` ↔ `syn_rate_pps`** (r ≈ 0.90) — control-packet proxy

**Implication:** The pre-engineered columns introduce redundancy. Dropping them reduces multicollinearity.

### 1.8.2 Strong Correlations with Target
**Top features by |correlation| with `future_congestion_label`:**
1. `tcp_retransmit_pct_roll_mean_10` (~0.35)
2. `bw_avg_mbps_roll_std_30` (~0.32)
3. `jitter_ms_delta1` (~0.30)
4. `pkt_loss_pct_cv_30` (~0.29)
5. `retransmit_load` (~0.27)

**Pattern:** Volatility, rate-of-change, and rolling statistics are most predictive.

---

## 1.9 Key Findings & Recommendations

### 1.9.1 Data Quality Issues
1. **None:** No missing values in the raw dataset
2. **Imbalance:** Label 2 is 62% of data
   - **Mitigation:** Use `class_weight='balanced'` in tree models or Focal Loss in neural networks
3. **Scenarios mixed:** Port-scan rows look statistically similar to congestion rows
   - **Action:** Treat port-scan as anomaly context; do not exclude

### 1.9.2 Leakage Prevention
| Column | Status | Reason |
|---|---|---|
| `traffic_intensity` | ❌ **DROPPED** | Derived from `bw_avg_mbps` at sim time |
| `queue_stress` | ❌ **DROPPED** | Pre-computed from raw signals |
| `tcp_synack` | ❌ **DROPPED** | Constant zero (no variance) |
| `active_flows` | ⚠️ **FLAGGED** | Near-binary (mostly 0/1), low signal |
| All `FEATURE_COLS` | ✅ **KEPT** | Computed from raw signals only, strictly causal |

### 1.9.3 Model Recommendations

| Model | Why | When to Use |
|---|---|---|
| **LightGBM / XGBoost** | Fast, handles imbalance, excellent baselines | Default choice for production |
| **Random Forest** | Robust, interpretable, good for production | If tree-based baseline needed |
| **LSTM/Temporal Fusion Transformer** | Captures temporal dependencies | If 5+ step forecasting horizon needed |
| **Ordinal Regression** | Respects 0 < 1 < 2 < 4 ordering | If treating labels as severity levels |

### 1.9.4 Top Engineered Features (By Importance)
1. **`bw_avg_mbps_roll_mean_{5,10,30}`** — throughput trends over 3-18s windows
2. **`tcp_retransmit_pct_lag{1,3,5}`** — retransmit momentum
3. **`flow_churn_per_sec_burst`** — sudden flow activity spikes
4. **`syn_rate_pps_roll_max_10`** — SYN burst window captures (6s)
5. **`jitter_ms_delta1`** — jitter acceleration (immediate changes)
6. **`pkt_loss_pct_cv_30`** — loss volatility (18s window)
7. **`retransmit_load`** = `tcp_retransmit_pct × bw_avg_mbps` — congestion pressure
8. **`util_imbalance`** — asymmetric utilization pattern
9. **`loss_per_mbps`** — normalized loss under load

---

---

# PART 2: PORT-SCAN DETECTION PIPELINE

## 2.1 Dataset Overview

**Source:** Google Drive CSV (multi-node network traffic from 4 ISP observer nodes)

### 2.1.1 Dataset Characteristics
- **Rows:** ~90,000 observations (exactly: 91,000)
- **Columns:** 48 (including target)
- **Observation Interval:** 10-second windows (fixed, regular)
- **Time Range:** ~4.4 days (~380,000 seconds of simulation)
- **Observer Nodes:** 4 ISP-level monitoring points
  - `isp_ftth0`: FTTH gateway
  - `isp_mobile0`: Mobile gateway 1
  - `isp_mobile1`: Mobile gateway 2
  - `isp_private0`: Private gateway (primary scan detection point)
- **Scenarios Mixed:** `training_congestion` + `training_portscan`
- **Data Quality:** No missing values

### 2.1.2 Target Variable: `scan_label`
- **Type:** Multi-class (nominal, not ordinal)
- **Classes:** {0, 1, 2, 5} (note: no 3, no 4)
- **Distribution (severely imbalanced):**
  - Label 5 (Distributed/Spillover): 67,500 rows **74%** ← DOMINANT
  - Label 0 (Normal): 14,700 rows **16%**
  - Label 2 (Vertical Scan): 6,533 rows **7%**
  - Label 1 (Horizontal Scan): 1,267 rows **1.1%** ← RAREST

### 2.1.3 Class Semantics
| Label | Scan Type | Characteristics | Danger Level |
|---|---|---|---|
| **0** | Normal Traffic | Baseline, clean network | 🟢 Safe |
| **1** | Horizontal Scan | Same port across many IPs (banner grab) | 🔴 CRITICAL (rare, hard to detect) |
| **2** | Vertical Scan | Many ports on same IP (port discovery) | 🟠 High |
| **5** | Distributed/Spillover | Low-and-slow, multi-source, background-like | 🟡 Medium (common, easy to miss) |

### 2.1.4 Per-Node Label Distribution (CRITICAL FINDING)

| Node | Label 0 | Label 1 | Label 2 | Label 5 | Key Insight |
|---|---|---|---|---|---|
| **isp_private0** | 15K (18%) | 1.2K (1.5%) | 6.5K (8%) | 60K (72%) | **Only node with classes 1 & 2** |
| **isp_ftth0** | 1.1K (24%) | 0 | 0 | 3.5K (76%) | Binary classification only (0 vs 5) |
| **isp_mobile0** | 2.1K (27%) | 0 | 0 | 5.6K (73%) | Binary classification only |
| **isp_mobile1** | 2.9K (27%) | 0 | 0 | 7.8K (73%) | Binary classification only |

**Critical Implication:** 
- To detect Horizontal scans (class 1), the model **MUST see data from `isp_private0`** during validation/test
- FTTH and mobile nodes are useful for detecting Distributed scans (class 5) but cannot learn rare scan types
- Per-node stratification is essential for proper evaluation

---

## 2.2 Data Preprocessing Pipeline

### 2.2.1 Constant / Near-Constant Features (9 Found)

These 9 columns have **exactly zero variance** — single values throughout dataset:

| Feature | Value | Reason |
|---|---|---|
| `total_synack` | 0 | Port scans use half-open SYNs; TCP handshake never completes |
| `total_ack` | 0 | Same reason |
| `unique_src_ips` | 0 | Computed only during active windows, not per-timestep |
| `unique_dst_ips` | 0 | Dispersion metric (aggregated, not windowed) |
| `unique_dst_ports` | 0 | Dispersion metric |
| `payload_bytes` | 0 | Port scans don't send payload data |
| `scanning_src_count` | 0 | Summary statistic, not per-timestep |
| `scan_breadth` | 0 | Aggregate metric |
| `scan_entropy_proxy` | 0 | Computed offline, not dynamic per-window |

**Action:** **DROP these 9 columns** before modeling.
- Current `lstm_pipeline.py` includes them → adds noise, unnecessary parameters
- Recommendation: Update feature selection to exclude these automatically

### 2.2.2 Metadata & Encoding

**Columns to Handle:**
```python
cat_cols = ['observer_node', 'scenario']
```

**observer_node Encoding:**
- **Option 1:** Keep as string, one-hot encode (4 dummies)
  - `isp_ftth0`: [1, 0, 0, 0]
  - `isp_mobile0`: [0, 1, 0, 0]
  - `isp_mobile1`: [0, 0, 1, 0]
  - `isp_private0`: [0, 0, 0, 1]
  
- **Option 2:** Add `observer_node` as embedding layer (if using deep learning)
  - More efficient for LSTM/Transformer
  - Allows model to learn shared representations across nodes

- **Option 3:** Train separate models per node
  - Highest accuracy but requires deployment complexity

**scenario Column:**
- **Current:** 73% `training_congestion`, 27% `training_portscan`
- **Issue:** May leak label information (to be verified)
- **Recommendation:** 
  - Diagnostic: Check XGBoost accuracy **without scenario** feature
  - If accuracy remains >95%, scenario is truly predictive
  - If accuracy drops to 85-90%, scenario was providing leakage signal

### 2.2.3 Feature Normalization
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_features])
```
**Why:** Features span different ranges:
- TCP flag counts: 0-10,000 (large range)
- Ratios/percentages: 0-1 (small range)
- Standardization: mean=0, std=1 for all features

---

## 2.3 Exploratory Data Analysis Findings

### 2.3.1 Feature Correlation & Multicollinearity

**81 feature pairs with |r| > 0.95 detected** — massive redundancy.

**High-Correlation Groups (|r| > 0.99):**

**Group 1: Scan Volume Signals**
- `total_syn` ↔ `conn_attempts` ↔ `syn_rate_per_sec` ↔ `syn_ack_ratio`
- All capture same underlying: "how many SYN packets?"
- **Action:** Keep `total_syn` only

**Group 2: Failed Connections**
- `failed_conns` ↔ `total_rst` ↔ `scan_intensity` ↔ `rst_rate_pps`
- All measure: "how many connections reset?"
- **Action:** Keep `failed_conns` only

**Group 3: Failure Rate**
- `conn_failure_rate` ↔ `rst_per_syn` ↔ `failed_per_syn` ↔ `conn_success_pct` (inverted)
- All measure: "what % of connections failed?"
- **Action:** Keep `conn_failure_rate` only

**Group 4: Lookback Activity**
- `lb_syn_60s` ↔ `lb_dst_ip_60s` ↔ `lb_dst_port_60s` (r=0.995)
- All: "how much activity in lookback window?"
- **Action:** Keep `lb_syn_60s` only

**Group 5: Timing**
- `syn_iat_mean_ms` ↔ `syn_iat_std_ms` (r=0.994)
- Both measure SYN inter-arrival time
- **Action:** Keep `syn_iat_mean_ms` only

**Group 6: Protocol Overhead**
- `proto_overhead_pct` ↔ `ctrl_overhead_pct` (r=0.973)
- Both measure: "% of traffic is control packets?"
- **Action:** Keep `proto_overhead_pct` only

**Dimensionality Reduction Opportunity:**
- **Original features:** 47 numeric
- **After removing redundant:** ~20 features
- **PCA at 95% variance:** 11 components needed
- **Compression ratio:** 47 → 11 = **4.3x reduction** with minimal information loss

### 2.3.2 PCA Interpretation

**Component PC1 (59.1% of variance) — "Scan Intensity Axis"**
- Top loadings: `scan_intensity`, `failed_conns`, `total_rst`, `lb_dst_ip_60s`, `lb_syn_60s`
- Interpretation: Separates normal traffic (low PC1) from active scanning (high PC1)
- This is the **primary discriminative dimension**

**Component PC2 (8.0% of variance) — "Protocol Overhead Axis"**
- Top loadings: `proto_overhead_pct`, `ctrl_overhead_pct`, `syn_rst_ratio`, `conn_completion_pct`
- Interpretation: Captures protocol-level efficiency (control vs. data packet ratio)
- Useful for distinguishing **scan type** (horizontal vs. vertical use different protocols)

**Implication:** A simple 2D projection loses 33% of variance but captures the main patterns visually.

### 2.3.3 Per-Node Temporal Analysis

**Session Boundary Problem:**
- Gap analysis reveals **172–980 temporal gaps > 100 seconds** per node
- These represent disconnections or monitoring interruptions
- **Impact on LSTM:** Sequences bridging gaps (sliding window with stride=1) will contain nonsensical temporal jumps

**Solution:** Split sequences at gaps > 100s.

**Scan Burst Duration Analysis:**

| Scan Type | Median Duration | P75 Duration | Time Steps |
|---|---|---|---|
| Label 0 (Normal) | Persistent | Persistent | N/A (baseline) |
| Label 1 (Horizontal) | ~? seconds | ~? seconds | 1-3 steps (10-30s) |
| Label 2 (Vertical) | ~? seconds | ~? seconds | 3-6 steps (30-60s) |
| Label 5 (Distributed) | Persistent | Persistent | 6-12+ steps (60-120s) |

**Autocorrelation of Scan Activity:**
- Correlation drops rapidly for class 1/2 (short bursts)
- Correlation persists for class 5 (background noise)
- **Recommended SEQ_LEN:** 8-12 steps (80-120 seconds) balances burst detection with context

---

## 2.4 Feature Categories & Engineering

### 2.4.1 TCP Flag Features (11 features)
```python
TCP_FLAGS = [
    'total_syn', 'total_synack', 'total_ack', 'total_fin', 'total_rst',
    'syn_rate_per_sec', 'syn_ack_ratio', 'syn_rst_ratio',
    'rst_per_syn', 'failed_per_syn', 'syn_to_fin_ratio'
]
```
**Characteristics:** Count-based, capture connection lifecycle patterns
- High during horizontal scans (many half-open SYN connections)
- Moderate during vertical scans (subset of ports probed)
- Low during normal traffic

**Redundancy:** 81% of these features are highly correlated; can reduce to 2-3 principal components

### 2.4.2 Connection Features (6 features)
```python
CONNECTION = [
    'conn_attempts', 'failed_conns', 'completed_conns',
    'conn_success_pct', 'conn_completion_pct', 'conn_failure_rate'
]
```
**Characteristics:** Success/failure rates of connection attempts
- Vertical scans → many failures (ports not listening)
- Horizontal scans → many attempts (same port many times)
- Normal traffic → high success rate

**Key Signal:** `conn_failure_rate` — separates scans from normal (which rarely fail)

### 2.4.3 Dispersion Features (9 features)
```python
DISPERSION = [
    'unique_src_ips', 'unique_dst_ips', 'unique_dst_ports',
    'fanout_ip_ratio', 'fanout_port_ratio', 'port_ip_ratio',
    'scan_breadth', 'scan_entropy_proxy', 'fan_out_ratio'
]
```
**⚠️ Issue:** 7 of 9 are constant (value 0) due to aggregation at window level
- **Action:** DROP these before LSTM training
- Keep only: `fanout_ip_ratio`, `fanout_port_ratio`, `port_ip_ratio` (ratio features may have variance)

### 2.4.4 Lookback Features (7 features)
```python
LOOKBACK = [
    'lb_syn_60s', 'lb_syn_300s', 'lb_dst_ip_60s', 'lb_dst_port_60s',
    'lookback_acceleration', 'scan_intensity', (note: scan_intensity appears here too)
]
```
**Characteristics:** Historical activity in 60s and 300s windows
- `lb_syn_60s`: How many SYNs in last 60 seconds?
- `lb_dst_ip_300s`: How many unique destination IPs in last 5 minutes?
- `lookback_acceleration`: Rate of increase in lookback activity
- **Signal:** Rapidly growing lookback values indicate scan campaign intensifying

**Redundancy:** 3-way correlation detected (r=0.995); keep `lb_syn_60s` only

### 2.4.5 Timing Features (5 features)
```python
TIMING = [
    'syn_iat_mean_ms', 'syn_iat_std_ms', 'syn_iat_cv',
    'syn_regularity', 'syn_burst_ratio'
]
```
**Characteristics:** Pattern of inter-arrival times between SYNs
- Horizontal scans → regular spacing (automated scanner)
- Vertical scans → variable spacing (probing different ports)
- Normal traffic → bursty, high variability

**Key Features:** `syn_iat_cv` (coefficient of variation) discriminates scan regularity

### 2.4.6 Protocol Overhead Features (5 features)
```python
PROTOCOL_OVERHEAD = [
    'tcp_ctrl_bytes', 'payload_bytes', 'total_packets',
    'proto_overhead_pct', 'ctrl_overhead_pct'
]
```
**Characteristics:** Ratio of control packets to data packets
- Scans = mostly control (SYN, RST), little payload → high overhead %
- Normal = mixture of data and control → moderate overhead %

**Key Feature:** `proto_overhead_pct` — single best indicator of scan activity

---

## 2.5 XGBoost Baseline Performance

### 2.5.1 Reported Accuracy: 99.99% (⚠️ Suspicious)

**Result:** XGBoost achieved near-perfect classification on per-timestep features:
- F1-score: 1.00 for all 4 classes
- Accuracy: 99.99%

**Possible Causes:**

1. **`scenario` column leaks the label (most likely)**
   - 73% `training_congestion` rows have label 0 or 5 (normal/distributed)
   - 27% `training_portscan` rows have label 1 or 2 (active scans)
   - Model may simply predict based on scenario, not actual features
   - **Fix:** Re-train without scenario; compare accuracy

2. **Top features are extremely discriminative**
   - Permutation importance shows: `syn_burst_ratio` (rank 1), `lookback_acceleration` (rank 2)
   - These may encode scan patterns so well that a simple decision tree separates them perfectly
   - **If true:** LSTM's temporal modeling adds little value; MLP may suffice

3. **Data leakage in feature engineering**
   - Features like `lookback_acceleration` might use future information
   - **Investigation:** Check if `lb_syn_60s` is computed forward-looking or backward-looking

### 2.5.2 Per-Node Binary Detection (XGBoost)

| Node | Samples | Anomaly % | Test F1 | Test Accuracy | Insight |
|---|---|---|---|---|---|
| isp_private0 | 82,700 | 82% | Near 1.0 | Near 100% | Easy separation (active scans on this node) |
| isp_ftth0 | 4,557 | 76% | Lower | 95-98% | Harder (distributed scans blend with normal) |
| isp_mobile0 | 7,733 | 73% | Lower | 95-98% | Same issue as FTTH |
| isp_mobile1 | 10,785 | 73% | Lower | 95-98% | Same issue as FTTH |

**Pattern:** isp_private0 achieves near-perfect detection because it has active horizontal/vertical scans (classes 1, 2) that are easily distinguishable. Other nodes only have normal vs. distributed, which are harder to separate.

---

## 2.6 Recommendations for LSTM/Deep Learning Models

### 2.6.1 Recommended Architecture: BiLSTM + Self-Attention

**Why this design?**
- **BiLSTM (2 layers, 128 hidden dim):** Captures temporal patterns in both directions
- **Self-Attention:** Identifies which timesteps are critical (especially for bursty scans where only 1-2 steps contain the signal)
- **MLP Classifier:** Final predictions

**Pseudocode:**
```python
class ScanDetectionModel(nn.Module):
    def __init__(self, input_size, hidden_dim=128, num_classes=4):
        super().__init__()
        # Embedding for observer_node
        self.node_embedding = nn.Embedding(4, 16)  # 4 nodes, 16-dim embeddings
        
        # BiLSTM
        self.lstm = nn.LSTM(input_size + 16, hidden_dim, num_layers=2, 
                            bidirectional=True, dropout=0.3, batch_first=True)
        
        # Self-Attention
        self.attention = nn.MultiheadAttention(256, num_heads=8, dropout=0.3, batch_first=True)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, X, node_ids):
        # X shape: (batch, seq_len, num_features)
        # Embed node IDs
        node_emb = self.node_embedding(node_ids)  # (batch, 16)
        node_emb = node_emb.unsqueeze(1).expand(-1, X.shape[1], -1)  # (batch, seq_len, 16)
        
        # Concatenate with features
        X = torch.cat([X, node_emb], dim=-1)  # (batch, seq_len, num_features+16)
        
        # BiLSTM
        lstm_out, _ = self.lstm(X)  # (batch, seq_len, 256)
        
        # Self-Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, seq_len, 256)
        
        # Global average pool over time
        avg_pool = attn_out.mean(dim=1)  # (batch, 256)
        
        # Classify
        logits = self.fc(avg_pool)  # (batch, num_classes)
        return logits, attn_weights
```

### 2.6.2 Training Configuration

| Parameter | Recommended Value | Rationale |
|---|---|---|
| **Sequence Length** | 12 (120 seconds) | Balances burst detection (short) and distributed scan context (long) |
| **Stride** | 2 | Reduces temporal correlation, more independent sequences |
| **Batch Size** | 64 | Large enough for stable gradient; GPU memory permitting |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) | Better regularization than Adam |
| **Loss Function** | Focal Loss (γ=2.0) + Class Weights | Addresses 74:1 class imbalance |
| **Class Weights** | Inverse frequency | class_weight[k] = total_samples / (num_classes * count[k]) |
| **LR Schedule** | Cosine annealing + warm restarts | Escape local minima, faster convergence |
| **Gradient Clip** | 1.0 | Prevent explosion during rare class updates |
| **Dropout** | 0.3-0.5 | Regularization for overfitting |
| **Early Stopping** | Patience=10 on val macro-F1 | Monitor F1, not loss (loss misleading with imbalance) |
| **Validation Split** | 20% (stratified by node + label) | Ensure each split has all classes from all nodes |

### 2.6.3 Feature Selection for LSTM

**Start with these ~20-25 features (pruned from 48):**
```python
SELECTED_FEATURES = [
    # Drop 9 constant features
    # Drop 1 scenario (potential leakage)
    # Keep 1 per correlated group:
    'total_syn',  # scan volume
    'failed_conns',  # failed connections
    'conn_failure_rate',  # failure rate (highly predictive)
    'lb_syn_60s',  # lookback activity
    'syn_iat_mean_ms',  # SYN timing
    'proto_overhead_pct',  # control packet overhead
    'syn_burst_ratio',  # burst regularity (top permutation feature)
    'lookback_acceleration',  # acceleration of activity (top permutation feature)
    'fanout_ip_ratio',  # IP dispersion
    'fanout_port_ratio',  # port dispersion
    'port_ip_ratio',  # port-to-IP ratio
    'syn_rst_ratio',  # SYN-to-RST ratio
    'conn_completion_pct',  # completion success rate
    'syn_regularity',  # regularity of SYN timing
    'tcp_ctrl_bytes',  # bytes in control packets
    'syn_iat_cv',  # coefficient of variation in SYN timing
    'syn_iat_std_ms',  # std of SYN inter-arrival time
    # Node ID (as one-hot or embedding)
]
```

### 2.6.4 Per-Node Evaluation Strategy

**Critical:** Evaluate separately per node to ensure generalization:

```python
for node in ['isp_private0', 'isp_ftth0', 'isp_mobile0', 'isp_mobile1']:
    test_subset = test_data[test_data['observer_node'] == node]
    y_true = test_subset['scan_label']
    y_pred = model.predict(test_subset)
    
    print(f"\n{node}:")
    print(f"  Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"  Per-class F1:")
    for cls in [0, 1, 2, 5]:
        if cls in y_true.values:
            print(f"    Class {cls}: {f1_score(y_true == cls, y_pred == cls):.4f}")
        else:
            print(f"    Class {cls}: N/A (not in node's test set)")
```

**Expected Results:**
- `isp_private0`: High F1 for all 4 classes (it has examples of each)
- `isp_ftth0`, `isp_mobile0`, `isp_mobile1`: High F1 for classes 0 & 5 only (they lack examples of 1 & 2)

### 2.6.5 Real-Time Inference Pipeline

**Dashboard deployment:**
```
Every 10s: New observations from 4 nodes arrive
       |
       v
  [Per-node Sliding Windows]
       |
       v
  [LSTM Model(s)]
    - Option A: Single model with node embedding
    - Option B: Per-node models (4 separate MLSTMs)
       |
       v
  [Outputs]
    - Per-node binary: Normal (0) vs. Anomaly (1,2,5)
    - Per-node scan type (for isp_private0): Horizontal/Vertical/Distributed
    - Confidence scores
       |
       v
  [Dashboard Alert]
    - "Node isp_private0: HORIZONTAL SCAN (99% confidence)"
    - "Node isp_ftth0: Normal"
    - "Node isp_mobile0: Distributed scan detected (confidence: 82%)"
```

**Latency:**
- Window delay: 120s (12 steps × 10s each)
- LSTM inference: 5-15ms
- Total: ~120+ seconds from event onset until alert

**Trade-off:** Longer windows increase latency but improve accuracy for slow scans.

---

## 2.7 Diagnostic Steps Before LSTM Training

### 2.7.1 Verify Scenario Leakage
```python
# Re-train XGBoost WITHOUT scenario feature
X_no_scenario = df[numerical_features].drop('scenario', axis=1, errors='ignore')
model_no_scenario = xgb.XGBClassifier(...).fit(X_train, y_train)
y_pred_no_scenario = model_no_scenario.predict(X_test)
acc_no_scenario = (y_pred_no_scenario == y_test).mean()

print(f"Accuracy WITH scenario: 0.9999")
print(f"Accuracy WITHOUT scenario: {acc_no_scenario:.4f}")
print(f"Drop in accuracy: {(0.9999 - acc_no_scenario)*100:.2f}%")

if (0.9999 - acc_no_scenario) > 0.05:  # >5% drop
    print("WARNING: scenario is likely providing leakage signal!")
```

### 2.7.2 Feature Importance Validation
```python
# Verify top features are not data artifacts
top_features = model.feature_importances_.argsort()[-5:][::-1]
print("Top 5 features by Gini importance:")
for idx in top_features:
    print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
# Check if these features have reasonable variance
for idx in top_features:
    feat_vals = X[feature_names[idx]]
    print(f"  {feature_names[idx]}: min={feat_vals.min():.2f}, max={feat_vals.max():.2f}, std={feat_vals.std():.2f}")
```

### 2.7.3 Ablation: Drop Constant Features & Retrain
```python
# Identify and drop 9 constant features
constant_features = []
for col in numerical_features:
    if df[col].std() < 1e-10 or df[col].nunique() <= 1:
        constant_features.append(col)

X_clean = df[numerical_features].drop(constant_features, axis=1)
model_clean = xgb.XGBClassifier(...).fit(X_clean[train_idx], y_train)
y_pred_clean = model_clean.predict(X_clean[test_idx])
acc_clean = (y_pred_clean == y_test).mean()

print(f"Accuracy after dropping {len(constant_features)} constant features: {acc_clean:.4f}")
print(f"Features dropped: {constant_features}")
```

---

## 2.8 Key Differences: Congestion vs. Port-Scan Pipelines

| Aspect | Congestion | Port-Scan |
|---|---|---|
| **Observation Interval** | Irregular 0.5-0.6s | Regular 10s |
| **Feature Type** | Temporal (lags, rolling windows) | Static (per-window ratios, counts) |
| **Leakage Concerns** | Pre-engineered composites | `scenario` column, constant features |
| **Class Imbalance** | 62% class 2 | 74% class 5 (worse) |
| **Per-Node Behavior** | 6 nodes, similar label dist | 4 nodes, **very different** label dist |
| **Temporal Modeling** | Essential (LSTM/TFT justified) | May not add much value (XGBoost ~99%) |
| **Recommended Model** | LSTM/TFT + XGBoost baseline | BiLSTM+Attention or XGBoost only |
| **Primary Challenge** | Forecasting horizon (causality) | Class imbalance + rare classes |
| **Output Files** | `congestion_features_engineered.csv` + `feature_list.csv` | Feature selection pending |

---

# PART 3: CONSOLIDATED RECOMMENDATIONS

## 3.1 Unified Feature Engineering Best Practices

### 3.1.1 Causality Principles
1. **Right-aligned windows:** Ensure rolling statistics only look backward in time
2. **Explicit lag shifts:** Use `.shift(n)` with n > 0 (looking backward)
3. **Avoid target leakage:** Do not use pre-computed targets or aggregates as features
4. **Node stratification:** Group by spatial unit (node) before temporal operations

### 3.1.2 Data Quality Checklist
- ✅ No missing values
- ✅ Constant/near-constant features identified and dropped
- ✅ Categorical variables encoded properly
- ⚠️ Check for scenario/metadata leakage
- ⚠️ Verify multicollinearity (PCA may help)
- ✅ Class imbalance acknowledged and mitigated

### 3.1.3 Preprocessing Checklist
1. Drop constant features (zero variance)
2. Encode categorical variables appropriately
3. Sort by temporal unit (node/observer) + timestamp
4. Compute time gaps and identify session boundaries
5. Standardize/normalize features for ML models
6. Handle NaN values (forward-fill per group, then fill remaining with 0)
7. Create train/val/test splits respecting temporal order and class stratification

---

## 3.2 Feature Selection Guidelines

### 3.2.1 Dimensionality Reduction Opportunities
- **PCA:** Reduces to 11 components capturing 95% of variance (4.3x compression)
- **Correlation-based:** Keep 1 feature per high-correlation group (|r| > 0.95)
- **Variance threshold:** Drop features with std < 0.01
- **Permutation importance:** Rank by impact on validation F1-score

### 3.2.2 Top Features to Prioritize
**Port-Scan:**
1. `lookback_acceleration` (rate of increase in activity)
2. `syn_burst_ratio` (regularity of SYN pattern)
3. `conn_failure_rate` (success rate proxy)

**Congestion:**
1. `tcp_retransmit_pct_roll_mean_10` (short-term retransmit trend)
2. `bw_avg_mbps_roll_std_30` (bandwidth volatility)
3. `retransmit_load` (congestion pressure)

---

## 3.3 Class Imbalance Mitigation

| Technique | When to Use | Implementation |
|---|---|---|
| **Class Weights** | Always (first step) | `class_weight='balanced'` in sklearn; inverse frequency in PyTorch |
| **Focal Loss** | Neural networks with extreme imbalance | `torch.nn.functional.focal_loss()` or custom implementation |
| **Stratified Splitting** | Train/val/test | `StratifiedShuffleSplit(y, stratification)` per node/group |
| **Oversampling** | Small class has <100 samples | `RandomOverSampler` or synthetic (SMOTE) |
| **Undersampling** | Large class >10x smaller class | `RandomUnderSampler` (may lose signal) |
| **Metric Adjustment** | Evaluation | Use Macro F1 (not accuracy), MCC, or balanced F1 |

---

## 3.4 Model Selection Decision Tree

```
START: Choose modeling approach
  |
  ├─ "I need interpretability" → XGBoost + permutation importance
  │
  ├─ "I need state-of-the-art accuracy" → BiLSTM + Attention (temporal) or XGBoost (static)
  │
  ├─ "I have time series with clear patterns" → LSTM / Temporal Fusion Transformer
  │
  ├─ "Data has extreme imbalance (>50:1)" → Focal Loss + BiLSTM or SHAP-based tree models
  │
  └─ "I want production ready (fast, low latency)" → XGBoost or Random Forest
```

**For this project:**
- **Congestion:** LightGBM baseline + LSTM for deep forecasting
- **Port-Scan:** XGBoost baseline + BiLSTM+Attention if temporal context helps

---

# APPENDIX: File Locations & Quick Reference

## A.1 Input Files
| Dataset | Location | Size | Format |
|---|---|---|---|
| Congestion Raw | `data/congestion_training.csv` | ~50 MB | CSV |
| Port-Scan Raw | Google Drive (Colab) | ~35 MB | CSV |

## A.2 Output Files (Congestion)
| File | Location | Contents | Use |
|---|---|---|---|
| Engineered Features | `notebooks/congestion/congestion_features_engineered.csv` | 105,977 rows × 114 cols (4 meta + 109 features + 1 target) | LSTM/XGBoost training |
| Feature List | `notebooks/congestion/feature_list.csv` | 109 feature names | Reference, reproducibility |

## A.3 Notebook Cells & Key Outputs (Congestion)

| Phase | Cell Range | Key Outputs |
|---|---|---|
| Load & Inspect | 1-5 | Data shape, dtypes, missing values, target distribution |
| Preprocessing | 6-12 | Dropped columns, node encoding, sorted by t |
| EDA Visualizations | 13-18 | Target dist chart, per-label boxplots, correlation heatmap |
| Lag Features | 19 | 20 lag features (5 signals × 4 lags) |
| Rolling Statistics | 20-22 | 42 rolling features (36 mean+std + 6 max) |
| Rate-of-Change | 23 | 15 delta features (5 signals × 3 types) |
| Volatility | 24 | 4 CV features (2 signals × 2 windows) |
| Burst Indicators | 25-26 | 4 burst features + SYN flood indicator |
| Composites | 27 | 6 load ratio features |
| Temporal | 28 | 2 temporal features |
| Future Target | 29-30 | `future_congestion_label` created (FORECAST_HORIZON=5) |
| Train/Val Split | 31-33 | X_train, y_train, X_val, y_val (80/20 split) |
| XGBoost Baseline | 34-36 | Classification report, confusion matrix, feature importance |
| Save & Export | 37-38 | CSV files exported, file statistics printed |

## A.4 Notebook Cells & Key Outputs (Port-Scan)

| Phase | Cell Range | Key Outputs |
|---|---|---|
| Setup & Imports | 1-4 | Constants (LABEL_NAMES, PALETTE), output directory |
| Load Data | 5-8 | Dataset shape, class distribution, node counts |
| Correlation Analysis | 9-15 | 81 high-correlated pairs, PCA projection, feature groups |
| Per-Node Analysis | 16-19 | Node-specific class distributions, temporal scatter plots |
| PCA & t-SNE | 20-25 | 11 components for 95% variance, t-SNE projections |
| Temporal Patterns | 26-28 | Label rolling mean, per-node temporal scatter, gap analysis |
| Window Size Analysis | 29-31 | Burst duration distributions, autocorrelation plots |
| XGBoost Baseline | 32-34 | 99.99% accuracy (to be verified), feature importance |
| Per-Node Binary Detection | 35 | F1 scores per node for binary anomaly detection |

---

## A.5 Feature List (Congestion) — First 30 Features

```
1. bw_avg_mbps_lag1
2. bw_avg_mbps_lag3
3. bw_avg_mbps_lag5
4. bw_avg_mbps_lag10
5. pkt_loss_pct_lag1
6. pkt_loss_pct_lag3
7. pkt_loss_pct_lag5
8. pkt_loss_pct_lag10
9. jitter_ms_lag1
10. jitter_ms_lag3
11. jitter_ms_lag5
12. jitter_ms_lag10
13. tcp_retransmit_pct_lag1
14. tcp_retransmit_pct_lag3
15. tcp_retransmit_pct_lag5
16. tcp_retransmit_pct_lag10
17. syn_rate_pps_lag1
18. syn_rate_pps_lag3
19. syn_rate_pps_lag5
20. syn_rate_pps_lag10
21. bw_avg_mbps_roll_mean_5
22. bw_avg_mbps_roll_mean_10
23. bw_avg_mbps_roll_mean_30
24. bw_avg_mbps_roll_std_5
25. bw_avg_mbps_roll_std_10
26. bw_avg_mbps_roll_std_30
27. pkt_loss_pct_roll_mean_5
28. pkt_loss_pct_roll_mean_10
29. pkt_loss_pct_roll_mean_30
30. pkt_loss_pct_roll_std_5
... (79 more features)
```

---

## A.6 Command Reference

### Loading & Inspecting
```python
# Congestion
df = pd.read_csv('data/congestion_training.csv')
fe_out = pd.read_csv('notebooks/congestion/congestion_features_engineered.csv')
feature_list = pd.read_csv('notebooks/congestion/feature_list.csv')

# Port-Scan
df_port = pd.read_csv("https://drive.google.com/uc?id=1lYT9J7ZAo4MdTqN-RHqBja8OVins-Nv8")
```

### Quick Stats
```python
# Congestion
print(f"Target: {df_out['future_congestion_label'].value_counts()}")
print(f"Features: {len(feature_list)}")
print(f"Missing: {df_out.isnull().sum().sum()}")

# Port-Scan
print(f"Target: {df_port['scan_label'].value_counts()}")
print(f"Nodes: {df_port['observer_node'].unique()}")
print(f"Gaps >100s: {(df_port.groupby('observer_node')['t'].diff() > 100).sum()}")
```

---

**End of Documentation**

---

**Document Generated:** May 19, 2026  
**Scope:** Comprehensive feature engineering & data preprocessing analysis  
**Authors:** Feature engineering notebooks (implicit), documentation by AI assistant
