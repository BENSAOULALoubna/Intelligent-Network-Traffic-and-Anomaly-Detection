# NetSim — Network Anomaly Detection Dashboard
### OMNeT++ simulation + real-time Dash dashboard + ML inference

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Install opp_env & OMNeT++](#2-install-opp_env--omnet)
3. [Create the Python Virtual Environment](#3-create-the-python-virtual-environment)
4. [Project Structure](#4-project-structure)
5. [Place the netsim Folder into the Workspace](#5-place-the-netsim-folder-into-the-workspace)
6. [Build the Simulation Binary](#6-build-the-simulation-binary)
7. [Configuration Reference](#7-configuration-reference)
8. [Running Simulations](#8-running-simulations)
9. [Running the Dashboard](#9-running-the-dashboard)
10. [ML Model Integration](#10-ml-model-integration)
11. [Data Flow Overview](#11-data-flow-overview)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Ubuntu | 22.04 / 24.04 LTS | WSL2 on Windows works |
| Python | 3.10 + | System or pyenv |
| pip | 23 + | `python3 -m pip install --upgrade pip` |
| git | any | for cloning |
| build-essential | any | `sudo apt install build-essential` |

```bash
sudo apt update && sudo apt install -y \
    build-essential git curl wget \
    python3 python3-pip python3-venv \
    libxml2-dev zlib1g-dev
```

---

## 2. Install opp_env & OMNeT++

`opp_env` is the official OMNeT++ environment manager. It downloads, builds, and activates a fully isolated OMNeT++ installation.

```bash
# Install opp_env
pip3 install --user opp_env

# Verify
opp_env --version
```

### 2.1 Install OMNeT++ 6.x (with INET)

```bash
# List available versions
opp_env list

# Install OMNeT++ 6.0.3 + INET 4.5.2 (adjust versions as needed)
opp_env install omnetpp-6.0.3 inet-4.5.2

# This downloads, patches, and builds everything — takes 10-20 min
```

### 2.2 Activate the environment

```bash
# Activate — run this every new shell session before anything else
opp_env shell omnetpp-6.0.3 inet-4.5.2
```

You should see the prompt change and `which opp_run` should return a valid path.

```bash
# Confirm
opp_run --version          # e.g. OMNeT++ 6.0.3
opp_makemake --version
```

### 2.3 Locate the default workspace

```bash
echo $OMNETPP_ROOT         # e.g. /home/<user>/.opp_env/omnetpp-6.0.3
ls /home/opp_env/default_workspace/
```

All projects go inside `default_workspace/`.

---

## 3. Create the Python Virtual Environment

Do this **inside** the activated opp_env shell.

```bash
cd /home/opp_env/default_workspace

# Create venv
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install dashboard dependencies
pip install dash plotly scikit-learn joblib numpy pandas

# For PyTorch models (optional)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Add activation to your workflow script so you don't forget:

```bash
# one-liner to activate both every session
opp_env shell omnetpp-6.0.3 inet-4.5.2 -- bash -c \
    "source /home/opp_env/default_workspace/.venv/bin/activate && exec bash"
```

---

## 4. Project Structure

```
/home/opp_env/default_workspace/
└── netsim/
    ├── netsim                  ← compiled simulation binary (after make)
    ├── Makefile
    ├── package.ned
    ├── configs/
    │   ├── omnetpp.ini         ← master config (all training + realtime configs)
    │   └── base.ini            ← shared parameters included by omnetpp.ini
    ├── src/
    │   ├── DataExporter.cc     ← sends JSON records over TCP + writes JSONL
    │   ├── DataExporter.h
    │   ├── DataExporter.ned
    │   └── ...                 ← other modules
    ├── xml/
    │   ├── scenario_normal.xml
    │   ├── scenario_congestion.xml
    │   ├── scenario_portscan.xml
    │   └── scenario_mixed.xml
    ├── results/
    │   ├── congestion_stream.jsonl   ← live JSONL written by DataExporter
    │   ├── portscan_stream.jsonl
    │   └── *.sca / *.vec             ← scalar/vector results
    ├── models/
    │   ├── congestion_clf.pkl        ← trained sklearn model
    │   ├── portscan_clf.pkl
    │   └── scaler.pkl                ← StandardScaler fitted on training data
    └── dashboard/
        └── dashboard.py              ← Dash app
```

---

## 5. Place the netsim Folder into the Workspace

If you are copying from an existing location:

```bash
# Option A — copy
cp -r /path/to/your/netsim /home/opp_env/default_workspace/netsim

# Option B — clone from git
cd /home/opp_env/default_workspace
git clone https://github.com/youruser/netsim.git
```

Ensure directory ownership is correct:

```bash
chown -R $(whoami):$(whoami) /home/opp_env/default_workspace/netsim
```

---

## 6. Build the Simulation Binary

Always run inside the activated opp_env shell.

```bash
cd /home/opp_env/default_workspace/netsim

# Generate Makefile (only needed once or after .ned changes)
opp_makemake -f --deep -o netsim -O out \
    -I$(INET_ROOT)/src \
    -L$(INET_ROOT)/src -lINET

# Build (use -j$(nproc) for parallel)
make -j$(nproc) MODE=release

# Verify binary exists
ls -lh netsim
```

After a source change, just run `make -j$(nproc) MODE=release` again — no need to re-run `opp_makemake`.

---

## 7. Configuration Reference

All configs live in `configs/omnetpp.ini`.

### Training configs (batch, finite duration)

| Config name | Description | Duration | ~Records |
|---|---|---|---|
| `Training_Normal` | Baseline normal traffic, label=0 only | 7200 s | ~7200 |
| `Training_Congestion` | 6 cycles × 5 congestion types, labels 1–5 | 7200 s | ~7200 |
| `Training_PortScan` | 6 cycles × 5 scan types, labels 1–5 | 7200 s | ~7200 |
| `Training_Mixed` | Congestion + scan simultaneously | 7200 s | ~7200 |

### Realtime configs (dashboard, run indefinitely)

| Config name | Hosts per LAN | Total hosts | Use |
|---|---|---|---|
| `Realtime_Hosts5` | 5 | 80 | Live monitor page |
| `Realtime_Hosts10` | 10 | 160 | Scalability comparison |
| `Realtime_Hosts20` | 20 | 320 | Scalability comparison |
| `Realtime_Hosts40` | 40 | 640 | Scalability comparison |
| `Realtime_Hosts80` | 80 | 1280 | Scalability comparison |

### Label definitions

**Congestion labels:**

| Label | Meaning |
|---|---|
| 0 | Normal |
| 1 | Queue congestion |
| 2 | Incast |
| 3 | Burst congestion |
| 4 | Periodic congestion |
| 5 | Link failure |

**Port scan labels:**

| Label | Meaning |
|---|---|
| 0 | Normal |
| 1–5 | Scan type (SYN, UDP, XMAS, NULL, FIN) |

---

## 8. Running Simulations

Always `cd` into the project directory first. Activate both the opp_env shell and the Python venv before running.

### 8.1 Training runs

```bash
cd /home/opp_env/default_workspace/netsim

# Prepare clean result files
truncate -s 0 results/congestion_stream.jsonl
truncate -s 0 results/portscan_stream.jsonl

# Normal baseline
./netsim -f configs/omnetpp.ini -c Training_Normal -u Cmdenv

# Congestion events
./netsim -f configs/omnetpp.ini -c Training_Congestion -u Cmdenv

# Port scan events
./netsim -f configs/omnetpp.ini -c Training_PortScan -u Cmdenv

# Mixed (both simultaneously)
./netsim -f configs/omnetpp.ini -c Training_Mixed -u Cmdenv
```

Run all training configs back-to-back (append, don't truncate between runs):

```bash
for cfg in Training_Normal Training_Congestion Training_PortScan Training_Mixed; do
    echo "=== Running $cfg ==="
    ./netsim -f configs/omnetpp.ini -c $cfg -u Cmdenv
done
```

### 8.2 Realtime / dashboard runs

Open **two terminals** (both with opp_env shell + venv activated):

**Terminal 1 — start the dashboard first:**
```bash
cd /home/opp_env/default_workspace/netsim
source ../.venv/bin/activate
python dashboard/dashboard.py
# Dashboard is now listening on TCP :5001 and http://localhost:8051
```

**Terminal 2 — start the simulation:**
```bash
cd /home/opp_env/default_workspace/netsim

# Live monitor (5 hosts/LAN = 80 total)
./netsim -f configs/omnetpp.ini -c Realtime_Hosts5 -u Cmdenv

# Or any other realtime config
./netsim -f configs/omnetpp.ini -c Realtime_Hosts10 -u Cmdenv
./netsim -f configs/omnetpp.ini -c Realtime_Hosts20 -u Cmdenv
./netsim -f configs/omnetpp.ini -c Realtime_Hosts40 -u Cmdenv
```

The simulation's DataExporter connects to `localhost:5001` and streams JSON records. The dashboard auto-updates every 3 seconds.

### 8.3 Stop a simulation

```bash
# In the simulation terminal
Ctrl+C

# Or kill by name
pkill -f netsim
```

---

## 9. Running the Dashboard

```bash
cd /home/opp_env/default_workspace/netsim
source ../.venv/bin/activate
python dashboard/dashboard.py
```

Open browser: **http://localhost:8051**

### Pages

**Page 1 — Live Monitor (80 hosts / 5 per LAN)**
- Mean throughput per router with max-bandwidth dashed reference line
- OSPF hello packets over time
- Overhead ratio (OSPF / total bytes)
- Packet loss %
- Router reaction time (queue spike → throughput drop lag)
- ML model prediction panel
- Alert banner for detected anomalies

**Page 2 — Scalability Comparison**
- All host configurations overlaid on same graph
- Mean throughput per host
- OSPF hello packets across configs

### Dashboard auto-starts

The dashboard auto-launches `Realtime_Hosts5` on startup. You can stop/start configs from the sidebar without restarting the dashboard.

---

## 10. ML Model Integration

### 10.1 What you need

| File | Description |
|---|---|
| `models/congestion_clf.pkl` | Trained classifier (sklearn, XGBoost, etc.) |
| `models/portscan_clf.pkl` | Trained port-scan classifier |
| `models/scaler.pkl` | `StandardScaler` or `MinMaxScaler` fitted on training features |
| Feature list | Exact column names and order used during `model.fit()` |

### 10.2 Feature extraction

Your DataExporter JSON records must contain the same features used during training. A typical record looks like:

```json
{
  "t": 120.0,
  "scenario": "realtime_h5",
  "node_id": "boundary",
  "bw_out_mbps": 432.1,
  "pkt_loss_pct": 0.02,
  "queue_bytes": 0,
  "ospf_pkts": 12,
  "active_flows": 87,
  "congestion_label": 0,
  "scan_label": 0
}
```

The feature vector fed to the model is extracted from this record. Example:

```python
FEATURE_COLS = [
    "bw_out_mbps", "pkt_loss_pct", "queue_bytes",
    "ospf_pkts", "active_flows"
]
```

**These must exactly match the columns used in training.**

### 10.3 Loading and running inference

```python
import joblib, numpy as np

scaler   = joblib.load("models/scaler.pkl")
clf_cong = joblib.load("models/congestion_clf.pkl")
clf_scan = joblib.load("models/portscan_clf.pkl")

FEATURE_COLS = ["bw_out_mbps", "pkt_loss_pct", "queue_bytes",
                "ospf_pkts", "active_flows"]

def predict(record: dict) -> dict:
    x = np.array([[record.get(f, 0) for f in FEATURE_COLS]])
    x_scaled = scaler.transform(x)
    cong_pred  = int(clf_cong.predict(x_scaled)[0])
    scan_pred  = int(clf_scan.predict(x_scaled)[0])
    cong_proba = clf_cong.predict_proba(x_scaled).max() if hasattr(clf_cong, "predict_proba") else None
    return {
        "congestion_pred": cong_pred,
        "scan_pred":       scan_pred,
        "confidence":      round(float(cong_proba), 3) if cong_proba else None,
    }
```

The dashboard calls `predict(record)` on each incoming TCP record and displays the result in the ML Prediction panel alongside the ground-truth label from the simulator.

### 10.4 Training pipeline (quick reference)

```bash
# 1. Run all training configs (see Section 8.1)
# 2. Merge JSONL output
cat results/congestion_stream.jsonl results/portscan_stream.jsonl > results/all_training.jsonl

# 3. Train (example script)
python train.py \
    --data  results/all_training.jsonl \
    --out   models/ \
    --target congestion_label   # or scan_label
```

Minimum `train.py` outline:

```python
import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURES = ["bw_out_mbps","pkt_loss_pct","queue_bytes","ospf_pkts","active_flows"]
TARGET   = "congestion_label"

df  = pd.read_json("results/all_training.jsonl", lines=True)
X   = df[FEATURES].fillna(0)
y   = df[TARGET]

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, stratify=y)
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
clf.fit(X_tr, y_tr)
print(f"Test accuracy: {clf.score(X_te, y_te):.4f}")

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(clf,    "models/congestion_clf.pkl")
```

---

## 11. Data Flow Overview

```
OMNeT++ netsim binary
        │
        │  JSON records over TCP :5001  (primary)
        │  JSONL append to results/*.jsonl  (fallback)
        ▼
  dashboard.py  (DataStore — single shared deque)
        │
        ├── Live Monitor page
        │     ├── Mean throughput + bandwidth limit line
        │     ├── OSPF hello packets
        │     ├── Overhead ratio
        │     ├── Packet loss %
        │     ├── Router reaction time
        │     └── ML Prediction panel  ◄── models/*.pkl
        │
        └── Scalability page
              ├── All host configs overlaid
              ├── Mean throughput per host
              └── OSPF packets across configs
```

---

## 12. Troubleshooting

### Binary not found
```bash
ls /home/opp_env/default_workspace/netsim/netsim
# If missing → re-run Section 6 (make)
```

### TCP not connecting / dashboard shows "Waiting"
```bash
# Check port is open
ss -tlnp | grep 5001

# Check DataExporter connects to correct host/port
# In DataExporter.cc look for: connectAddress and connectPort parameters
# In base.ini ensure:
# *.dataExporter.connectAddress = "127.0.0.1"
# *.dataExporter.connectPort    = 5001
```

### All metrics are zero
The field names in the dashboard must match exactly what DataExporter writes.
Check `DataExporter.cc` for the JSON key names and compare against these fields used in the dashboard:
`bw_out_mbps`, `pkt_loss_pct`, `queue_bytes`, `ospf_pkts`, `active_flows`, `congestion_label`, `scan_label`, `node_id`, `scenario`, `t`

### opp_env shell drops on new terminal
```bash
# Re-activate every new terminal
opp_env shell omnetpp-6.0.3 inet-4.5.2
source /home/opp_env/default_workspace/.venv/bin/activate
```

### make fails — INET not found
```bash
echo $INET_ROOT    # should be non-empty
# If empty, re-run: opp_env shell omnetpp-6.0.3 inet-4.5.2
```

### Dashboard import errors
```bash
pip install dash plotly joblib scikit-learn numpy pandas
```

---

*Generated for the NetSim project — update Section 10.2 feature list to match your actual DataExporter JSON keys.*