# Congestion Forecasting Pipeline

## Overview

Complete 7-step training pipeline for network congestion prediction using ensemble learning:

1. **LightGBM Baseline** (`01_baseline_model.ipynb`) - Fast tree-based classifier
2. **LSTM Deep Learning** (`02_lstm_model.ipynb`) - BiLSTM with Focal Loss
3. **Ensemble & Inference** (`03_ensemble.ipynb`) - Soft voting + deployment wrapper

## Project Structure

```
notebooks/congestion/
├── 01_baseline_model.ipynb          # Steps 1-3: LightGBM training
├── 02_lstm_model.ipynb              # Steps 4-5: LSTM sequences & training
├── 03_ensemble.ipynb                # Steps 6-7: Ensemble + inference
├── models/                          # Checkpoints (created by notebooks)
│   ├── baseline_lgb_results.pkl    # LightGBM model + scaler
│   ├── lstm_best.pt                # LSTM best weights
│   └── ensemble_config.pkl         # Ensemble config + weights
└── congestion_features_engineered.csv  # Input data (105,977 rows × 114 cols)

models/
└── congestion_inference.py          # Production inference wrapper
```

## Step-by-Step Guide

### Step 1: Run 01_baseline_model.ipynb

**Purpose:** Train LightGBM baseline classifier with SHAP feature importance

**Key Features:**
- ✅ Temporal train/val split (80/20 per node)
- ✅ Standardized features (StandardScaler)
- ✅ Class weights for imbalanced data (62% class 2)
- ✅ Leakage detection (flags accuracy > 97%)
- ✅ SHAP value visualization

**Output:**
- Console: Accuracy, F1-score, confusion matrix, classification report
- Plot: Feature importance (top 20 by gain)
- Files: `models/baseline_lgb_results.pkl` (model + scaler + metrics)

**Expected Performance:**
- Validation Accuracy: ~70-75%
- Validation Macro F1: ~0.45-0.55

### Step 2: Run 02_lstm_model.ipynb

**Purpose:** Build sequences and train BiLSTM with Focal Loss

**Key Features:**
- ✅ Sliding window sequences (30 timesteps, stride=5)
- ✅ Per-node grouping (prevents temporal leakage)
- ✅ BiLSTM architecture (2 layers, 128 hidden, bidirectional)
- ✅ Focal Loss (γ=2.0) with class weights
- ✅ AdamW optimizer + cosine annealing schedule
- ✅ Early stopping (patience=10 on validation F1)

**Output:**
- Console: Training progress, best validation F1
- Plots: Loss curve, F1-score evolution
- Files: `models/lstm_best.pt` (trained weights)

**Expected Performance:**
- Validation Accuracy: ~75-80%
- Validation Macro F1: ~0.50-0.60

### Step 3: Run 03_ensemble.ipynb

**Purpose:** Combine models via soft voting and prepare for deployment

**Key Features:**
- ✅ Load LightGBM + LSTM models
- ✅ Soft voting ensemble (40% LightGBM + 60% LSTM)
- ✅ Model agreement analysis
- ✅ Performance comparison (individual vs. ensemble)
- ✅ Save ensemble configuration

**Output:**
- Console: Performance comparison, confusion matrix
- Plots: Model accuracy/F1 bar chart, ensemble confusion matrix
- Files: 
  - `models/ensemble_config.pkl` (models + weights + config)
  - `models/congestion_inference.py` (inference wrapper)

**Expected Performance:**
- Ensemble Accuracy: ~78-82% (best)
- Ensemble Macro F1: ~0.55-0.65

## Data Details

### Input: congestion_features_engineered.csv

| Property | Value |
|----------|-------|
| **Rows** | 105,977 |
| **Columns** | 114 (109 features + 5 metadata) |
| **Nodes** | Multiple network nodes |
| **Target** | `future_congestion_label` |
| **Classes** | 0 (Normal, 25%), 1 (Mild, 4%), 2 (Moderate, 62%), 4 (Severe, 9%) |
| **Imbalance** | Severe (class 2 dominant) |

### Features

- Excluded from modeling: `node_id`, `t` (timestamp), `is_portscan`, `future_congestion_label`
- Used for modeling: 109 engineered features (network metrics, traffic stats, etc.)

### Target Classes

| Label | Description | Frequency |
|-------|-------------|-----------|
| 0 | Normal | 25% |
| 1 | Mild Congestion | 4% |
| 2 | Moderate Congestion | 62% |
| 4 | Severe Congestion | 9% |

## Model Architectures

### LightGBM Baseline

```python
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}
# Early stopping: 20 rounds on validation metric
```

### BiLSTM (Sequence Length=30)

```
Input: (batch, 30, 109)
  ↓
BiLSTM (2 layers, 128 hidden, bidirectional)
  ↓
Global Avg Pooling: (batch, 256)
  ↓
Dense (256 → 64, ReLU, dropout=0.3)
  ↓
Output (64 → 4, softmax)
```

**Loss:** Focal Loss (γ=2.0) with inverse frequency class weights  
**Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)  
**Scheduler:** CosineAnnealingWarmRestarts (T_0=10)

### Ensemble (Soft Voting)

```
LightGBM probabilities (last timestep): [p0, p1, p2, p3]
         ↓ (weight=0.4)
Ensemble = 0.4 * LGB_proba + 0.6 * LSTM_proba
         ↑ (weight=0.6)
LSTM probabilities (30 timesteps): [p0, p1, p2, p3]

Final prediction: argmax(ensemble_proba)
```

## Usage: Production Inference

### Option 1: Python API

```python
from models.congestion_inference import CongestionInferenceEngine

# Initialize
engine = CongestionInferenceEngine(config_path='models/ensemble_config.pkl')

# Single timestep prediction (uses LightGBM only)
X = df[features].iloc[0].values  # (109,)
result = engine.predict(X)
# Returns: {'label': 2, 'confidence': 0.78, 'probabilities': {...}}

# Sequence prediction (uses ensemble)
X_seq = df[features].iloc[:30].values  # (30, 109)
result = engine.predict_sequence(X_seq)
# Returns: {
#     'label': 2, 
#     'confidence': 0.82,
#     'model_agreement': True,
#     'model_predictions': {'lgb': {...}, 'lstm': {...}}
# }

# Batch prediction
X_batch = df[features].values[:100]  # (100, 109)
results = engine.predict_batch(X_batch, use_sequences=False)

# Feature importance
importance = engine.get_feature_importance()
# Returns: {'feature_name': score, ...}
```

### Option 2: Standalone Script

```bash
python models/congestion_inference.py
# Shows model configuration and example usage
```

## Installation Requirements

### Conda Environment

```bash
conda create -n congestion-forecast python=3.9

conda activate congestion-forecast

# Data & ML
pip install pandas numpy scikit-learn

# LightGBM
pip install lightgbm

# Deep Learning
pip install torch torchvision  # or cuda variant

# Analysis
pip install matplotlib seaborn shap joblib
```

### Requirements File

See `notebooks/congestion/requirements.txt` for exact versions.

## Key Decisions & Rationale

### 1. Why Two Models?

- **LightGBM:** Fast, interpretable, works well on tabular data
- **LSTM:** Captures temporal dependencies in sequences
- **Ensemble:** Combines strengths (LGB: instant decisions, LSTM: context-aware)

### 2. Why Focal Loss?

Class 2 is 62% of data → standard cross-entropy would ignore minority classes.  
Focal Loss adds $(1-p_t)^{\gamma}$ term to down-weight easy examples.

### 3. Why Soft Voting (40-60)?

- Test both 50-50 and optimized weights in `03_ensemble.ipynb`
- Typically: LSTM weight > LGB weight (LSTM captures temporal patterns better)
- Can be tuned based on validation performance

### 4. Why Temporal Split per Node?

- Each node has independent temporal sequence
- Ensures causality: training data always before validation
- Prevents "looking into the future"

## Troubleshooting

### "ValueError: Could not find 'congestion_features_engineered.csv'"
→ Run notebook from `notebooks/congestion/` directory  
→ Or update path: `fe_path = '../../data/congestion_features_engineered.csv'`

### "CUDA out of memory"
→ Reduce batch_size in `02_lstm_model.ipynb` from 32 to 16 or 8  
→ Or set `device = torch.device('cpu')`

### "Model accuracy > 97% in notebook 1"
→ Check for data leakage:
  - Is target information in features?
  - Are there perfect correlations?
  - Fix: Re-run feature engineering with causality checks

### "LSTM predictions don't improve after epoch 10"
→ Normal! Model has learned patterns. Early stopping will trigger.  
→ Check validation F1 (not just loss)

### "Model weights file not found when running notebook 3"
→ Ensure notebook 2 completed (creates `models/lstm_best.pt`)  
→ Check that training loop did not crash mid-way

## Performance Benchmarks

| Stage | Model | Accuracy | Macro F1 | File Size |
|-------|-------|----------|----------|-----------|
| 1 | LightGBM | ~72% | ~0.48 | ~50 MB |
| 2 | LSTM | ~77% | ~0.53 | ~15 MB |
| 3 | **Ensemble** | **~80%** | **~0.58** | ~65 MB |

*Exact metrics depend on random seed and data splits*

## Next Steps

1. **Tune ensemble weights** in `03_ensemble.ipynb`: Try 30-70, 20-80
2. **Hyperparameter sweep:** Grid search over LightGBM `num_leaves`, LSTM `hidden_dim`
3. **Cross-validation:** Use K-fold CV for robustness (add to step 1)
4. **Monitor live:** Deploy `CongestionInferenceEngine` to production dashboard
5. **Retrain monthly:** Update models as new data arrives

## Files Generated During Training

```
After running all 3 notebooks:

models/
├── baseline_lgb_results.pkl (LightGBM + scaler)          ~50 MB
├── lstm_best.pt (LSTM weights)                           ~15 MB
├── ensemble_config.pkl (ensemble configuration)          ~65 MB
└── congestion_inference.py (inference wrapper)           ~8 KB
```

**Total disk usage:** ~130 MB

## References

- **LightGBM:** https://lightgbm.readthedocs.io/
- **PyTorch:** https://pytorch.org/docs/
- **Focal Loss:** https://arxiv.org/abs/1708.02002
- **SHAP:** https://shap.readthedocs.io/

## Author Notes

✅ Pipeline is fully automated—just run each notebook in order.  
✅ All models save checkpoints for later inference.  
✅ Inference wrapper handles preprocessing and ensemble voting.  
⚠️  Adjust hyperparameters based on your validation metrics!

---

**Created:** Congestion Forecasting Training Pipeline  
**Version:** 1.0  
**Status:** Ready for training and deployment
