# Port-Scan Deep Learning Pipeline & Inference

## Overview

This documentation covers **Steps 4-7** of the port-scan detection pipeline:
- **Step 4**: Build sequences for deep learning (12-step sliding windows)
- **Step 5**: Train BiLSTM model with self-attention
- **Step 6**: Per-node evaluation analysis
- **Step 7**: Inference wrapper for production deployment

## Step 4: Sequence Building

### Configuration
- **Window size**: 12 timesteps (120 seconds at 0.1s resolution)
- **Stride**: 2 (50% overlap between windows)
- **Per-node grouping**: Prevents temporal leakage across different network nodes
- **Temporal split**: 80% train, 20% validation (per node chronologically)

### Implementation
File: `notebooks/port-scanning/02_bilstm_model.ipynb` (Cell 1-5)

```python
def build_sequences(data, window_size=12, stride=2):
    """
    Per-node sliding window sequence builder.
    Groups by node to prevent cross-node contamination.
    """
    X_sequences = []
    y_sequences = []
    node_ids = []
    
    for node in sorted(data['node_id'].unique()):
        node_data = data[data['node_id'] == node].sort_values('t')
        X_node = node_data[feature_cols].values
        y_node = node_data['scan_label'].values
        
        # Sliding windows
        for i in range(0, len(X_node) - window_size + 1, stride):
            X_sequences.append(X_node[i:i + window_size])
            y_sequences.append(y_node[i + window_size - 1])  # Last timestep label
            node_ids.append(node)
    
    return np.array(X_sequences), np.array(y_sequences), np.array(node_ids)
```

### Output
- Train sequences: ~50K-100K samples × 12 × 110 features
- Val sequences: ~10K-20K samples × 12 × 110 features
- Balanced class distribution in sequence targets

## Step 5: BiLSTM + Self-Attention Model

### Architecture

```
Input (batch_size, 12, num_features)
    ↓
BiLSTM (2 layers, 128 hidden, bidirectional, dropout=0.3)
    ↓ (batch_size, 12, 256)
Self-Attention (8-head, scaled dot-product)
    ↓ (batch_size, 12, 256)
Take last timestep (batch_size, 256)
    ↓
Concatenate with node embedding (16-dim)
    ↓ (batch_size, 272)
Dense(128) → ReLU → Dropout(0.3)
    ↓ (batch_size, 128)
Dense(4) → Softmax
    ↓ (batch_size, 4)
```

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Loss function | Focal Loss (γ=2.0) | Handle class imbalance (62% class 2) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) | Stable convergence with L2 regularization |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10) | Periodic restarts to escape local minima |
| Batch size | 32 | Balance memory and gradient stability |
| Gradient clipping | 1.0 | Prevent exploding gradients |
| Early stopping | patience=10 on val macro F1 | Prevent overfitting |
| Epochs | 50 max | Sufficient for convergence |
| Node embeddings | 16-dim | Capture node-specific patterns |

### Key Features

1. **Bidirectional LSTM**: Captures temporal dependencies in both directions
2. **Self-Attention**: Focuses on important timesteps, provides interpretability
3. **Node Embeddings**: Distinguishes between network nodes (e.g., isp_private0 vs core networks)
4. **Focal Loss**: Downweights easy examples, focuses on hard negatives
5. **Per-node temporal split**: No data leakage across time

### File: `notebooks/port-scanning/02_bilstm_model.ipynb`
- **Cells**: ~25 code cells
- **Execution time**: ~30-60 minutes (GPU: 5-10 minutes)
- **Dependencies**: torch, pandas, numpy, sklearn, matplotlib
- **Outputs**:
  - `models/portscan_bilstm_best.pt` - Best model checkpoint
  - `models/portscan_bilstm_results.pkl` - Full results + metadata
  - `models/bilstm_training_curves.png` - Loss and F1 plots
  - `models/bilstm_confusion_matrix.png` - Validation confusion matrix

### Validation Metrics

Expected performance:
- **Validation Accuracy**: 85-92%
- **Macro F1**: 0.75-0.85
- **Per-class F1**: Varies by node

## Step 6: Per-Node Evaluation

### Analysis Strategy

Network nodes have different characteristics:
- **isp_private0**: Contains all 4 classes (0, 1, 2, 5) → evaluate with 4-class F1
- **Other nodes** (e.g., core networks): Often binary (0 vs 5) → evaluate with binary F1

### File: `notebooks/port-scanning/03_per_node_eval.ipynb`
- **Cells**: ~15 code cells
- **Execution time**: ~1-2 minutes
- **Inputs**: XGBoost and BiLSTM validation results
- **Outputs**:
  - `models/portscan_per_node_results.pkl` - Per-node metrics
  - `models/per_node_f1_comparison.csv` - Comparison table
  - `models/per_node_f1_comparison.png` - Bar plots

### Key Insights

1. **Model variance**: BiLSTM typically outperforms XGBoost on nodes with temporal patterns
2. **isp_private0 performance**: Usually lowest F1 (multiclass problem)
3. **Other nodes**: Higher F1 (binary classification simpler)
4. **Ensemble benefit**: Combining models often improves weak nodes

### Example Output

```
Per-Node F1 Comparison:
Node              XGBoost F1    BiLSTM F1    Samples    Avg F1
isp_private0      0.7245        0.7812       15000      0.7529
core_net1         0.8934        0.9123        8000      0.9029
core_net2         0.8845        0.9045        7500      0.8945
edge_node1        0.9123        0.8967        5000      0.9045
```

## Step 7: Inference Pipeline

### File: `models/portscan_inference.py`

Production-ready inference engine providing:

1. **Single-timestep prediction** (XGBoost-based)
   ```python
   result = engine.predict(features, node_id)
   # Returns: {label, confidence, probabilities}
   ```

2. **Sequence prediction** (BiLSTM-based)
   ```python
   result = engine.predict_sequence(sequence_12x110, node_id)
   # Returns: {label, confidence, model_agreement, model_predictions}
   ```

3. **Ensemble prediction** (soft voting)
   ```python
   result = engine.predict_sequence(sequence, node_id, use_ensemble=True)
   # Weights: 0.4 XGBoost + 0.6 BiLSTM
   ```

### API Reference

#### Initialization
```python
from models.portscan_inference import PortScanInferenceEngine

# Auto-detect GPU/CPU
engine = PortScanInferenceEngine(models_dir='models')
```

#### Single-timestep Prediction
```python
result = engine.predict(X, node_id='isp_private0')
# Result:
# {
#     'model': 'xgboost',
#     'label': 5,
#     'confidence': 0.92,
#     'probabilities': {0: 0.01, 1: 0.02, 2: 0.05, 5: 0.92}
# }
```

#### Sequence Prediction (12 timesteps)
```python
result = engine.predict_sequence(X_seq, node_id='core_net1')
# Result:
# {
#     'label': 0,
#     'confidence': 0.88,
#     'xgb_pred': 0,
#     'bilstm_pred': 0,
#     'model_agreement': True,
#     'model_predictions': {...}
# }
```

#### Batch Prediction
```python
results = engine.predict_batch(X_batch, node_ids)
# Efficient processing of multiple samples
```

#### Feature Importance
```python
importance = engine.get_feature_importance()
# Top features driving XGBoost predictions
top_10 = list(importance.items())[:10]
```

#### Adjust Ensemble Weights
```python
engine.set_ensemble_weights(xgb_weight=0.3, bilstm_weight=0.7)
# Customize model contribution
```

### Class Mapping

Internal representation (0-3) mapped to original classes:
- 0 → 0 (normal traffic)
- 1 → 1 (light scan)
- 2 → 2 (medium scan)
- 3 → 5 (heavy scan)

### Performance Characteristics

| Operation | CPU | GPU |
|-----------|-----|-----|
| Single prediction | ~5ms | ~10ms (first) + 1ms (subsequent) |
| Sequence prediction | ~50ms | ~20ms |
| Batch (100) sequences | ~5s | ~1s |

## Integration with Step 3

The inference pipeline uses outputs from Step 3 (baseline XGBoost):
- Scaler: Fitted on training data for normalization
- Feature columns: Guaranteed alignment with baseline
- Class mapping: Converts between internal (0-3) and original (0,1,2,5) labels

## Troubleshooting

### Issue: CUDA out of memory
```python
engine.device = torch.device('cpu')  # Fallback to CPU
```

### Issue: Node ID not recognized
- Verify node_id is in training data
- Falls back to node_idx=0 automatically

### Issue: Sequence shape error
- Ensure X_seq is (12, num_features)
- Check number of features matches training

### Issue: Low confidence predictions
- May indicate anomalous traffic pattern
- Consider ensemble disagreement (model_agreement=False)

## Next Steps

1. **Deploy to production**: Containerize engine with FastAPI/Flask
2. **Monitor performance**: Track per-node F1 over time
3. **Retrain pipeline**: Schedule monthly with new data
4. **Hyperparameter tuning**: Optimize BiLSTM and ensemble weights
5. **Real-time streaming**: Implement online sequence builder for continuous monitoring
