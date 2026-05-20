"""
Port-Scan Inference Pipeline
Step 7 - Save model + inference wrapper

This module provides production-ready inference for port-scan detection.
Supports both single-timestep (XGBoost) and sequence-based (BiLSTM + ensemble) predictions.

Usage:
    engine = PortScanInferenceEngine('path/to/config')
    
    # Single timestep
    label, confidence = engine.predict(features, node_id)
    
    # 12-step sequence
    label, confidence, agreement = engine.predict_sequence(sequence, node_id)
"""

import numpy as np
import torch
import torch.nn as nn
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union


class MultiHeadAttention(nn.Module):
    """Self-attention mechanism for BiLSTM."""
    
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        Q = self.Q(Q)
        K = self.K(K)
        V = self.V(V)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.fc_out(context)
        
        return output, attn_weights


class BiLSTMAttentionModel(nn.Module):
    """BiLSTM with self-attention for port-scan detection."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_heads=8, 
                 num_classes=4, num_nodes=10, node_embed_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        self.node_embed = nn.Embedding(num_nodes, node_embed_dim)
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                              bidirectional=True, dropout=0.3, batch_first=True)
        self.attention = MultiHeadAttention(hidden_dim * 2, num_heads=num_heads)
        
        lstm_output_dim = hidden_dim * 2
        self.fc1 = nn.Linear(lstm_output_dim + node_embed_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x, node_ids):
        lstm_out, _ = self.bilstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        last_output = attn_out[:, -1, :]
        node_emb = self.node_embed(node_ids)
        combined = torch.cat([last_output, node_emb], dim=1)
        
        x = self.fc1(combined)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits, attn_weights


class PortScanInferenceEngine:
    """
    Unified inference engine for port-scan detection.
    
    Provides:
    - Single-timestep predictions (XGBoost)
    - Sequence-based predictions (BiLSTM)
    - Ensemble predictions (weighted combination)
    - Per-node confidence scores
    """
    
    def __init__(self, config_path: Optional[str] = None, models_dir: str = 'models'):
        """
        Initialize inference engine.
        
        Args:
            config_path: Path to config JSON (optional)
            models_dir: Directory containing model files
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = Path(models_dir)
        
        # Load models
        self._load_models()
        
        # Model weights for ensemble (can be tuned)
        self.xgb_weight = 0.4
        self.bilstm_weight = 0.6
    
    def _load_models(self):
        """Load XGBoost and BiLSTM models."""
        # Load XGBoost results
        xgb_path = self.models_dir / 'portscan_xgb_results.pkl'
        xgb_results = joblib.load(xgb_path)
        self.xgb_model = xgb_results['model']
        self.xgb_class_mapping = xgb_results['class_mapping']
        
        # Load BiLSTM results
        bilstm_path = self.models_dir / 'portscan_bilstm_results.pkl'
        bilstm_results = joblib.load(bilstm_path)
        self.scaler = bilstm_results['scaler']
        self.feature_cols = bilstm_results['feature_cols']
        self.bilstm_class_mapping = bilstm_results['class_mapping']
        
        # Reconstruct BiLSTM model architecture
        num_features = len(self.feature_cols)
        train_dataset = bilstm_results['train_dataset']
        
        self.bilstm_model = BiLSTMAttentionModel(
            input_dim=num_features,
            hidden_dim=128,
            num_layers=2,
            num_heads=8,
            num_classes=4,
            num_nodes=train_dataset.num_nodes,
            node_embed_dim=16
        ).to(self.device)
        
        self.bilstm_model.load_state_dict(bilstm_results['model_state'])
        self.bilstm_model.eval()
        
        # Node to index mapping
        self.node_to_idx = {node: idx for idx, node in enumerate(
            sorted(np.unique(bilstm_results['train_node_ids']))
        )}
        
        print(f'Models loaded on {self.device}')
        print(f'XGBoost class mapping: {self.xgb_class_mapping}')
        print(f'BiLSTM class mapping: {self.bilstm_class_mapping}')
        print(f'Feature columns: {len(self.feature_cols)}')
        print(f'Node list: {list(self.node_to_idx.keys())}')
    
    def predict(self, X: np.ndarray, node_id: str) -> Dict[str, Union[int, float]]:
        """
        Single-timestep prediction using XGBoost.
        
        Args:
            X: Feature vector (1D array of shape (num_features,))
            node_id: Node identifier
        
        Returns:
            {
                'label': predicted class (0, 1, 2, or 5),
                'confidence': prediction confidence [0, 1],
                'probabilities': class probabilities
            }
        """
        if X.ndim != 1 or len(X) != len(self.feature_cols):
            raise ValueError(f'Expected shape ({len(self.feature_cols)},), got {X.shape}')
        
        # Predict with XGBoost
        proba = self.xgb_model.predict_proba(X.reshape(1, -1))[0]
        pred_class_idx = np.argmax(proba)
        pred_label = self.xgb_class_mapping[pred_class_idx]
        confidence = proba[pred_class_idx]
        
        return {
            'model': 'xgboost',
            'label': int(pred_label),
            'confidence': float(confidence),
            'probabilities': {
                0: float(proba[0]),
                1: float(proba[1]),
                2: float(proba[2]),
                5: float(proba[3])
            }
        }
    
    def predict_sequence(self, X_seq: np.ndarray, node_id: str, 
                        use_ensemble: bool = True) -> Dict[str, Union[int, float, bool]]:
        """
        Sequence-based prediction (12-step window).
        
        Args:
            X_seq: Sequence of shape (12, num_features)
            node_id: Node identifier
            use_ensemble: If True, combine XGBoost and BiLSTM via soft voting
        
        Returns:
            {
                'label': predicted class,
                'confidence': ensemble confidence,
                'xgb_pred': XGBoost prediction,
                'bilstm_pred': BiLSTM prediction,
                'model_agreement': whether both models agree,
                'model_predictions': {xgboost: ..., bilstm: ...}
            }
        """
        if X_seq.ndim != 2 or X_seq.shape[1] != len(self.feature_cols):
            raise ValueError(f'Expected shape (12, {len(self.feature_cols)}), got {X_seq.shape}')
        
        if X_seq.shape[0] != 12:
            raise ValueError(f'Expected 12 timesteps, got {X_seq.shape[0]}')
        
        # XGBoost: use last timestep
        xgb_pred = self.predict(X_seq[-1], node_id)
        
        # BiLSTM: use full sequence
        X_seq_scaled = self.scaler.transform(X_seq)
        X_tensor = torch.FloatTensor(X_seq_scaled).unsqueeze(0).to(self.device)
        
        node_idx = self.node_to_idx.get(node_id, 0)
        node_tensor = torch.LongTensor([node_idx]).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.bilstm_model(X_tensor, node_tensor)
            bilstm_proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        bilstm_pred_idx = np.argmax(bilstm_proba)
        bilstm_pred_label = self.bilstm_class_mapping[bilstm_pred_idx]
        bilstm_confidence = bilstm_proba[bilstm_pred_idx]
        
        bilstm_pred = {
            'model': 'bilstm',
            'label': int(bilstm_pred_label),
            'confidence': float(bilstm_confidence),
            'probabilities': {
                0: float(bilstm_proba[0]),
                1: float(bilstm_proba[1]),
                2: float(bilstm_proba[2]),
                5: float(bilstm_proba[3])
            }
        }
        
        if use_ensemble:
            # Soft voting: weighted probability averaging
            xgb_proba_dict = xgb_pred['probabilities']
            bilstm_proba_dict = bilstm_pred['probabilities']
            
            ensemble_proba = {
                0: self.xgb_weight * xgb_proba_dict[0] + self.bilstm_weight * bilstm_proba_dict[0],
                1: self.xgb_weight * xgb_proba_dict[1] + self.bilstm_weight * bilstm_proba_dict[1],
                2: self.xgb_weight * xgb_proba_dict[2] + self.bilstm_weight * bilstm_proba_dict[2],
                5: self.xgb_weight * xgb_proba_dict[5] + self.bilstm_weight * bilstm_proba_dict[5]
            }
            
            ensemble_label = max(ensemble_proba, key=ensemble_proba.get)
            ensemble_confidence = ensemble_proba[ensemble_label]
        else:
            ensemble_label = bilstm_pred['label']
            ensemble_confidence = bilstm_pred['confidence']
            ensemble_proba = bilstm_proba_dict
        
        model_agreement = xgb_pred['label'] == bilstm_pred['label']
        
        return {
            'label': int(ensemble_label),
            'confidence': float(ensemble_confidence),
            'probabilities': ensemble_proba,
            'xgb_pred': xgb_pred['label'],
            'bilstm_pred': bilstm_pred['label'],
            'model_agreement': bool(model_agreement),
            'model_predictions': {
                'xgboost': xgb_pred,
                'bilstm': bilstm_pred
            }
        }
    
    def predict_batch(self, X_batch: np.ndarray, node_ids: np.ndarray, 
                     use_sequences: bool = False) -> list:
        """
        Batch prediction.
        
        Args:
            X_batch: Batch of features or sequences
            node_ids: Batch of node identifiers
            use_sequences: If True, X_batch shape is (batch_size, 12, num_features)
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i, node_id in enumerate(node_ids):
            if use_sequences:
                pred = self.predict_sequence(X_batch[i], node_id)
            else:
                pred = self.predict(X_batch[i], node_id)
            results.append(pred)
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get XGBoost feature importance.
        
        Returns:
            Dictionary of feature names to importance scores
        """
        importance = self.xgb_model.feature_importances_
        feature_importance = {
            self.feature_cols[i]: float(importance[i])
            for i in range(len(self.feature_cols))
        }
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def set_ensemble_weights(self, xgb_weight: float, bilstm_weight: float):
        """Adjust ensemble weights."""
        total = xgb_weight + bilstm_weight
        self.xgb_weight = xgb_weight / total
        self.bilstm_weight = bilstm_weight / total
        print(f'Ensemble weights updated: XGBoost={self.xgb_weight:.2f}, BiLSTM={self.bilstm_weight:.2f}')


# Example usage and testing
if __name__ == '__main__':
    # Initialize engine
    engine = PortScanInferenceEngine(models_dir='models')
    
    print('\n' + '='*80)
    print('SINGLE-TIMESTEP PREDICTION EXAMPLE')
    print('='*80)
    
    # Create dummy feature vector for testing
    dummy_features = np.random.randn(len(engine.feature_cols))
    result = engine.predict(dummy_features, 'isp_private0')
    print(f'\nPrediction result: {result}')
    
    print('\n' + '='*80)
    print('SEQUENCE PREDICTION EXAMPLE')
    print('='*80)
    
    # Create dummy sequence for testing
    dummy_sequence = np.random.randn(12, len(engine.feature_cols))
    result_seq = engine.predict_sequence(dummy_sequence, 'isp_private0', use_ensemble=True)
    print(f'\nSequence prediction result:')
    for key, value in result_seq.items():
        if key != 'model_predictions':
            print(f'  {key}: {value}')
    
    print('\n' + '='*80)
    print('FEATURE IMPORTANCE (Top 10)')
    print('='*80)
    
    importance = engine.get_feature_importance()
    for feat, imp in list(importance.items())[:10]:
        print(f'  {feat}: {imp:.6f}')
