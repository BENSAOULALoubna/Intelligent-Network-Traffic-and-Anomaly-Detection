"""
Congestion Forecasting Inference Engine

Production-ready wrapper for ensemble predictions (LightGBM + LSTM)
Handles preprocessing, model loading, and soft voting.

Usage:
    from congestion_inference import CongestionInferenceEngine
    
    engine = CongestionInferenceEngine()
    
    # Single timestep prediction
    result = engine.predict(X)  # (num_features,) array
    print(result['label'], result['confidence'])
    
    # Sequence prediction (30 timesteps)
    result = engine.predict_sequence(X_seq)  # (30, num_features) array
    print(result['label'], result['model_agreement'])
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import warnings
warnings.filterwarnings('ignore')


class BiLSTMClassifier(nn.Module):
    """BiLSTM model architecture (must match training)."""
    
    def __init__(self, input_size, hidden_dim=128, num_layers=2, num_classes=4, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_dim, num_layers=num_layers,
            bidirectional=True, dropout=dropout, batch_first=True
        )
        
        lstm_output_size = hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """Forward pass with global average pooling."""
        lstm_out, _ = self.lstm(x)
        avg_pool = torch.mean(lstm_out, dim=1)
        logits = self.fc(avg_pool)
        return logits


class CongestionInferenceEngine:
    """
    Ensemble inference engine for congestion forecasting.
    
    Combines LightGBM (baseline) and LSTM (deep learning) via soft voting.
    Supports both single-timestep and sequence-level predictions.
    
    Attributes:
        model_lgb: Trained LightGBM classifier
        model_lstm: Trained BiLSTM classifier
        scaler: StandardScaler (fitted on training data)
        feature_cols: List of feature column names
        idx_map_to_class: Mapping from model output (0-3) to original classes (0,1,2,4)
        ensemble_weights: Dictionary of LightGBM and LSTM weights for soft voting
        window_size: Sequence window size for LSTM (default: 30)
    """
    
    def __init__(self, config_path='models/ensemble_config.pkl', device=None):
        """
        Initialize inference engine.
        
        Args:
            config_path (str): Path to ensemble configuration pickle file
            device (torch.device): Device for LSTM (default: auto-detect GPU)
        """
        self.config_path = config_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        try:
            config = joblib.load(config_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config not found at {config_path}. Train model first.")
        
        # Load LightGBM
        self.model_lgb = config['model_lgb']
        
        # Load LSTM
        self.model_lstm = BiLSTMClassifier(
            input_size=len(config['feature_cols']),
            hidden_dim=128,
            num_layers=2,
            num_classes=4,
            dropout=0.3
        ).to(self.device)
        self.model_lstm.load_state_dict(config['model_lstm'])
        self.model_lstm.eval()
        
        # Load preprocessing components
        self.scaler = config['scaler']
        self.feature_cols = config['feature_cols']
        self.idx_map_to_class = config['idx_map_to_class']
        
        # Load ensemble configuration
        self.ensemble_weights = config['ensemble_weights']
        self.window_size = config.get('window_size', 30)
        self.stride = config.get('stride', 5)
        
        # Metrics for reference
        self.metrics = config.get('metrics', {})
    
    def predict(self, X, return_raw_probas=False):
        """
        Single timestep prediction using LightGBM.
        
        For single timesteps, LSTM cannot be applied (requires sequences).
        Falls back to LightGBM baseline.
        
        Args:
            X (np.ndarray or pd.Series): Feature vector of shape (num_features,)
            return_raw_probas (bool): Return raw probability distribution
        
        Returns:
            dict: {
                'label': predicted class (0, 1, 2, 4),
                'confidence': max probability [0-1],
                'probabilities': dict of class -> probability,
                'method': 'lightgbm_baseline'
            }
        
        Raises:
            ValueError: If input shape is incorrect
        """
        # Input validation
        X = np.asarray(X).flatten()
        if len(X) != len(self.feature_cols):
            raise ValueError(
                f"Expected {len(self.feature_cols)} features, got {len(X)}. "
                f"Features: {self.feature_cols}"
            )
        
        # Standardize
        X_scaled = self.scaler.transform(X.reshape(1, -1))[0]
        
        # LightGBM prediction
        lgb_proba = self.model_lgb.predict(X_scaled.reshape(1, -1))[0]
        
        # Get prediction
        pred_idx = np.argmax(lgb_proba)
        pred_label = self.idx_map_to_class[pred_idx]
        confidence = lgb_proba[pred_idx]
        
        # Build result
        result = {
            'label': int(pred_label),
            'confidence': float(confidence),
            'method': 'lightgbm_baseline'
        }
        
        if return_raw_probas:
            result['raw_probabilities'] = lgb_proba
        
        result['probabilities'] = {
            self.idx_map_to_class[i]: float(p) 
            for i, p in enumerate(lgb_proba)
        }
        
        return result
    
    def predict_sequence(self, X_seq, return_model_proba=False):
        """
        Sequence prediction using ensemble (LightGBM + LSTM soft voting).
        
        Args:
            X_seq (np.ndarray): Sequence of shape (window_size, num_features)
                               Must have window_size >= 2
            return_model_proba (bool): Return individual model probabilities
        
        Returns:
            dict: {
                'label': predicted class (0, 1, 2, 4),
                'confidence': max ensemble probability [0-1],
                'probabilities': dict of class -> probability,
                'model_predictions': {
                    'lgb': {'label': int, 'confidence': float},
                    'lstm': {'label': int, 'confidence': float}
                },
                'model_agreement': bool (all models agree),
                'method': 'ensemble_soft_voting'
            }
        
        Raises:
            ValueError: If input shape is incorrect
        """
        # Input validation
        X_seq = np.asarray(X_seq)
        if X_seq.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_seq.shape}")
        if X_seq.shape[0] < 2:
            raise ValueError(f"Sequence too short: {X_seq.shape[0]} < 2 (min required)")
        if X_seq.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"Expected {len(self.feature_cols)} features per timestep, "
                f"got {X_seq.shape[1]}"
            )
        
        # Standardize sequence
        X_seq_scaled = np.array([
            self.scaler.transform(x.reshape(1, -1))[0] 
            for x in X_seq
        ])
        
        # LightGBM uses last timestep
        X_last_scaled = X_seq_scaled[-1].reshape(1, -1)
        lgb_proba = self.model_lgb.predict(X_last_scaled)[0]
        lgb_pred_idx = np.argmax(lgb_proba)
        lgb_pred_label = self.idx_map_to_class[lgb_pred_idx]
        
        # LSTM uses full sequence
        with torch.no_grad():
            X_seq_tensor = torch.FloatTensor(X_seq_scaled).unsqueeze(0).to(self.device)
            lstm_logits = self.model_lstm(X_seq_tensor)[0].cpu().numpy()
            lstm_proba = np.exp(lstm_logits) / np.exp(lstm_logits).sum()
        
        lstm_pred_idx = np.argmax(lstm_proba)
        lstm_pred_label = self.idx_map_to_class[lstm_pred_idx]
        
        # Ensemble: soft voting
        w_lgb = self.ensemble_weights['lgb']
        w_lstm = self.ensemble_weights['lstm']
        ensemble_proba = w_lgb * lgb_proba + w_lstm * lstm_proba
        
        ensemble_pred_idx = np.argmax(ensemble_proba)
        ensemble_pred_label = self.idx_map_to_class[ensemble_pred_idx]
        ensemble_confidence = ensemble_proba[ensemble_pred_idx]
        
        # Check agreement
        all_agree = (lgb_pred_label == lstm_pred_label == ensemble_pred_label)
        
        # Build result
        result = {
            'label': int(ensemble_pred_label),
            'confidence': float(ensemble_confidence),
            'probabilities': {
                self.idx_map_to_class[i]: float(p)
                for i, p in enumerate(ensemble_proba)
            },
            'model_predictions': {
                'lgb': {
                    'label': int(lgb_pred_label),
                    'confidence': float(lgb_proba[lgb_pred_idx]),
                    'weight': w_lgb
                },
                'lstm': {
                    'label': int(lstm_pred_label),
                    'confidence': float(lstm_proba[lstm_pred_idx]),
                    'weight': w_lstm
                }
            },
            'model_agreement': bool(all_agree),
            'method': 'ensemble_soft_voting'
        }
        
        if return_model_proba:
            result['raw_probabilities'] = {
                'lgb': lgb_proba,
                'lstm': lstm_proba,
                'ensemble': ensemble_proba
            }
        
        return result
    
    def predict_batch(self, X_batch, use_sequences=False):
        """
        Batch prediction for multiple samples.
        
        Args:
            X_batch (np.ndarray): 
                - If use_sequences=False: shape (batch_size, num_features)
                - If use_sequences=True: shape (batch_size, window_size, num_features)
            use_sequences (bool): Treat batch as sequences (True) or timesteps (False)
        
        Returns:
            list: List of prediction dictionaries
        """
        predictions = []
        
        if use_sequences:
            for X_seq in X_batch:
                pred = self.predict_sequence(X_seq)
                predictions.append(pred)
        else:
            for X in X_batch:
                pred = self.predict(X)
                predictions.append(pred)
        
        return predictions
    
    def get_feature_importance(self):
        """
        Get feature importance from LightGBM model.
        
        Returns:
            dict: {feature_name: importance_score}
        """
        importance = self.model_lgb.feature_importance(importance_type='gain')
        return {
            self.feature_cols[i]: float(imp)
            for i, imp in enumerate(importance)
        }
    
    def __repr__(self):
        """String representation."""
        return (
            f"CongestionInferenceEngine("
            f"lgb_weight={self.ensemble_weights['lgb']}, "
            f"lstm_weight={self.ensemble_weights['lstm']}, "
            f"window_size={self.window_size}, "
            f"device={self.device})"
        )


# CLI interface for quick testing
if __name__ == "__main__":
    import sys
    
    # Initialize engine
    engine = CongestionInferenceEngine()
    
    print("Congestion Forecasting Inference Engine")
    print("=" * 70)
    print(f"Model: {engine}")
    print(f"Features: {len(engine.feature_cols)}")
    print(f"Classes: {list(engine.idx_map_to_class.values())}")
    print(f"Ensemble weights: LightGBM={engine.ensemble_weights['lgb']}, "
          f"LSTM={engine.ensemble_weights['lstm']}")
    print(f"Metrics: {engine.metrics}")
    
    # Example usage (requires actual data)
    print("\nExample Usage:")
    print("  1. Single timestep: result = engine.predict(X)")
    print("  2. Sequence (30 timesteps): result = engine.predict_sequence(X_seq)")
    print("  3. Batch: results = engine.predict_batch(X_batch)")
    print("  4. Feature importance: engine.get_feature_importance()")
