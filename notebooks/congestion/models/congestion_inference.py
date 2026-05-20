import numpy as np
import torch
import joblib
from pathlib import Path

class BiLSTMClassifier(torch.nn.Module):
    """BiLSTM classifier for sequence classification."""
    def __init__(self, input_size, hidden_dim=128, num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_dim, num_layers=num_layers, 
                                   batch_first=True, bidirectional=True, dropout=dropout)
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 = torch.nn.Linear(hidden_dim * 2, 64)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled = self.gap(lstm_out.transpose(1, 2)).squeeze(-1)
        hidden = self.fc1(pooled)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        return logits

class CongestionInference:
    """Inference wrapper for congestion forecasting ensemble."""

    def __init__(self, config_path='models/ensemble_config.pkl'):
        config = joblib.load(config_path)

        self.model_lgb = config['model_lgb']
        self.scaler = config['scaler']
        self.feature_cols = config['feature_cols']
        self.idx_map_to_class = config['idx_map_to_class']
        self.ensemble_weights = config['ensemble_weights']
        self.window_size = config['window_size']
        self.stride = config['stride']

        # Initialize LSTM
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_lstm = BiLSTMClassifier(
            input_size=len(self.feature_cols),
            hidden_dim=128,
            num_layers=2,
            num_classes=4,
            dropout=0.3
        ).to(self.device)
        self.model_lstm.load_state_dict(config['model_lstm'])
        self.model_lstm.eval()

    def predict(self, X):
        """Single timestep prediction.

        Args:
            X: (num_features,) array for single timestep

        Returns:
            label: predicted congestion label (0, 1, 2, 4)
            confidence: max probability across classes
        """
        # Standardize
        X_scaled = self.scaler.transform(X.reshape(1, -1))[0]

        # LightGBM prediction
        lgb_proba = self.model_lgb.predict(X_scaled.reshape(1, -1))[0]

        # Note: For single timesteps, LSTM isn't applicable
        # Use LightGBM only
        pred_idx = np.argmax(lgb_proba)
        pred_label = self.idx_map_to_class[pred_idx]
        confidence = lgb_proba[pred_idx]

        return {
            'label': pred_label,
            'confidence': float(confidence),
            'probabilities': {self.idx_map_to_class[i]: float(p) for i, p in enumerate(lgb_proba)}
        }

    def predict_sequence(self, X_seq):
        """Sequence prediction using ensemble.

        Args:
            X_seq: (window_size, num_features) array

        Returns:
            label: predicted congestion label
            confidence: max probability
            model_agreement: {lgb_label, lstm_label}
        """
        # Standardize sequence
        X_seq_scaled = np.array([self.scaler.transform(x.reshape(1, -1))[0] for x in X_seq])

        # LightGBM uses last timestep
        X_last_scaled = X_seq_scaled[-1].reshape(1, -1)
        lgb_proba = self.model_lgb.predict(X_last_scaled)[0]

        # LSTM uses full sequence
        with torch.no_grad():
            X_seq_tensor = torch.FloatTensor(X_seq_scaled).unsqueeze(0).to(self.device)
            lstm_logits = self.model_lstm(X_seq_tensor)[0].cpu().numpy()
            lstm_proba = np.exp(lstm_logits) / np.exp(lstm_logits).sum()

        # Ensemble
        w_lgb = self.ensemble_weights['lgb']
        w_lstm = self.ensemble_weights['lstm']
        ensemble_proba = w_lgb * lgb_proba + w_lstm * lstm_proba

        pred_idx = np.argmax(ensemble_proba)
        pred_label = self.idx_map_to_class[pred_idx]
        confidence = ensemble_proba[pred_idx]

        # Check model agreement
        lgb_pred = self.idx_map_to_class[np.argmax(lgb_proba)]
        lstm_pred = self.idx_map_to_class[np.argmax(lstm_proba)]
        agreement = (lgb_pred == lstm_pred == pred_label)

        return {
            'label': pred_label,
            'confidence': float(confidence),
            'probabilities': {self.idx_map_to_class[i]: float(p) for i, p in enumerate(ensemble_proba)},
            'model_agreement': {'lgb': lgb_pred, 'lstm': lstm_pred, 'all_agree': bool(agreement)}
        }
