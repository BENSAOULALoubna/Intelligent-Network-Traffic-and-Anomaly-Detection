"""
ML Inference Module for Anomaly Detection
- Portscan: LSTM model (PyTorch)
- Congestion: XGBoost model
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONGESTION_MODEL_PATH = os.path.join(BASE_DIR, "congestion", "xgb_congestion_model.json")
PORTSCAN_MODEL_PATH = os.path.join(BASE_DIR, "port-scanning", "portscan_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "port-scanning", "lstm_scaler.pkl")

# Label mappings
PORTSCAN_LABELS = {0: "Normal", 1: "Horizontal", 2: "Vertical", 3: "Distributed/Spillover"}

# LSTM model class
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=24, hidden_size=128, num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.classifier(last_out)
CONGESTION_LABELS = {0: "Normal", 1: "Queue", 2: "Incast", 3: "Burst", 4: "Periodic", 5: "Link failure"}

# Node encoding for portscan (observer nodes)
OBSERVER_NODES = [
    "isp_mobile0.mobile.gateway",
    "isp_mobile1.mobile.gateway",
    "isp_ftth0.ftth.gateway",
    "isp_private0.private.gateway",
]

# Portscan feature names (20 features)
PORTSCAN_FEATURE_NAMES = [
    'total_syn', 'total_fin', 'syn_rst_ratio', 'syn_to_fin_ratio',
    'failed_conns', 'conn_failure_rate', 'conn_completion_pct',
    'fan_out_ratio', 'port_scan_ratio', 'temporal_scan_density',
    'lb_syn_60s', 'lb_syn_300s', 'syn_iat_mean_ms', 'syn_iat_cv',
    'syn_regularity', 'syn_burst_ratio', 'lookback_acceleration',
    'proto_overhead_pct', 'tcp_ctrl_bytes', 'total_packets'
]


class ModelLoader:
    def __init__(self):
        self.congestion_model = None
        self.portscan_model = None
        self.scaler = None
        self.congestion_feature_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()

    def _load_models(self):
        # Load congestion model (XGBoost from JSON)
        try:
            with open(CONGESTION_MODEL_PATH, 'r') as f:
                model_dict = json.load(f)
            
            # Get feature names if available
            self.congestion_feature_names = model_dict.get('learner', {}).get('feature_names', [])
            
            # Load using xgboost's JSON booster
            self.congestion_model = xgb.Booster()
            self.congestion_model.load_model(CONGESTION_MODEL_PATH)
            print(f"[Inference] Congestion model loaded. Feature count: {len(self.congestion_feature_names)}")
        except Exception as e:
            print(f"[Inference] Error loading congestion model: {e}")

        # Load portscan model (PyTorch state_dict from .pt)
        try:
            self.portscan_model = LSTMClassifier(input_size=24, hidden_size=128,
                                                  num_layers=2, num_classes=4)
            state_dict = torch.load(PORTSCAN_MODEL_PATH, map_location=self.device)
            self.portscan_model.load_state_dict(state_dict)
            self.portscan_model.to(self.device)
            self.portscan_model.eval()
            print("[Inference] Portscan model loaded from .pt")
        except Exception as e:
            print(f"[Inference] Error loading portscan model: {e}")

        # Load scaler (joblib format)
        try:
            self.scaler = joblib.load(SCALER_PATH)
            print("[Inference] Scaler loaded, mean shape:", self.scaler.mean_.shape if hasattr(self.scaler, 'mean_') else '?')
        except Exception as e:
            print(f"[Inference] Error loading scaler: {e}")

    def is_loaded(self):
        return self.congestion_model is not None and self.portscan_model is not None


# Global model loader instance
_models = None


def load_models():
    global _models
    _models = ModelLoader()
    return _models.is_loaded()


def get_models():
    global _models
    if _models is None:
        load_models()
    return _models


# ============================================================================
# PORTSCAN DETECTION (LSTM)
# ============================================================================

def compute_portscan_features(record):
    """Compute 20 features from a single portscan record."""
    features = {}

    total_syn = record.get('total_syn', 0)
    total_fin = record.get('total_fin', 0)
    total_rst = record.get('total_rst', 0)
    failed_conns = record.get('failed_conns', 0)
    conn_attempts = record.get('conn_attempts', 1)
    unique_dst_ips = record.get('unique_dst_ips', 0)
    unique_dst_ports = record.get('unique_dst_ports', 0)
    unique_src_ips = record.get('unique_src_ips', 1)
    tcp_ctrl = record.get('tcp_ctrl_bytes', 0)
    total_pkts = record.get('total_packets', 1)

    features['total_syn'] = total_syn
    features['total_fin'] = total_fin
    features['syn_rst_ratio'] = total_syn / (total_rst + 1)
    features['syn_to_fin_ratio'] = total_syn / (total_fin + 1)
    features['failed_conns'] = failed_conns
    features['conn_failure_rate'] = failed_conns / (conn_attempts + 1)
    features['conn_completion_pct'] = record.get('conn_completion_pct', 0)
    features['fan_out_ratio'] = unique_dst_ips / (unique_src_ips + 1)
    features['port_scan_ratio'] = unique_dst_ports / (unique_dst_ips + 1)
    features['temporal_scan_density'] = unique_dst_ports * unique_dst_ips / (total_syn + 1)
    features['lb_syn_60s'] = record.get('lb_syn_60s', 0)
    features['lb_syn_300s'] = record.get('lb_syn_300s', 0)
    features['syn_iat_mean_ms'] = record.get('syn_iat_mean_ms', 0)
    features['syn_iat_cv'] = record.get('syn_iat_cv', 0)
    features['syn_regularity'] = 1.0 / (abs(record.get('syn_rate_per_sec', 0) - 1.0) + 1)
    features['syn_burst_ratio'] = 1.0 if record.get('syn_rate_per_sec', 0) > 10 else 0.0
    features['lookback_acceleration'] = 0.0
    features['proto_overhead_pct'] = record.get('proto_overhead_pct', 0)
    features['tcp_ctrl_bytes'] = tcp_ctrl
    features['total_packets'] = total_pkts

    return features


def prepare_lstm_input(window_records, scaler):
    """Prepare input for LSTM: (1, 12, 24)"""
    if len(window_records) < 12:
        return None

    recent = window_records[-12:]
    feature_matrix = []

    for record in recent:
        features = compute_portscan_features(record)
        feature_vec = [features.get(fn, 0) for fn in PORTSCAN_FEATURE_NAMES]
        
        # Add 4 node dummies
        node_id = record.get('observer_node', '')
        dummies = [0, 0, 0, 0]
        if node_id in OBSERVER_NODES:
            idx = OBSERVER_NODES.index(node_id)
            if idx < 4:
                dummies[idx] = 1
        feature_vec.extend(dummies)
        feature_matrix.append(feature_vec)

    arr = np.array(feature_matrix, dtype=np.float32)
    original_shape = arr.shape
    arr_flat = arr.reshape(-1, 24)
    arr_scaled = scaler.transform(arr_flat)
    arr = arr_scaled.reshape(original_shape)
    arr = np.expand_dims(arr, axis=0)
    return arr


def detect_portscan(window_records):
    """Run portscan LSTM detection."""
    models = get_models()
    if models.portscan_model is None or models.scaler is None:
        return None, 0.0

    try:
        X = prepare_lstm_input(window_records, models.scaler)
        if X is None:
            return None, 0.0

        X_tensor = torch.tensor(X, dtype=torch.float32).to(models.device)
        with torch.no_grad():
            output = models.portscan_model(X_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        label = PORTSCAN_LABELS.get(pred_class, "Unknown")
        return label, confidence

    except Exception as e:
        print(f"[Portscan Detection] Error: {e}")
        return None, 0.0


# ============================================================================
# CONGESTION DETECTION (XGBoost)
# ============================================================================

def compute_congestion_features(record, history=None):
    """Compute features for XGBoost from a congestion record."""
    features = {}
    
    # Basic features
    features['bw_in_mbps'] = record.get('bw_in_mbps', 0)
    features['bw_out_mbps'] = record.get('bw_out_mbps', 0)
    features['bw_avg_mbps'] = record.get('bw_avg_mbps', 0)
    features['util_in_pct'] = record.get('util_in_pct', 0)
    features['util_out_pct'] = record.get('util_out_pct', 0)
    features['pkt_in'] = record.get('pkt_in', 0)
    features['pkt_out'] = record.get('pkt_out', 0)
    features['pkt_dropped'] = record.get('pkt_dropped', 0)
    features['pkt_loss_pct'] = record.get('pkt_loss_pct', 0)
    features['bytes_in'] = record.get('bytes_in', 0)
    features['bytes_out'] = record.get('bytes_out', 0)
    features['payload_bytes'] = record.get('payload_bytes', 0)
    features['total_bytes'] = record.get('total_bytes', 0)
    features['jitter_ms'] = record.get('jitter_ms', 0)
    features['iat_mean_ms'] = record.get('iat_mean_ms', 0)
    features['rtt_avg_ms'] = record.get('rtt_avg_ms', 0)
    features['rtt_std_ms'] = record.get('rtt_std_ms', 0)
    features['ospf_pkts'] = record.get('ospf_pkts', 0)
    features['arp_pkts'] = record.get('arp_pkts', 0)
    features['icmp_pkts'] = record.get('icmp_pkts', 0)
    features['udp_pkts'] = record.get('udp_pkts', 0)
    features['tcp_pkts'] = record.get('tcp_pkts', 0)
    features['udp_fraction'] = record.get('udp_fraction', 0)
    features['tcp_fraction'] = record.get('tcp_fraction', 0)
    features['proto_overhead_pct'] = record.get('proto_overhead_pct', 0)
    features['ospf_overhead_pct'] = record.get('ospf_overhead_pct', 0)
    features['tcp_ctrl_pct'] = record.get('tcp_ctrl_pct', 0)
    features['tcp_syn'] = record.get('tcp_syn', 0)
    features['tcp_fin'] = record.get('tcp_fin', 0)
    features['tcp_rst'] = record.get('tcp_rst', 0)
    features['tcp_retransmit_pct'] = record.get('tcp_retransmit_pct', 0)
    features['syn_rate_pps'] = record.get('syn_rate_pps', 0)
    features['rst_rate_pps'] = record.get('rst_rate_pps', 0)
    features['fin_rate_pps'] = record.get('fin_rate_pps', 0)
    features['active_flows'] = record.get('active_flows', 0)
    features['new_flows'] = record.get('new_flows', 0)
    features['finished_flows'] = record.get('finished_flows', 0)
    features['flow_churn_per_sec'] = record.get('flow_churn_per_sec', 0)
    features['peak_kbps_1s_window'] = record.get('peak_kbps_1s_window', 0)

    # Node encoding
    node_map = {'boundary': 0, 'core0': 1, 'core1': 2, 'core2': 3, 'edge0': 4, 'edge1': 5}
    features['node_id_enc'] = node_map.get(record.get('node_id', ''), 0)

    # Lag features
    bw = record.get('bw_avg_mbps', 0)
    pkt_loss = record.get('pkt_loss_pct', 0)
    jitter = record.get('jitter_ms', 0)
    retrans = record.get('tcp_retransmit_pct', 0)
    syn_rate = record.get('syn_rate_pps', 0)
    
    for lag in [1, 3, 5, 10]:
        features[f'bw_avg_mbps_lag{lag}'] = bw
        features[f'pkt_loss_pct_lag{lag}'] = pkt_loss
        features[f'jitter_ms_lag{lag}'] = jitter
        features[f'tcp_retransmit_pct_lag{lag}'] = retrans
        features[f'syn_rate_pps_lag{lag}'] = syn_rate

    # Rolling features (using history if available)
    if history:
        for window in [5, 10, 30]:
            bw_vals = [r.get('bw_avg_mbps', 0) for r in history[-window:]]
            features[f'bw_avg_mbps_roll_mean_{window}'] = np.mean(bw_vals) if bw_vals else 0
            features[f'bw_avg_mbps_roll_std_{window}'] = np.std(bw_vals) if len(bw_vals) > 1 else 0
            features[f'bw_avg_mbps_roll_max_{window}'] = max(bw_vals) if bw_vals else 0
    else:
        for window in [5, 10, 30]:
            features[f'bw_avg_mbps_roll_mean_{window}'] = bw
            features[f'bw_avg_mbps_roll_std_{window}'] = 0
            features[f'bw_avg_mbps_roll_max_{window}'] = bw

    for window in [5, 10, 30]:
        features[f'pkt_loss_pct_roll_mean_{window}'] = pkt_loss
        features[f'pkt_loss_pct_roll_std_{window}'] = 0
        features[f'jitter_ms_roll_mean_{window}'] = jitter
        features[f'jitter_ms_roll_std_{window}'] = 0
        features[f'tcp_retransmit_pct_roll_mean_{window}'] = retrans
        features[f'tcp_retransmit_pct_roll_std_{window}'] = 0
        features[f'flow_churn_per_sec_roll_mean_{window}'] = record.get('flow_churn_per_sec', 0)
        features[f'flow_churn_per_sec_roll_std_{window}'] = 0
        features[f'peak_kbps_1s_window_roll_mean_{window}'] = record.get('peak_kbps_1s_window', 0)
        features[f'peak_kbps_1s_window_roll_std_{window}'] = 0
        features[f'peak_kbps_1s_window_roll_max_{window}'] = record.get('peak_kbps_1s_window', 0)

    # Delta features
    for delta in [1, 5]:
        features[f'bw_avg_mbps_delta{delta}'] = 0
        features[f'bw_avg_mbps_pct_chg'] = 0
        features[f'pkt_loss_pct_delta{delta}'] = 0
        features[f'pkt_loss_pct_pct_chg'] = 0
        features[f'jitter_ms_delta{delta}'] = 0
        features[f'jitter_ms_pct_chg'] = 0
        features[f'util_in_pct_delta{delta}'] = 0
        features[f'util_in_pct_pct_chg'] = 0
        features[f'util_out_pct_delta{delta}'] = 0
        features[f'util_out_pct_pct_chg'] = 0

    # CV features
    features['bw_avg_mbps_cv_10'] = 0
    features['bw_avg_mbps_cv_30'] = 0
    features['pkt_loss_pct_cv_10'] = 0
    features['pkt_loss_pct_cv_30'] = 0

    # Burst indicators
    features['bw_avg_mbps_burst'] = 1.0 if bw > 8000 else 0.0
    features['syn_rate_pps_burst'] = 1.0 if syn_rate > 50 else 0.0
    features['flow_churn_per_sec_burst'] = 1.0 if record.get('flow_churn_per_sec', 0) > 100 else 0.0

    # Derived ratios
    features['syn_flood_indicator'] = 1.0 if syn_rate > 100 else 0.0
    features['util_imbalance'] = abs(record.get('util_in_pct', 0) - record.get('util_out_pct', 0))
    features['util_max'] = max(record.get('util_in_pct', 0), record.get('util_out_pct', 0))
    features['bw_asym_abs'] = abs(record.get('bw_in_mbps', 0) - record.get('bw_out_mbps', 0))
    features['loss_per_mbps'] = pkt_loss / (record.get('bw_avg_mbps', 1) + 1)
    features['retransmit_load'] = retrans * bw
    features['jitter_per_mbps'] = jitter / (bw + 1)
    features['t_log'] = 0
    features['t_relative'] = 0

    # Additional derived
    pkt_in = record.get('pkt_in', 0)
    pkt_out = record.get('pkt_out', 0)
    features['pkt_asymmetry'] = abs(pkt_in - pkt_out) / (pkt_in + pkt_out + 1)
    features['bw_asymmetry'] = abs(record.get('bw_in_mbps', 0) - record.get('bw_out_mbps', 0)) / (bw + 1)
    features['flow_density'] = record.get('active_flows', 0) / 1000
    features['ctrl_to_data_ratio'] = record.get('tcp_ctrl_pct', 0) / (101 - record.get('tcp_ctrl_pct', 1))
    features['bytes_per_pkt_out'] = record.get('bytes_out', 0) / (pkt_out + 1)
    features['syn_rst_ratio_router'] = record.get('tcp_syn', 0) / (record.get('tcp_rst', 1) + 1)
    features['is_portscan'] = 0

    return features


def detect_congestion(record, history=None):
    """Run congestion XGBoost detection."""
    models = get_models()
    if models.congestion_model is None:
        return None, 0.0

    try:
        features = compute_congestion_features(record, history)
        
        # Get feature vector in correct order
        if models.congestion_feature_names:
            # Use model's expected feature order
            feature_order = models.congestion_feature_names
            feature_vec = [features.get(fn, 0) for fn in feature_order]
        else:
            # Fallback: use computed features
            feature_vec = list(features.values())

        X = np.array([feature_vec], dtype=np.float32)
        dmat = xgb.DMatrix(X)

        # Get prediction probabilities
        probs = models.congestion_model.predict(dmat)
        pred_class = np.argmax(probs[0])
        confidence = probs[0][pred_class]

        label = CONGESTION_LABELS.get(int(pred_class), "Unknown")
        return label, float(confidence)

    except Exception as e:
        print(f"[Congestion Detection] Error: {e}")
        return None, 0.0


# ============================================================================
# COMBINED DETECTION
# ============================================================================

def detect_all(portscan_window, congestion_record, congestion_history=None):
    """
    Combined detection - prioritizes portscan over congestion
    Returns: (type, label, confidence)
    """
    # Detect portscan first (higher priority)
    ps_label, ps_conf = detect_portscan(portscan_window)

    if ps_label and ps_label != "Normal" and ps_conf >= 0.5:
        return "portscan", ps_label, ps_conf

    # Then check congestion
    cg_label, cg_conf = detect_congestion(congestion_record, congestion_history)

    if cg_label and cg_label != "Normal" and cg_conf >= 0.5:
        return "congestion", cg_label, cg_conf

    return None, "Normal", 0.0