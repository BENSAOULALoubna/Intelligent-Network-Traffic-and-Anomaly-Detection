import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

processed = Path("notebooks/port-scanning/results/processed_data")
X_train = np.load(processed / "X_train.npy")
B, T, F = X_train.shape
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, F))
joblib.dump(scaler, processed / "scaler.pkl")
print("scaler.pkl saved")