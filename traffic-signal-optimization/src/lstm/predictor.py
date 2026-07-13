import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import pickle

# ─────────────────────────────────────────────
# LSTM MODEL DEFINITION (must match Colab)
# ─────────────────────────────────────────────

class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ─────────────────────────────────────────────
# PREDICTOR CLASS
# ─────────────────────────────────────────────

class TrafficPredictor:
    def __init__(self,
                 model_path  = 'models/lstm_best.pth',
                 scaler_X_path = 'models/scaler_X.pkl',
                 scaler_y_path = 'models/scaler_y.pkl',
                 feature_cols_path = 'models/feature_cols.pkl'):

        # Load scalers
        with open(scaler_X_path, 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            self.scaler_y = pickle.load(f)
        with open(feature_cols_path, 'rb') as f:
            self.feature_cols = pickle.load(f)

        self.input_size = len(self.feature_cols)
        self.seq_len    = 24   # must match training

        # Load model
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TrafficLSTM(input_size=self.input_size).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path,
                       map_location=self.device))
        self.model.eval()

        # Rolling buffer of recent states
        self.buffer = []

        print(f"[Predictor] Model loaded — "
              f"input={self.input_size} features, "
              f"seq_len={self.seq_len}")

    def update_buffer(self, state_dict):
        """
        Add latest network state to rolling buffer.

        Args:
            state_dict: output of get_network_state()
                        {tls_id: {waiting_time, queue_length, vehicle_count}}
        """
        from src.utils.tls_ids import TLS_IDS

        # Extract only the features the model was trained on
        row = {}
        for tls_id in TLS_IDS:
            s     = state_dict[tls_id]
            short = tls_id[:20]
            row[f'{short}_queue'] = s['queue_length']
            row[f'{short}_veh']   = s['vehicle_count']

        # Keep only feature_cols order
        vec = [row.get(col, 0.0) for col in self.feature_cols]
        self.buffer.append(vec)

        # Keep only last seq_len steps
        if len(self.buffer) > self.seq_len:
            self.buffer = self.buffer[-self.seq_len:]

    def predict(self):
        """
        Predict total waiting time 60 seconds ahead.

        Returns:
            float: predicted total waiting time (seconds)
                   or None if buffer not full yet
        """
        if len(self.buffer) < self.seq_len:
            return None   # not enough history yet

        X = np.array(self.buffer, dtype=np.float32)
        X = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_scaled = self.model(X_tensor).cpu().numpy()

        pred = self.scaler_y.inverse_transform(pred_scaled)
        return float(pred[0, 0])

    def is_ready(self):
        """True when buffer has enough history to predict."""
        return len(self.buffer) >= self.seq_len