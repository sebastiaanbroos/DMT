"""
Regression LSTM Model for Predicting Mood (No extra scaling, per-subject sequences)

Architecture:
- LSTM layer with 64 nodes (tanh activation)
- Dropout (0.2)
- Dense layer with 64 nodes (tanh activation)
- Dropout (0.2)
- Final Dense layer with 1 node (linear activation for regression)

Optimizer: Adam with learning rate 0.0003
Loss: Either MSE or MAE, based on the `loss_type` parameter.
Evaluation Metrics: MAE, MSE, RMSE, and R².
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === PARAMETERS ===
loss_type       = 'mae'      # 'mse' or 'mae'
learning_rate   = 0.0003
sequence_length = 3          # adjust as needed
batch_size      = 16
epochs          = 50

# === 1. LOAD & PREPARE DATA ===
df = pd.read_csv('/Users/s.broos/Documents/DMT/data/daily_removed_incomplete_moods_imputated.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['id','date'], inplace=True)

# Identify features & target
ignore_cols = ['id','date','mood_avg']
feature_cols = [c for c in df.columns if c not in ignore_cols]
target_col   = 'mood_avg'

# === 2. BUILD PER‐SUBJECT SEQUENCES ===
X_seq, y_seq = [], []

for subj, grp in df.groupby('id'):
    grp = grp.sort_values('date')
    X_subj = grp[feature_cols].values         # already normalized features
    y_subj = grp[target_col].values.reshape(-1,1)  # shape (n_samples,1)
    
    # create sliding windows per subject
    for i in range(sequence_length, len(grp)):
        X_seq.append(X_subj[i-sequence_length:i, :])
        y_seq.append(y_subj[i, 0])

# to numpy arrays
X_seq = np.array(X_seq)      # shape: (total_windows, sequence_length, n_features)
y_seq = np.array(y_seq)      # shape: (total_windows,)

# reshape y for Keras
y_seq = y_seq.reshape(-1,1)

# === 3. SPLIT INTO TRAIN & TEST ===
split_idx = int(0.9 * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# === 4. BUILD THE LSTM MODEL ===
model = Sequential([
    LSTM(16, activation='tanh', input_shape=(sequence_length, X_seq.shape[2])),
    Dropout(0.2),
    Dense(16, activation='tanh'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss_type,
    metrics=['mse','mae']
)
model.summary()

# === 5. EARLY STOPPING ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === 6. TRAIN ===
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=1
)

# === 7. EVALUATE ===
test_loss, test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss ({loss_type.upper()}): {test_loss:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# --- Predictions (already original scale) ---
y_pred = model.predict(X_test).flatten()
y_true = y_test.flatten()

# --- Final Metrics on original scale ---
mse_val = mean_squared_error(y_true, y_pred)
mae_val = mean_absolute_error(y_true, y_pred)
rmse_val= np.sqrt(mse_val)
r2_val  = r2_score(y_true, y_pred)

print(f"Final MSE : {mse_val:.4f}")
print(f"Final MAE : {mae_val:.4f}")
print(f"Final RMSE: {rmse_val:.4f}")
print(f"Final R²  : {r2_val:.4f}")
