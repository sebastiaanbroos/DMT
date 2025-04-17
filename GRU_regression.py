"""
Subject‑Aware Bidirectional GRU Model for Predicting Mood

Architecture:
- Input A: sequence of past `sequence_length` days of features
- Input B: subject ID → Embedding( num_subjects → 8 dims )
- Bidirectional GRU (32 units, tanh) over the sequence
- Dropout(0.2)
- Concatenate GRU output + subject embedding
- Dense(32, relu) → Dropout(0.2)
- Dense(1, linear) for regression

Optimizer: Adam(lr=0.0003), loss = MAE or MSE (configurable)
Metrics: MAE, MSE, RMSE, R²
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, GRU,
    Dense, Dropout, concatenate, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === PARAMETERS ===
loss_type       = 'mae'       # 'mse' or 'mae'
learning_rate   = 3e-4
sequence_length = 3           # try a longer window
batch_size      = 16
epochs          = 100
embedding_dim   = 8

# === 1. LOAD & PREPARE DATA ===
df = pd.read_csv('/Users/s.broos/Documents/DMT/data/daily_removed_incomplete_moods_imputated.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['id','date'], inplace=True)

# Map subject IDs to integers
df['subj_idx'] = pd.factorize(df['id'])[0]
num_subjects   = df['subj_idx'].nunique()

# Identify features & target
ignore = ['id','date','mood_avg','mood_avg_hist','subj_idx']
feature_cols = [c for c in df.columns if c not in ignore]
target_col   = 'mood_avg'

# === 2. BUILD PER‑SUBJECT SEQUENCES (with subj indices) ===
X_seq, S_seq, y_seq = [], [], []
for subj, grp in df.groupby('subj_idx'):
    grp = grp.sort_values('date')
    Xg = grp[feature_cols].values
    yg = grp[target_col].values
    for i in range(sequence_length, len(grp)):
        X_seq.append(Xg[i-sequence_length:i])
        S_seq.append(subj)           # same subj for each window
        y_seq.append(yg[i])

X_seq = np.array(X_seq)    # (n_windows, seq_len, n_features)
S_seq = np.array(S_seq)    # (n_windows,)
y_seq = np.array(y_seq).reshape(-1,1)

# === 3. SPLIT INTO TRAIN & TEST (90/10) ===
split_idx = int(0.9 * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
S_train, S_test = S_seq[:split_idx], S_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# === 4. BUILD THE SUBJECT‑AWARE GRU MODEL ===
# Sequence input
seq_in = Input(shape=(sequence_length, X_seq.shape[2]), name='seq_input')
# Subject input
subj_in = Input(shape=(1,), name='subj_input')

# Embedding for subject
emb = Embedding(input_dim=num_subjects, output_dim=embedding_dim, name='subj_embed')(subj_in)
emb_flat = Reshape((embedding_dim,))(emb)

# Bidirectional GRU
x = Bidirectional(GRU(32, activation='tanh'), name='bi_gru')(seq_in)
x = Dropout(0.2, name='drop_gru')(x)

# Concatenate embedding + GRU output
x = concatenate([x, emb_flat], name='concat')

# Dense layers
x = Dense(32, activation='relu', name='dense1')(x)
x = Dropout(0.2, name='drop_dense')(x)
out = Dense(1, activation='linear', name='out')(x)

model = Model(inputs=[seq_in, subj_in], outputs=out, name='SubjectAwareGRU')
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss_type,
    metrics=['mse','mae']
)
model.summary()

# === 5. EARLY STOPPING ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# === 6. TRAIN ===
history = model.fit(
    {'seq_input': X_train, 'subj_input': S_train},
    y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=1
)

# === 7. EVALUATE ===
test_loss, test_mse, test_mae = model.evaluate(
    {'seq_input': X_test, 'subj_input': S_test},
    y_test,
    verbose=0
)
print(f"\nTest Loss ({loss_type.upper()}): {test_loss:.4f}")
print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")

# --- Predictions & Final Metrics ---
y_pred = model.predict({'seq_input': X_test, 'subj_input': S_test}).flatten()
y_true = y_test.flatten()

mse_val  = mean_squared_error(y_true, y_pred)
mae_val  = mean_absolute_error(y_true, y_pred)
rmse_val = np.sqrt(mse_val)
r2_val   = r2_score(y_true, y_pred)

print(f"Final MSE : {mse_val:.4f}")
print(f"Final MAE : {mae_val:.4f}")
print(f"Final RMSE: {rmse_val:.4f}")
print(f"Final R²  : {r2_val:.4f}")
