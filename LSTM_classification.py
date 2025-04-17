import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === PARAMETERS ===
learning_rate   = 0.0003
sequence_length = 3     # number of past samples to look at
batch_size      = 16
epochs          = 50

# === 1. LOAD & PREPARE DATA ===
df = pd.read_csv('/Users/s.broos/Documents/DMT/data/daily_removed_incomplete_moods_imputated.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['id','date'], inplace=True)

# Create binary label if missing
if 'mood_label' not in df.columns:
    df['mood_label'] = (df['mood_avg'] >= 0).astype(int)

# Identify features (no scaling)
ignore = ['id','date','mood_avg','mood_label']
feature_cols = [c for c in df.columns if c not in ignore]

# === 2. BUILD PER‑SUBJECT SEQUENCES ===
X_seq, y_seq = [], []
for subj, grp in df.groupby('id'):
    grp = grp.sort_values('date')
    X_subj = grp[feature_cols].values
    y_subj = grp['mood_label'].values
    
    for i in range(sequence_length, len(grp)):
        X_seq.append(X_subj[i-sequence_length:i, :])
        y_seq.append(y_subj[i])

X_seq = np.array(X_seq)    # (n_windows, seq_len, n_features)
y_seq = np.array(y_seq)    # (n_windows,)

# === 3. SPLIT INTO TRAIN & TEST ===
split = int(0.9 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# === 4. BUILD THE LSTM CLASSIFIER ===
model = Sequential([
    LSTM(16, activation='tanh', input_shape=(sequence_length, X_seq.shape[2])),
    Dropout(0.2),
    Dense(16, activation='tanh'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
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

# === 7. EVALUATE & METRICS ===
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# Predictions & additional metrics
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("F1 Score:   ", f1_score(y_test, y_pred))
print("Precision:  ", precision_score(y_test, y_pred))
print("Recall:     ", recall_score(y_test, y_pred))
print("Accuracy:   ", accuracy_score(y_test, y_pred))
