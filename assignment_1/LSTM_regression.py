import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import (
    Input, Embedding, Reshape,
    Bidirectional, LSTM, Dense, Dropout, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# === PARAMETERS ===
learning_rate   = 3e-4
sequence_length = 3
batch_size      = 16
epochs          = 50
embedding_dim   = 8
lstm_units      = 32
dense_units     = 32
dropout_rate    = 0.2
loss_type       = 'mse'   # or 'mse'

# === 1) LOAD & PREPARE DATA ===
df = pd.read_csv('/Users/s.broos/Documents/DMT/data/daily_removed_incomplete_moods_imputated.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['id','date'], inplace=True)

# Map subjects to indices
df['subj_idx'] = pd.factorize(df['id'])[0]
num_subjects   = df['subj_idx'].nunique()

# Features and target
ignore = ['id','date','mood_avg']
feature_cols = [c for c in df.columns if c not in ignore + ['subj_idx']]
target_col   = 'mood_avg'

# Build per‐subject sequences
X_seq, S_seq, y_seq = [], [], []
for sid, grp in df.groupby('subj_idx'):
    grp = grp.sort_values('date')
    Xg = grp[feature_cols].values
    yg = grp[target_col].values
    for i in range(sequence_length, len(grp)):
        X_seq.append(Xg[i-sequence_length:i])
        S_seq.append(sid)
        y_seq.append(yg[i])

X_seq = np.array(X_seq)   # (n_samples, seq_len, n_features)
S_seq = np.array(S_seq)   # (n_samples,)
y_seq = np.array(y_seq)   # (n_samples,)

# Chronological 90/10 split
X_train, X_test, S_train, S_test, y_train, y_test = train_test_split(
    X_seq, S_seq, y_seq,
    test_size=0.1,
    shuffle=False
)

# === 2) BUILD THE SUBJECT‑AWARE BIDIR LSTM ===
seq_in  = Input(shape=(sequence_length, X_seq.shape[2]), name='seq_input')
subj_in = Input(shape=(1,), name='subj_input')

# subject embed
emb = Embedding(input_dim=num_subjects, output_dim=embedding_dim, name='subj_embed')(subj_in)
emb = Reshape((embedding_dim,))(emb)

# bidirectional LSTM
x = Bidirectional(LSTM(lstm_units, activation='tanh'), name='bi_lstm')(seq_in)
x = Dropout(dropout_rate, name='drop_lstm')(x)

# concat with subject embedding
x = concatenate([x, emb], name='concat')

# dense head
x = Dense(dense_units, activation='relu', name='dense')(x)
x = Dropout(dropout_rate, name='drop_dense')(x)

# regression output
out = Dense(1, activation='linear', name='out')(x)

model = Model(inputs=[seq_in, subj_in], outputs=out, name='SubjectAwareBiLSTM_Regressor')
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss_type,
    metrics=['mse','mae']
)
model.summary()

# === 3) TRAIN ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    {'seq_input': X_train, 'subj_input': S_train},
    y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=1
)

# === 4) EVALUATE ===
# test metrics
test_loss, test_mse, test_mae = model.evaluate(
    {'seq_input': X_test, 'subj_input': S_test},
    y_test, verbose=0
)
y_pred = model.predict({'seq_input': X_test, 'subj_input': S_test}).flatten()

# additional metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nTest {loss_type.upper()}: {test_loss:.4f}")
print(f"Test MSE: {test_mse:.4f} | Test MAE: {test_mae:.4f}")
print(f"RMSE:      {rmse:.4f} | R²:       {r2:.4f}")
