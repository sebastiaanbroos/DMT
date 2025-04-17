import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, GRU,
    Dense, Dropout, concatenate, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === PARAMETERS ===
learning_rate   = 3e-4
sequence_length = 3     # number of past samples
batch_size      = 16
epochs          = 50
embedding_dim   = 8

# === 1. LOAD & PREPARE DATA ===
df = pd.read_csv('/Users/s.broos/Documents/DMT/data/daily_removed_incomplete_moods_imputated.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['id','date'], inplace=True)

# Create binary label if missing
if 'mood_label' not in df.columns:
    df['mood_label'] = (df['mood_avg'] >= 0).astype(int)

# Map subject IDs to integer indices
df['subj_idx'] = pd.factorize(df['id'])[0]
num_subjects   = df['subj_idx'].nunique()

# Identify feature columns (exclude id, date, mood_avg, mood_label, subj_idx)
ignore = ['id','date','mood_avg','mood_label','subj_idx']
feature_cols = [c for c in df.columns if c not in ignore]

# === 2. BUILD PER‑SUBJECT SEQUENCES ===
X_seq, S_seq, y_seq = [], [], []
for subj, grp in df.groupby('subj_idx'):
    grp = grp.sort_values('date')
    Xg = grp[feature_cols].values
    yg = grp['mood_label'].values
    for i in range(sequence_length, len(grp)):
        X_seq.append(Xg[i-sequence_length:i, :])
        S_seq.append(subj)
        y_seq.append(yg[i])

X_seq = np.array(X_seq)   # (n_samples, seq_len, n_features)
S_seq = np.array(S_seq)   # (n_samples,)
y_seq = np.array(y_seq)   # (n_samples,)

# === 3. SPLIT INTO TRAIN & TEST (90/10) ===
split_idx = int(0.9 * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
S_train, S_test = S_seq[:split_idx], S_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# === 4. BUILD THE SUBJECT‑AWARE GRU CLASSIFIER ===
# Sequence input
seq_in = Input(shape=(sequence_length, X_seq.shape[2]), name='seq_input')
# Subject input
subj_in = Input(shape=(1,), name='subj_input')

# Embedding for subject IDs
emb = Embedding(input_dim=num_subjects, output_dim=embedding_dim, name='subj_embed')(subj_in)
emb_flat = Reshape((embedding_dim,))(emb)

# Bidirectional GRU over the sequence
x = Bidirectional(GRU(32, activation='tanh'), name='bi_gru')(seq_in)
x = Dropout(0.2, name='drop_gru')(x)

# Concatenate GRU output with subject embedding
x = concatenate([x, emb_flat], name='concat')

# Dense layers
x = Dense(32, activation='relu', name='dense1')(x)
x = Dropout(0.2, name='drop_dense')(x)
out = Dense(1, activation='sigmoid', name='out')(x)

model = Model(inputs=[seq_in, subj_in], outputs=out, name='SubjectAwareGRU_Classifier')
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
    {'seq_input': X_train, 'subj_input': S_train},
    y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=1
)

# === 7. EVALUATE & METRICS ===
loss, acc = model.evaluate(
    {'seq_input': X_test, 'subj_input': S_test},
    y_test,
    verbose=0
)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# Get predicted labels
y_prob = model.predict({'seq_input': X_test, 'subj_input': S_test}).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("F1 Score:   ", f1_score(y_test, y_pred))
print("Precision:  ", precision_score(y_test, y_pred))
print("Recall:     ", recall_score(y_test, y_pred))
print("Accuracy:   ", accuracy_score(y_test, y_pred))
