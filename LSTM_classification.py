import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Input, Embedding, Reshape,
    Bidirectional, LSTM, Dense, Dropout, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# === PARAMETERS ===
learning_rate   = 3e-4
sequence_length = 3     # past days
batch_size      = 16
epochs          = 50
embedding_dim   = 8
lstm_units1     = 32
lstm_units2     = 16
dense_units     = 32
dropout_rate    = 0.3

# === 1) LOAD & PREPARE DATA ===
df = pd.read_csv("/Users/s.broos/Documents/DMT/data/daily_removed_incomplete_moods_imputated.csv")
df["date"] = pd.to_datetime(df["date"])
df.sort_values(["id","date"], inplace=True)

# Map subjects to indices for embedding
df["subj_idx"] = pd.factorize(df["id"])[0]
num_subjects = df["subj_idx"].nunique()

# 3 mood classes on normalized scale
df["label"] = pd.cut(
    df["mood_avg"],
    bins=[-np.inf, -0.5, 0.5, np.inf],
    labels=[0, 1, 2]
).astype(int)

# Feature columns (exclude id, date, mood_avg, label, subj_idx)
ignore = ["id","date","mood_avg","label"]
feature_cols = [c for c in df.columns if c not in ignore + ["subj_idx"]]

# Build sequences
X_seq, S_seq, y_seq = [], [], []
for sid, grp in df.groupby("subj_idx"):
    grp = grp.sort_values("date")
    Xg = grp[feature_cols].values
    yg = grp["label"].values
    for i in range(sequence_length, len(grp)):
        X_seq.append(Xg[i-sequence_length:i])
        S_seq.append(sid)
        y_seq.append(yg[i])

X_seq = np.array(X_seq)
S_seq = np.array(S_seq)
y_seq = np.array(y_seq)

# Train/test split (90/10), stratified on y
X_train, X_test, S_train, S_test, y_train, y_test = train_test_split(
    X_seq, S_seq, y_seq,
    test_size=0.1,
    stratify=y_seq,
    random_state=42
)

# Compute class weights
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = dict(zip(classes, weights))

# === 2) BUILD THE MODEL ===
# Sequence input
seq_in  = Input(shape=(sequence_length, X_seq.shape[2]), name="seq_input")
# Subject ID input
subj_in = Input(shape=(1,), name="subj_input")
# Embedding
emb = Embedding(input_dim=num_subjects, output_dim=embedding_dim, name="subj_embed")(subj_in)
emb = Reshape((embedding_dim,))(emb)

# Stacked bidirectional LSTM
x = Bidirectional(LSTM(lstm_units1, return_sequences=True), name="bilstm_1")(seq_in)
x = Bidirectional(LSTM(lstm_units2),            name="bilstm_2")(x)
x = Dropout(dropout_rate, name="dropout_lstm")(x)

# Combine with subject embedding
x = concatenate([x, emb], name="concat")

# Dense layers
x = Dense(dense_units, activation="relu", name="dense_1")(x)
x = Dropout(dropout_rate, name="dropout_dense")(x)
out = Dense(3, activation="softmax", name="out")(x)

model = Model(inputs=[seq_in, subj_in], outputs=out, name="SubjectAware_BiLSTM")
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy"
)
model.summary()

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

# === 3) TRAIN ===
history = model.fit(
    {"seq_input": X_train, "subj_input": S_train},
    y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === 4) EVALUATE ===
y_prob = model.predict({"seq_input": X_test, "subj_input": S_test})
y_pred = np.argmax(y_prob, axis=1)

print("\nTest set classification report:")
print(classification_report(y_test, y_pred, digits=4))

# === 5) PLOT LOSS ===
plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
