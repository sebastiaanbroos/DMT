import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report

# === 1) LOAD & PREPARE DATA ===
df = pd.read_csv("/Users/s.broos/Documents/DMT/data/data_after_fe.csv")
df["date"] = pd.to_datetime(df["date"])
df.sort_values(["id","date"], inplace=True)

# Define 3 classes on the normalized mood_avg
df["label"] = pd.cut(
    df["mood_avg"],
    bins=[-np.inf, -0.5, 0.5, np.inf],
    labels=[0, 1, 2]
).astype(int)

# Drop non‑features
X = df.drop(columns=["id","date","mood_avg","label"])
y = df["label"].values

# 90/10 train_val / test split (stratified)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

# === 2) INITIAL XGBOOST FOR FEATURE IMPORTANCE ===
base_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42,
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100
)
base_model.fit(X_trainval, y_trainval)

# Order features by importance
feat_imp = pd.Series(base_model.feature_importances_, index=X_trainval.columns)
ordered_features = feat_imp.sort_values(ascending=False).index.tolist()

# === 3) SWEEP k & COLLECT CV MACRO‑F1 SCORES ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ks, train_f1s, val_f1s = [], [], []
best_k, best_val_f1 = 0, -np.inf

for k in range(1, len(ordered_features)+1):
    sel = ordered_features[:k]
    cv_res = cross_validate(
        base_model,
        X_trainval[sel], y_trainval,
        cv=kf,
        scoring=['f1_macro'],
        return_train_score=True,
        n_jobs=-1
    )
    mean_train = cv_res['train_f1_macro'].mean()
    mean_val   = cv_res['test_f1_macro'].mean()
    
    ks.append(k)
    train_f1s.append(mean_train)
    val_f1s.append(mean_val)
    
    if mean_val > best_val_f1:
        best_val_f1, best_k = mean_val, k
    
    print(f"k={k:2d} → train F1_macro={mean_train:.4f}, val F1_macro={mean_val:.4f}")

print(f"\nBest k by val F1_macro: {best_k} (val F1_macro={best_val_f1:.4f})")

# === 4) PLOT TRAIN vs VALIDATION F1_MACRO BY # FEATURES ===
plt.figure(figsize=(8,5))
plt.plot(ks, train_f1s, marker='o', label="Train F1_macro")
plt.plot(ks, val_f1s,   marker='o', label="Val   F1_macro")
plt.xlabel("Number of Features (k)")
plt.ylabel("Macro‑F1 Score")
plt.title("Train vs. Validation Macro‑F1 by # of Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_val_macro_f1_vs_k.png", dpi=150)
plt.show()

# === 5) FINAL TRAIN & TEST EVALUATION ===
final_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42,
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100
)
selected = ordered_features[:best_k]
final_model.fit(X_trainval[selected], y_trainval)

y_pred = final_model.predict(X_test[selected])
print("\nHeld‑out Test Set Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
