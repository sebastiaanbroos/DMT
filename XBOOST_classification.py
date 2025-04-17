import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === 1) LOAD & SPLIT DATA ===
df = pd.read_csv("/Users/s.broos/Documents/DMT/data/data_after_fe.csv")
df["date"] = pd.to_datetime(df["date"])
df.sort_values(["id","date"], inplace=True)
df["label"] = (df["mood_avg"] >= 0).astype(int)

# Drop non‑features
X = df.drop(columns=["id","date","mood_avg","label"])
y = df["label"].values

# 90/10 train_val / test split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

# === 2) INITIAL MODEL TO GET IMPORTANCE ORDER ===
base_model = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100
)
base_model.fit(X_trainval, y_trainval)

# Build a list of features ordered by descending importance
feat_imp = pd.Series(
    base_model.feature_importances_,
    index=X_trainval.columns
).sort_values(ascending=False)
ordered_features = feat_imp.index.tolist()

# === 3) SWEEP k & COLLECT CV SCORES ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ks, train_accs, val_accs = [], [], []
best_k, best_val_acc = None, -np.inf

for k in range(1, len(ordered_features) + 1):
    sel = ordered_features[:k]
    X_sub = X_trainval[sel]
    
    # 5-fold CV, capturing train & test accuracy
    cv_res = cross_validate(
        base_model,
        X_sub, y_trainval,
        cv=kf,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1
    )
    mean_train = np.mean(cv_res["train_score"])
    mean_val   = np.mean(cv_res["test_score"])
    
    ks.append(k)
    train_accs.append(mean_train)
    val_accs.append(mean_val)
    
    if mean_val > best_val_acc:
        best_val_acc = mean_val
        best_k = k
    
    print(f"k={k:2d} → train_acc={mean_train:.4f}, val_acc={mean_val:.4f}")

print(f"\nBest k by val acc: {best_k} (val acc={best_val_acc:.4f})")

# === 4) PLOT & SAVE ===
plt.figure(figsize=(8,5))
plt.plot(ks, train_accs, marker='o', label="Train Accuracy")
plt.plot(ks, val_accs,   marker='o', label="Val Accuracy")
plt.xlabel("Number of Features (k)")
plt.ylabel("Accuracy")
plt.title("Train vs. Validation Accuracy by # of Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_val_accuracy_vs_k.png", dpi=150)
plt.show()

# === 5) FINAL TRAIN & TEST EVALUATION ===
best_feats = ordered_features[:best_k]
final_model = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100
)
final_model.fit(X_trainval[best_feats], y_trainval)

y_pred = final_model.predict(X_test[best_feats])
print("\nHeld‑out Test Set Performance:")
print("  Accuracy : ", accuracy_score(y_test, y_pred))
print("  Precision: ", precision_score(y_test, y_pred))
print("  Recall   : ", recall_score(y_test, y_pred))
print("  F1 Score : ", f1_score(y_test, y_pred))
