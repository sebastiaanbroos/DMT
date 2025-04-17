import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === PARAMETERS ===
DATA_PATH    = "/Users/s.broos/Documents/DMT/data/data_after_fe.csv"
TARGET       = "mood_avg"
TEST_SIZE    = 0.10
CV_FOLDS     = 5
RANDOM_STATE = 42

# XGB defaults (can be tuned later)
xgb_kwargs = dict(
    random_state=RANDOM_STATE,
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    objective='reg:squarederror'
)

# === 1) LOAD & HOLD‑OUT TEST SET ===
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['id','date'], inplace=True)

# features / target
X = df.drop(columns=['id','date', TARGET])
y = df[TARGET].values

# chronological 90/10 split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    shuffle=False,
    random_state=RANDOM_STATE
)

# === 2) INITIAL XGB TO RANK FEATURES ===
base = XGBRegressor(**xgb_kwargs)
base.fit(X_trainval, y_trainval)

feat_imp = pd.Series(base.feature_importances_, index=X_trainval.columns)
ordered_features = feat_imp.sort_values(ascending=False).index.tolist()

# === 3) SWEEP k & COLLECT CV MSE ===
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

ks, train_mses, val_mses = [], [], []
best_k, best_val_mse = 0, np.inf

for k in range(1, len(ordered_features) + 1):
    sel = ordered_features[:k]
    X_sub = X_trainval[sel]
    
    cv_res = cross_validate(
        XGBRegressor(**xgb_kwargs),
        X_sub, y_trainval,
        cv=kf,
        scoring='neg_mean_squared_error',    # use MSE
        return_train_score=True,
        n_jobs=-1
    )
    train_mse = -cv_res['train_score'].mean()
    val_mse   = -cv_res['test_score'].mean()
    
    ks.append(k)
    train_mses.append(train_mse)
    val_mses.append(val_mse)
    
    if val_mse < best_val_mse:
        best_val_mse, best_k = val_mse, k
    
    print(f"k={k:2d} → train MSE={train_mse:.4f}, val MSE={val_mse:.4f}")

print(f"\nBest k by val MSE: {best_k} (MSE={best_val_mse:.4f})")

# === 4) PLOT TRAIN vs VALIDATION MSE BY # FEATURES ===
plt.figure(figsize=(8,5))
plt.plot(ks, train_mses, marker='o', label='Train MSE')
plt.plot(ks, val_mses,   marker='o', label='Val   MSE')
plt.xlabel("Number of Features (k)")
plt.ylabel("Mean Squared Error")
plt.title("Train vs. Validation MSE by # of Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_val_mse_vs_k.png", dpi=150)
plt.show()

# === 5) FINAL TRAIN & TEST EVALUATION ===
best_feats = ordered_features[:best_k]
final_model = XGBRegressor(**xgb_kwargs)
final_model.fit(X_trainval[best_feats], y_trainval)

y_pred = final_model.predict(X_test[best_feats])
mae   = mean_absolute_error(y_test, y_pred)
mse   = mean_squared_error(y_test, y_pred)
rmse  = np.sqrt(mse)
r2    = r2_score(y_test, y_pred)

print("\nHeld‑out Test Set Performance:")
print(f"  MAE : {mae:.4f}")
print(f"  MSE : {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²  : {r2:.4f}")
