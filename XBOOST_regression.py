import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# === PARAMETERS ===
DATA_PATH       = "/Users/s.broos/Documents/DMT/data/data_after_fe.csv"
TARGET          = "mood_avg"
TEST_SIZE       = 0.10
CV_FOLDS        = 5
RANDOM_STATE    = 42

# XGB defaults (you can tune these later)
xgb_kwargs = dict(
    random_state=RANDOM_STATE,
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    objective='reg:squarederror'
)

# === 1) LOAD & HOLD OUT TEST SET ===
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['id','date'], inplace=True)

X = df.drop(columns=['id','date', TARGET]).copy()
y = df[TARGET].values

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# === 2) GET FEATURE IMPORTANCE ORDERING ===
base = XGBRegressor(**xgb_kwargs)
base.fit(X_trainval, y_trainval)

imp = pd.Series(base.feature_importances_, index=X.columns)
ordered_features = imp.sort_values(ascending=False).index.tolist()

# === 3) SWEEP k AND COLLECT CV MAE ===
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

ks = []
train_maes = []
val_maes = []

best_k = None
best_val_mae = np.inf

for k in range(1, len(ordered_features)+1):
    feats = ordered_features[:k]
    X_sub = X_trainval[feats].values
    
    cv_res = cross_validate(
        XGBRegressor(**xgb_kwargs),
        X_sub, y_trainval,
        cv=kf,
        scoring='neg_mean_absolute_error',
        return_train_score=True,
        n_jobs=-1
    )
    mean_train_mae = -cv_res['train_score'].mean()
    mean_val_mae   = -cv_res['test_score'].mean()
    
    ks.append(k)
    train_maes.append(mean_train_mae)
    val_maes.append(mean_val_mae)
    
    if mean_val_mae < best_val_mae:
        best_val_mae = mean_val_mae
        best_k = k
    
    print(f"k={k:2d} → train MAE={mean_train_mae:.4f}, val MAE={mean_val_mae:.4f}")

print(f"\nBest k by validation MAE: {best_k} (MAE={best_val_mae:.4f})")

# === 4) PLOT & SAVE ===
plt.figure(figsize=(8,5))
plt.plot(ks, train_maes, marker='o', label='Train MAE')
plt.plot(ks, val_maes,   marker='o', label='Val MAE')
plt.xlabel("Number of Features (k)")
plt.ylabel("Mean Absolute Error")
plt.title("Train vs. Validation MAE by # of Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_val_mae_vs_k.png", dpi=150)
plt.show()

# === 5) FINAL TRAIN & TEST EVALUATION ===
best_feats = ordered_features[:best_k]
final_model = XGBRegressor(**xgb_kwargs)
final_model.fit(X_trainval[best_feats], y_trainval)

y_pred = final_model.predict(X_test[best_feats])
mse   = mean_squared_error(y_test, y_pred)
mae   = mean_absolute_error(y_test, y_pred)
rmse  = sqrt(mse)
r2    = r2_score(y_test, y_pred)

print("\nHeld‑out Test Set Performance:")
print(f"  MAE : {mae:.4f}")
print(f"  MSE : {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²  : {r2:.4f}")
