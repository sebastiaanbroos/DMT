import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# Helper: Temporal Imputation Function
# -----------------------------------------------------------------------------
def temporal_impute(df, target, method='time'):
    """
    Performs temporal imputation for each subject (grouped by 'id') in df.
    For each group, the function sorts by 'date', and if there are at least 2
    nonmissing values in target, it sets the index to the date and applies time 
    interpolation. Otherwise, it falls back on forward/backward fill.
    
    Returns a Series with imputed values (aligned with df.index).
    """
    result = pd.Series(index=df.index, dtype=float)
    for sid, group in df.groupby('id'):
        group_sorted = group.sort_values('date')
        if group_sorted[target].notna().sum() < 2:
            imputed = group_sorted[target].ffill().bfill()
        else:
            # For time interpolation, use the 'date' column as the index.
            ser = pd.Series(data=group_sorted[target].values, index=group_sorted['date'])
            imputed = ser.interpolate(method='time').ffill().bfill()
            # Align the imputed values with the original DataFrame index.
            imputed = pd.Series(imputed.values, index=group_sorted.index)
        result.loc[group_sorted.index] = imputed.values
    return result

# =============================================================================
# 1. LOAD THE DAILY AGGREGATED DATA
# =============================================================================
# The data file should include the following columns:
# id, date, circumplex.arousal_avg, circumplex.valence_avg, mood_avg,
# circumplex.arousal_std, circumplex.valence_std, mood_std, activity,
# appCat.builtin, appCat.communication, appCat.entertainment, appCat.finance,
# appCat.game, appCat.office, appCat.other, appCat.social, appCat.travel,
# appCat.unknown, appCat.utilities, appCat.weather, call, screen, sms

data_path = "/Users/s.broos/Documents/DMT/data/daily_aggregated_completemoods.csv"
df_daily = pd.read_csv(data_path, parse_dates=["date"])
print("Loaded daily aggregated data with shape:", df_daily.shape)
df_daily = df_daily.sort_values("date").reset_index(drop=True)

# Candidate variables: all columns except "id" and "date"
impute_cols = [col for col in df_daily.columns if col not in ["id", "date"]]

# -----------------------------------------------------------------------------
# Normalize all candidate columns (z-score normalization)
# Note: normalization is applied over non-missing values.
# -----------------------------------------------------------------------------
df_daily[impute_cols] = df_daily[impute_cols].apply(lambda col: (col - col.mean()) / col.std())

# Dictionary to store evaluation results
evaluation_results = {}

# =============================================================================
# 2. LOOP OVER EACH VARIABLE TO EVALUATE ITS IMPUTATION QUALITY
# =============================================================================
for target in impute_cols:
    # Use only rows where target is observed (for ground truth evaluation)
    df_target = df_daily[df_daily[target].notna()].copy()
    if len(df_target) < 30:
        print(f"Skipping {target} due to insufficient observed data.")
        continue

    print(f"\n===== Evaluating Target: {target} =====")
    
    # ------------------------------
    # 2A. TRAIN/TEST SPLIT (time-based)
    # ------------------------------
    split_idx = int(0.8 * len(df_target))
    df_train = df_target.iloc[:split_idx].copy()
    df_test = df_target.iloc[split_idx:].copy()
    print(f"  Train rows: {len(df_train)} | Test rows: {len(df_test)}")
    
    # Save true values before simulating missingness
    df_train[target + "_true"] = df_train[target]
    df_test[target + "_true"] = df_test[target]
    df_test[target] = np.nan  # simulate missingness in test set
    
    # ------------------------------
    # 2B. IMPUTATION METHODS
    # ------------------------------
    # Concatenate train and test sets and sort by "id" and "date"
    df_combo = pd.concat([df_train, df_test]).sort_values(["id", "date"])
    
    # (i) Time Interpolation (uses date/time)
    time_col = target + "_time"
    df_combo[time_col] = temporal_impute(df_combo, target, method='time')
    df_combo[time_col] = df_combo[time_col].fillna(df_train[target].mean())
    df_test_temp = df_combo.loc[df_test.index]
    true_vals = df_test_temp[target + "_true"]
    pred_vals_time = df_test_temp[time_col]
    mae_time = mean_absolute_error(true_vals, pred_vals_time)
    rmse_time = np.sqrt(mean_squared_error(true_vals, pred_vals_time))
    print(f"  Time Interpolation: MAE={mae_time:.4f}, RMSE={rmse_time:.4f}")
    
    # (ii) Forward/Backward Fill (does not use time)
    ffill_col = target + "_ffill"
    df_combo[ffill_col] = df_combo.groupby("id")[target].transform(lambda grp: grp.ffill().bfill())
    df_combo[ffill_col] = df_combo[ffill_col].fillna(df_train[target].mean())
    df_test_temp = df_combo.loc[df_test.index]
    pred_vals_ffill = df_test_temp[ffill_col]
    mae_ffill = mean_absolute_error(true_vals, pred_vals_ffill)
    rmse_ffill = np.sqrt(mean_squared_error(true_vals, pred_vals_ffill))
    print(f"  Forward/Backward Fill: MAE={mae_ffill:.4f}, RMSE={rmse_ffill:.4f}")
    
    # (iii) Group Mean Imputation (does not use time)
    mean_impute_col = target + "_group_mean"
    df_combo[mean_impute_col] = df_combo.groupby("id")[target].transform(lambda grp: grp.fillna(grp.mean()))
    df_combo[mean_impute_col] = df_combo[mean_impute_col].fillna(df_train[target].mean())
    df_test_temp = df_combo.loc[df_test.index]
    pred_vals_mean = df_test_temp[mean_impute_col]
    mae_mean = mean_absolute_error(true_vals, pred_vals_mean)
    rmse_mean = np.sqrt(mean_squared_error(true_vals, pred_vals_mean))
    print(f"  Group Mean Imputation: MAE={mae_mean:.4f}, RMSE={rmse_mean:.4f}")
    
    # (iv) Group Median Imputation (does not use time)
    median_impute_col = target + "_group_median"
    df_combo[median_impute_col] = df_combo.groupby("id")[target].transform(lambda grp: grp.fillna(grp.median()))
    df_combo[median_impute_col] = df_combo[median_impute_col].fillna(df_train[target].median())
    df_test_temp = df_combo.loc[df_test.index]
    pred_vals_median = df_test_temp[median_impute_col]
    mae_median = mean_absolute_error(true_vals, pred_vals_median)
    rmse_median = np.sqrt(mean_squared_error(true_vals, pred_vals_median))
    print(f"  Group Median Imputation: MAE={mae_median:.4f}, RMSE={rmse_median:.4f}")
    
    # ------------- Advanced Model-Based Methods -------------
    # For these methods, we will use the remaining candidate columns (other than the target)
    # as predictors to model the missing target.
    features = [col for col in impute_cols if col != target]
    
    # Prepare training data: use rows from df_train; ensure predictors have no missing values
    X_train = df_train[features].copy()
    y_train = df_train[target]
    X_test = df_test[features].copy()
    
    # Fill potential missing values in predictors with the training set mean for each predictor.
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    
    # (v) KNN Regression Imputation
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    pred_vals_knn = knn_model.predict(X_test)
    mae_knn = mean_absolute_error(true_vals, pred_vals_knn)
    rmse_knn = np.sqrt(mean_squared_error(true_vals, pred_vals_knn))
    print(f"  KNN Regression Imputation: MAE={mae_knn:.4f}, RMSE={rmse_knn:.4f}")
    
    # (vi) Random Forest Regression Imputation
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    pred_vals_rf = rf_model.predict(X_test)
    mae_rf = mean_absolute_error(true_vals, pred_vals_rf)
    rmse_rf = np.sqrt(mean_squared_error(true_vals, pred_vals_rf))
    print(f"  Random Forest Regression Imputation: MAE={mae_rf:.4f}, RMSE={rmse_rf:.4f}")
    
    # ------------------------------
    # 2C. STORE THE RESULTS FOR THIS TARGET
    # ------------------------------
    evaluation_results[target] = {
        "time_interpolation": {"MAE": mae_time, "RMSE": rmse_time},
        "ffill": {"MAE": mae_ffill, "RMSE": rmse_ffill},
        "group_mean": {"MAE": mae_mean, "RMSE": rmse_mean},
        "group_median": {"MAE": mae_median, "RMSE": rmse_median},
        "knn_regression": {"MAE": mae_knn, "RMSE": rmse_knn},
        "random_forest": {"MAE": mae_rf, "RMSE": rmse_rf}
    }

# =============================================================================
# 3. PRINT AGGREGATED RESULTS
# =============================================================================
print("\n====== AGGREGATED RESULTS ======")
for targ, res in evaluation_results.items():
    print(f"\nTarget: {targ}")
    #print(f"  Time Interpolation: MAE={res['time_interpolation']['MAE']:.4f}, RMSE={res['time_interpolation']['RMSE']:.4f}")
    print(f"  Forward/Backward Fill: MAE={res['ffill']['MAE']:.4f}, RMSE={res['ffill']['RMSE']:.4f}")
    print(f"  Group Mean Imputation: MAE={res['group_mean']['MAE']:.4f}, RMSE={res['group_mean']['RMSE']:.4f}")
    print(f"  Group Median Imputation: MAE={res['group_median']['MAE']:.4f}, RMSE={res['group_median']['RMSE']:.4f}")
    print(f"  KNN Regression Imputation: MAE={res['knn_regression']['MAE']:.4f}, RMSE={res['knn_regression']['RMSE']:.4f}")
    print(f"  Random Forest Regression Imputation: MAE={res['random_forest']['MAE']:.4f}, RMSE={res['random_forest']['RMSE']:.4f}")

# =============================================================================
# 4. RETURN BEST METHOD PER VARIABLE (based on lowest MAE)
# =============================================================================
best_methods = {}

for target, results in evaluation_results.items():
    best_method = min(results.items(), key=lambda x: x[1]['RMSE'])[0]
    best_rmse = results[best_method]['RMSE']
    best_methods[target] = {
        "best_method": best_method,
        "RMSE": best_rmse,
        "MAE": results[best_method]['MAE']
    }

# Print a summary
print("\n====== BEST METHOD PER VARIABLE (based on RMSE) ======")
for var, info in best_methods.items():
    print(f"{var}: {info['best_method']} (MAE={info['MAE']:.4f}, RMSE={info['RMSE']:.4f})")
