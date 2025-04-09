import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed

# --------------------------
# 1. CUSTOM KNN PREDICTION
# --------------------------
def custom_knn_predict(missing_row, X_train, y_train, predictors, k=5):
    """
    For 'missing_row' (Series), use only the predictor columns that are non-missing
    in that row. Then filter X_train to rows that also have no missing in those columns,
    compute Euclidean distances, and return the mean target of the k nearest neighbors.
    """
    # 1) Which predictors are non-NaN in the missing row?
    available_predictors = missing_row[predictors].dropna().index.tolist()
    if not available_predictors:
        return np.nan
    
    # 2) Filter train set to rows that also have no NaN in those columns
    X_train_sub = X_train[available_predictors].dropna()
    y_train_sub = y_train.loc[X_train_sub.index]
    if X_train_sub.empty:
        return np.nan
    
    # 3) One row of data to compare
    x_missing = missing_row[available_predictors]  # Series of floats
    
    # 4) Euclidean distances
    #    - diffs: DataFrame where each row is (X_train_sub_row - x_missing)
    diffs = X_train_sub - x_missing
    #    - sum of squared diffs across columns => then sqrt
    distances = np.sqrt((diffs**2).sum(axis=1))  # Must be numeric -> fix by ensuring numeric
    if distances.empty:
        return np.nan
    
    # 5) Get mean of k nearest neighbors
    k = min(k, len(distances))
    nearest_indices = distances.nsmallest(k).index
    return y_train_sub.loc[nearest_indices].mean()


# --------------------------
# 2. PARALLEL KNN EVALUATION
# --------------------------
def evaluate_knn_for_k(k, df_train, df_test, predictors, target):
    """
    For a given k, fill the target in df_test using custom_knn_predict. 
    Return a tuple (method_name, metrics_dict).
    """
    X_train = df_train[predictors]
    y_train = df_train[target + '_true']
    
    # Debug check: confirm we are numeric
    # (Uncomment if you need to confirm everything is float)
    # print(f"[DEBUG] KNN for k={k} => X_train dtypes:\n", X_train.dtypes)
    
    knn_preds = {}
    for idx, row in df_test.iterrows():
        pred = custom_knn_predict(row, X_train, y_train, predictors, k=k)
        knn_preds[idx] = pred
    
    # Evaluate
    true_vals = df_test[target + '_true']
    pred_vals = pd.Series(knn_preds)
    mask = (~true_vals.isna()) & (~pred_vals.isna())
    
    if not mask.any():
        knn_mae = np.nan
        knn_rmse = np.nan
    else:
        knn_mae = mean_absolute_error(true_vals[mask], pred_vals[mask])
        knn_rmse = np.sqrt(mean_squared_error(true_vals[mask], pred_vals[mask]))
    
    return (f"knn_k_{k}", {"MAE": knn_mae, "RMSE": knn_rmse})


# --------------------------
# 3. GLOBAL PARAMS
# --------------------------
spline_orders = [2, 3, 4]
lasso_alphas = [0.1, 1, 10]
knn_k_values = [3, 5, 7]

expected_variables = [
    'mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 
    'screen', 'call', 'sms', 'appCat.builtin', 'appCat.communication',
    'appCat.entertainment', 'appCat.finance', 'appCat.game', 
    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
    'appCat.unknown', 'appCat.utilities', 'appCat.weather'
]


# --------------------------
# 4. LOAD + FILTER DATA
# --------------------------
data_path = "/Users/s.broos/Documents/DMT/data/daily_aggregated_imputed.csv"
df_daily = pd.read_csv(data_path, parse_dates=["date"])
print("Initial shape:", df_daily.shape)

# Filter out rows missing too many variables
max_missing = 50
expected_cols = [c for c in expected_variables if c in df_daily.columns]
thresh = len(expected_cols) - max_missing
df_daily = df_daily.dropna(subset=expected_cols, thresh=thresh)
print("Shape after dropping rows with > 5 missing among expected cols:", df_daily.shape)

# Sort by date, do time-based split
df_daily = df_daily.sort_values('date').reset_index(drop=True)
split_idx = int(0.8 * len(df_daily))
df_train_full = df_daily.iloc[:split_idx].copy()
df_test_full  = df_daily.iloc[split_idx:].copy()

print("Train shape:", df_train_full.shape, "Test shape:", df_test_full.shape)


# --------------------------
# 5. EVALUATION LOOP
# --------------------------
evaluation_results = {}

for target in expected_cols:
    print(f"\n===== EVALUATING TARGET: {target} =====")
    
    # Subset to rows that have the target present
    df_train = df_train_full.dropna(subset=[target]).copy()
    df_test  = df_test_full.dropna(subset=[target]).copy()
    if df_train.empty or df_test.empty:
        print(f"Skipping {target} -> no train or test data.")
        continue
    
    print(f"  Train rows = {len(df_train)} | Test rows = {len(df_test)}")
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # STEP 5A: SELECT PREDICTORS + ENFORCE NUMERIC
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    predictors = [col for col in expected_cols if col != target and col in df_train.columns]
    
    # Convert these predictor columns to numeric, forcing invalid to NaN
    df_train[predictors] = df_train[predictors].apply(pd.to_numeric, errors='coerce')
    df_test[predictors]  = df_test[predictors].apply(pd.to_numeric, errors='coerce')
    
    # Fill in NaN with median from the train set
    train_medians = df_train[predictors].median()
    df_train[predictors] = df_train[predictors].fillna(train_medians)
    df_test[predictors]  = df_test[predictors].fillna(train_medians)
    
    # Extra debug: ensure everything is numeric
    # (Optional to show; comment out if too verbose)
    # print("[DEBUG] df_train dtypes after to_numeric + fill:\n", df_train[predictors].dtypes)
    # print("[DEBUG] df_test dtypes after to_numeric + fill:\n", df_test[predictors].dtypes)
    
    # Store the 'true' target
    df_train[target + '_true'] = df_train[target]
    df_test[target + '_true']  = df_test[target]
    
    # Make test target = NaN so we can "impute"
    df_test[target] = np.nan
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # STEP 5B: TIME INTERPOLATION
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df_combo = pd.concat([df_train, df_test]).sort_values(['id', 'date'])
    time_col = target + '_time'
    df_combo[time_col] = df_combo[target]  # Copy current target
    
    for sid in df_combo['id'].unique():
        sub_mask = (df_combo['id'] == sid)
        sub_df = df_combo.loc[sub_mask].sort_values('date')
        
        if sub_df[target].notna().sum() < 2:
            continue  # Cannot do time interpolation with <2 real points
        
        sub_df = sub_df.set_index('date')
        sub_df[time_col] = sub_df[time_col].interpolate(method='time')
        sub_df = sub_df.reset_index()
        df_combo.loc[sub_df.index, time_col] = sub_df[time_col]
    
    # Evaluate on test portion
    df_test_imputed = df_combo.loc[df_combo.index.intersection(df_test.index)]
    true_vals = df_test_imputed[target + '_true']
    pred_vals = df_test_imputed[time_col]
    mask = (~true_vals.isna()) & (~pred_vals.isna())
    if not mask.any():
        mae_time = np.nan
        rmse_time = np.nan
        print(f"  Time interpolation => No valid test rows to compare.")
    else:
        mae_time = mean_absolute_error(true_vals[mask], pred_vals[mask])
        rmse_time = np.sqrt(mean_squared_error(true_vals[mask], pred_vals[mask]))
    
    print(f"  Time Interpolation: MAE={mae_time}, RMSE={rmse_time}")
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # STEP 5C: SPLINE INTERPOLATION
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    spline_dict = {}
    for order in spline_orders:
        df_combo = pd.concat([df_train, df_test]).sort_values(['id', 'date'])
        spline_col = f"{target}_spline_{order}"
        df_combo[spline_col] = df_combo[target]
        
        for sid in df_combo['id'].unique():
            sub_mask = (df_combo['id'] == sid)
            sub_df = df_combo.loc[sub_mask].sort_values('date')
            
            n_pts = sub_df[target].notna().sum()
            if n_pts < (order + 1):
                # fallback => linear
                sub_df = sub_df.set_index('date')
                sub_df[spline_col] = sub_df[spline_col].interpolate(method='linear')
                sub_df = sub_df.reset_index()
            else:
                sub_df = sub_df.set_index('date')
                sub_df[spline_col] = sub_df[spline_col].interpolate(method='spline', order=order)
                sub_df = sub_df.reset_index()
            
            df_combo.loc[sub_df.index, spline_col] = sub_df[spline_col]
        
        df_test_imputed = df_combo.loc[df_combo.index.intersection(df_test.index)]
        true_vals = df_test_imputed[target + '_true']
        pred_vals = df_test_imputed[spline_col]
        
        mask = (~true_vals.isna()) & (~pred_vals.isna())
        if not mask.any():
            mae_spl = np.nan
            rmse_spl = np.nan
        else:
            mae_spl = mean_absolute_error(true_vals[mask], pred_vals[mask])
            rmse_spl = np.sqrt(mean_squared_error(true_vals[mask], pred_vals[mask]))
        spline_dict[f"order_{order}"] = {"MAE": mae_spl, "RMSE": rmse_spl}
        print(f"  Spline(order={order}): MAE={mae_spl}, RMSE={rmse_spl}")
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # STEP 5D: LINEAR REGRESSION
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X_train = df_train[predictors]
    X_test  = df_test[predictors]
    y_train = df_train[target + '_true']
    y_test  = df_test[target + '_true']
    
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred_lr = linreg.predict(X_test)
    
    mask = (~y_test.isna()) & (~pd.isna(y_pred_lr))
    if not mask.any():
        lr_mae = np.nan
        lr_rmse = np.nan
    else:
        lr_mae = mean_absolute_error(y_test[mask], y_pred_lr[mask])
        lr_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred_lr[mask]))
    print(f"  LinearRegression: MAE={lr_mae}, RMSE={lr_rmse}")
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # STEP 5E: LASSO
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    lasso_dict = {}
    for alpha in lasso_alphas:
        lassomodel = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        lassomodel.fit(X_train, y_train)
        y_pred_lasso = lassomodel.predict(X_test)
        
        mask = (~y_test.isna()) & (~pd.isna(y_pred_lasso))
        if not mask.any():
            lasso_mae = np.nan
            lasso_rmse = np.nan
        else:
            lasso_mae = mean_absolute_error(y_test[mask], y_pred_lasso[mask])
            lasso_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred_lasso[mask]))
        lasso_dict[f"alpha_{alpha}"] = {"MAE": lasso_mae, "RMSE": lasso_rmse}
        print(f"  Lasso(alpha={alpha}): MAE={lasso_mae}, RMSE={lasso_rmse}")
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # STEP 5F: KNN in PARALLEL
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We'll pass df_train, df_test, the predictor list, etc. 
    knn_eval_list = Parallel(n_jobs=-1)(
        delayed(evaluate_knn_for_k)(
            k=k,
            df_train=df_train,
            df_test=df_test,
            predictors=predictors,
            target=target
        )
        for k in knn_k_values
    )
    knn_dict = {key: val for (key, val) in knn_eval_list}
    for kkey, vals in knn_dict.items():
        print(f"  KNN({kkey}): MAE={vals['MAE']}, RMSE={vals['RMSE']}")
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # STEP 5G: STORE RESULTS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    evaluation_results[target] = {
        "time_interpolation": {"MAE": mae_time, "RMSE": rmse_time},
        "spline": spline_dict,
        "regression": {"MAE": lr_mae, "RMSE": lr_rmse},
        "lasso": lasso_dict,
        "knn": knn_dict
    }


# --------------------------
# 6. PRINT FINAL RESULTS
# --------------------------
print("\n========= AGGREGATED RESULTS =========")
for var, methods in evaluation_results.items():
    print(f"\nTARGET: {var}")
    for method_name, metrics in methods.items():
        if isinstance(metrics, dict) and "MAE" in metrics:
            # Single dict with {MAE, RMSE}
            print(f"  {method_name}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}")
        else:
            # Potentially nested dict (e.g., spline orders, lasso alphas, KNN k)
            print(f"  {method_name}:")
            for sub_key, sub_val in metrics.items():
                print(f"    {sub_key}: MAE={sub_val['MAE']}, RMSE={sub_val['RMSE']}")
