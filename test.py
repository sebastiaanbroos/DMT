import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --------------------------
# Custom KNN Prediction Function
# --------------------------
def custom_knn_predict(missing_row, X_train, y_train, predictors, k=5):
    """
    For a missing-row (a Series with predictor values, may contain NaNs),
    use only the predictors that are non-missing in that row. Then, among the training rows,
    select those that also have non-missing values in these predictors.
    Compute Euclidean distances (over the available predictors) and return the average
    'target' of the k nearest neighbors.
    """
    # Predictors that are non-missing in the missing_row
    available_predictors = missing_row[predictors].dropna().index.tolist()
    if not available_predictors:
        return np.nan
    
    # Subset training data to rows that are non-missing in those predictors
    X_train_sub = X_train[available_predictors].dropna()
    y_train_sub = y_train.loc[X_train_sub.index]
    
    if len(X_train_sub) == 0:
        return np.nan
    
    # Compute Euclidean distances
    diffs = X_train_sub - missing_row[available_predictors]
    distances = np.sqrt((diffs**2).sum(axis=1))
    
    # k nearest neighbors
    k_eff = min(k, len(distances))
    nearest_indices = distances.nsmallest(k_eff).index
    return y_train_sub.loc[nearest_indices].mean()

# --------------------------
# Step 1: Load and Prepare the Data
# --------------------------
data = pd.read_csv("./data/dataset_mood_smartphone.csv", parse_dates=["time"])
data['value_numeric'] = pd.to_numeric(data['value'], errors='coerce')

# Define the columns to evaluate
expected_variables = [
    'mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 
    'screen', 'call', 'sms', 'appCat.builtin', 'appCat.communication',
    'appCat.entertainment', 'appCat.finance', 'appCat.game', 
    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
    'appCat.unknown', 'appCat.utilities', 'appCat.weather'
]

# --------------------------
# All Subject IDs
# --------------------------
subject_ids = data['id'].unique()
print(f"Total number of subjects: {len(subject_ids)}")

# This dict will store the final results:
# accuracy_results[subject][variable] = { 'interp_mae': ..., 'interp_rmse': ..., etc. }
accuracy_results = {}

# --------------------------
# Step 2: Process Each Subject, Using Only Complete Rows
# --------------------------
for subject in subject_ids:
    # Filter data for this subject
    df_sub = data[data['id'] == subject].copy()
    
    # Pivot to wide format
    df_wide = df_sub.pivot_table(index='time', columns='variable', values='value_numeric')
    df_wide.sort_index(inplace=True)
    
    # Keep only the expected vars that exist in this subject
    vars_available = [v for v in expected_variables if v in df_wide.columns]
    
    # Step 2A: Restrict to rows that are COMPLETE for all these variables
    # i.e., no missing data across the relevant columns
    df_complete = df_wide[vars_available].dropna(how='any', subset=vars_available)
    
    # If there's no complete data, skip
    if df_complete.shape[0] == 0:
        continue
    
    # Initialize results storage for this subject
    accuracy_results[subject] = {}
    
    # --------------------------
    # Step 3: For each variable, artificially remove ~10% from these COMPLETE data
    #         Then apply interpolation, regression, KNN on that subset.
    # --------------------------
    for target_var in vars_available:
        # We'll measure how well we can recover the removed values for `target_var`.
        df_var = df_complete.copy()
        
        # We'll keep a "true" column for the original target variable
        df_var[f'{target_var}_true'] = df_var[target_var]
        
        # Artificially remove 10% of this variable's values (random)
        np.random.seed(42)
        missing_fraction = 0.1
        missing_mask = np.random.rand(len(df_var)) < missing_fraction
        df_var.loc[missing_mask, target_var] = np.nan
        
        # If no values are removed, skip
        if missing_mask.sum() == 0:
            continue
        
        # --------- Time Interpolation ---------
        # Because df_var is time-indexed, method='time' can work
        # (So long as the index is a properly-sorted DateTimeIndex)
        df_var[f'{target_var}_interp'] = df_var[target_var].interpolate(method='time')
        
        # --------- Regression Imputation ---------
        df_var[f'{target_var}_reg'] = df_var[target_var].copy()
        
        # Split into training (non-missing) and missing subsets
        df_train = df_var[df_var[target_var].notna()].copy()
        df_missing = df_var[df_var[target_var].isna()].copy()
        
        # Exclude the target variable from the predictor list
        predictors = [v for v in vars_available if v != target_var]
        
        # If no predictors, skip
        if len(predictors) == 0:
            # No possible regression -> fill in "interp" or do nothing
            pass
        else:
            # Prepare X_train, y_train
            X_train = df_train[predictors].fillna(df_train[predictors].mean())
            y_train = df_train[target_var]
            
            # Prepare X_missing
            X_missing = df_missing[predictors].fillna(df_train[predictors].mean())
            
            # Fit the model if we have at least 1 row and 1 column
            if X_train.shape[0] > 0 and X_train.shape[1] > 0 and len(X_missing) > 0:
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred_reg = lr.predict(X_missing)
                df_var.loc[df_missing.index, f'{target_var}_reg'] = y_pred_reg
        
        # --------- KNN Imputation ---------
        df_var[f'{target_var}_knn'] = df_var[target_var].copy()
        
        df_train_knn = df_var[df_var[target_var].notna()].copy()
        df_missing_knn = df_var[df_var[target_var].isna()].copy()
        
        if len(df_train_knn) > 0 and len(df_missing_knn) > 0 and len(predictors) > 0:
            for idx, row in df_missing_knn.iterrows():
                pred = custom_knn_predict(
                    missing_row=row, 
                    X_train=df_train_knn, 
                    y_train=df_train_knn[target_var], 
                    predictors=predictors,
                    k=3
                )
                df_var.loc[idx, f'{target_var}_knn'] = pred
        
        # ---------------------------
        # Step 4: Evaluate Accuracy (MAE and RMSE)
        #   Only on the artificially removed rows
        # ---------------------------
        removed_indices = df_var.index[missing_mask]
        
        # Time interpolation
        interp_errors = df_var.loc[removed_indices, f'{target_var}_interp'] - df_var.loc[removed_indices, f'{target_var}_true']
        interp_mae = np.mean(np.abs(interp_errors))
        interp_rmse = np.sqrt(np.mean(interp_errors**2))
        
        # Regression
        reg_errors = df_var.loc[removed_indices, f'{target_var}_reg'] - df_var.loc[removed_indices, f'{target_var}_true']
        reg_mae = np.mean(np.abs(reg_errors))
        reg_rmse = np.sqrt(np.mean(reg_errors**2))
        
        # KNN
        knn_errors = df_var.loc[removed_indices, f'{target_var}_knn'] - df_var.loc[removed_indices, f'{target_var}_true']
        knn_mae = np.mean(np.abs(knn_errors))
        knn_rmse = np.sqrt(np.mean(knn_errors**2))
        
        # Store results
        accuracy_results[subject][target_var] = {
            'interp_mae': interp_mae,
            'interp_rmse': interp_rmse,
            'regression_mae': reg_mae,
            'regression_rmse': reg_rmse,
            'knn_mae': knn_mae,
            'knn_rmse': knn_rmse,
        }

# --------------------------
# Step 5: Print Results
# --------------------------
# Per-subject/variable
for subj in accuracy_results:
    print(f"\n=== Subject: {subj} ===")
    for var, metrics in accuracy_results[subj].items():
        print(f"Variable: {var}")
        print(f"  Time Interpolation - MAE: {metrics['interp_mae']:.3f}, RMSE: {metrics['interp_rmse']:.3f}")
        print(f"  Regression Imputation - MAE: {metrics['regression_mae']:.3f}, RMSE: {metrics['regression_rmse']:.3f}")
        print(f"  KNN Imputation - MAE: {metrics['knn_mae']:.3f}, RMSE: {metrics['knn_rmse']:.3f}")

# --------------------------
# Overall Summary
# --------------------------
all_interp_mae = []
all_interp_rmse = []
all_reg_mae = []
all_reg_rmse = []
all_knn_mae = []
all_knn_rmse = []

for subj in accuracy_results:
    for var, metrics in accuracy_results[subj].items():
        all_interp_mae.append(metrics['interp_mae'])
        all_interp_rmse.append(metrics['interp_rmse'])
        all_reg_mae.append(metrics['regression_mae'])
        all_reg_rmse.append(metrics['regression_rmse'])
        all_knn_mae.append(metrics['knn_mae'])
        all_knn_rmse.append(metrics['knn_rmse'])

def safe_mean(values):
    """Mean of values if not empty, else nan."""
    vals = [v for v in values if not pd.isna(v)]
    return np.mean(vals) if len(vals) > 0 else np.nan

overall_interp_mae = safe_mean(all_interp_mae)
overall_interp_rmse = safe_mean(all_interp_rmse)
overall_reg_mae = safe_mean(all_reg_mae)
overall_reg_rmse = safe_mean(all_reg_rmse)
overall_knn_mae = safe_mean(all_knn_mae)
overall_knn_rmse = safe_mean(all_knn_rmse)

print("\n=== Overall Accuracy Metrics (Complete-Case Rows Only) ===")
print(f"Time Interpolation:    MAE = {overall_interp_mae:.3f}, RMSE = {overall_interp_rmse:.3f}")
print(f"Regression Imputation: MAE = {overall_reg_mae:.3f}, RMSE = {overall_reg_rmse:.3f}")
print(f"KNN Imputation:        MAE = {overall_knn_mae:.3f}, RMSE = {overall_knn_rmse:.3f}")
