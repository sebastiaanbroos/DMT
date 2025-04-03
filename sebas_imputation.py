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
    mood of the k nearest neighbors.
    """
    # Identify available predictors in this missing row.
    available_predictors = missing_row[predictors].dropna().index.tolist()
    if not available_predictors:
        return np.nan  # if no predictors are available, return NaN.
    # Subset training data: only keep rows with non-missing values in all available predictors.
    X_train_sub = X_train[available_predictors].dropna()
    # Corresponding y values:
    y_train_sub = y_train[X_train_sub.index]
    # Get the missing row's vector for the available predictors.
    x_missing = missing_row[available_predictors]
    # Compute Euclidean distances.
    diffs = X_train_sub - x_missing
    distances = np.sqrt((diffs**2).sum(axis=1))
    if len(distances) == 0:
        return np.nan
    # Use k nearest neighbors.
    k = min(k, len(distances))
    nearest_indices = distances.nsmallest(k).index
    return y_train_sub.loc[nearest_indices].mean()

# --------------------------
# Step 1: Load and Prepare the Data
# --------------------------
data = pd.read_csv("./data/dataset_mood_smartphone.csv", parse_dates=["time"])
data['value_numeric'] = pd.to_numeric(data['value'], errors='coerce')

# --------------------------
# Step 2: Select About 10% of the Subjects
# --------------------------
subject_ids = data['id'].unique()
sample_size = max(1, len(subject_ids) // 10)
sampled_subjects = np.random.choice(subject_ids, size=sample_size, replace=False)
print(f"Selected subjects: {sampled_subjects}")

expected_variables = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 
                      'screen', 'call', 'sms', 'appCat.builtin', 'appCat.communication',
                      'appCat.entertainment', 'appCat.finance', 'appCat.game', 
                      'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
                      'appCat.unknown', 'appCat.utilities', 'appCat.weather']

accuracy_results = {}  # To store accuracy metrics for each subject

# --------------------------
# Step 3: Process Each Selected Subject
# --------------------------
for subject in sampled_subjects:
    # Filter data for this subject and pivot to wide format.
    df_sub = data[data['id'] == subject].copy()
    df_wide = df_sub.pivot_table(index='time', columns='variable', values='value_numeric')
    df_wide.sort_index(inplace=True)
    
    # Only keep expected variables that exist for this subject.
    available_vars = [var for var in expected_variables if var in df_wide.columns]
    df_wide = df_wide[available_vars]
    
    # Save the original mood values for accuracy evaluation.
    df_wide['mood_true'] = df_wide['mood'].copy()
    
    # Simulate missing mood data: randomly remove about 10% of mood observations.
    missing_fraction = 0.1
    np.random.seed(42)  # For reproducibility.
    missing_mask = np.random.rand(len(df_wide)) < missing_fraction
    df_wide.loc[missing_mask, 'mood'] = np.nan

    print(f"\nSubject: {subject}")
    print(df_wide['mood'].head(10))
    
    # --------------------------
    # Approach 1: Time-Based Interpolation for Mood
    # --------------------------
    df_interp = df_wide.copy()
    df_interp['mood_interp'] = df_interp['mood'].interpolate(method='time')
    
    # --------------------------
    # Approach 2: Regression-Based Imputation for Mood
    # --------------------------
    df_reg = df_wide.copy()
    df_train = df_reg[df_reg['mood'].notna()].copy()
    df_missing = df_reg[df_reg['mood'].isna()].copy()
    
    predictors = [v for v in available_vars if v != 'mood']
    # Drop predictors that are entirely missing in training.
    predictors = [v for v in predictors if df_train[v].notna().sum() > 0]
    
    # For regression, fill missing predictor values with the column mean (as baseline).
    X_train = df_train[predictors].fillna(df_train[predictors].mean())
    y_train = df_train['mood']
    X_missing = df_missing[predictors].fillna(df_train[predictors].mean())
    
    if len(X_train) > 0 and len(X_missing) > 0:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_reg = lr.predict(X_missing)
        df_reg.loc[df_reg['mood'].isna(), 'mood_reg'] = y_pred_reg
        df_reg['mood_reg'] = df_reg['mood_reg'].fillna(df_reg['mood'])
    else:
        df_reg['mood_reg'] = df_reg['mood']
    
    # --------------------------
    # Approach 3: Custom KNN-Based Imputation for Mood
    # --------------------------
    df_knn = df_wide.copy()
    df_train_knn = df_knn[df_knn['mood'].notna()].copy()
    df_missing_knn = df_knn[df_knn['mood'].isna()].copy()
    
    # For KNN-based imputation, use only the available predictor values.
    mood_knn_predictions = {}
    for idx, row in df_missing_knn.iterrows():
        pred = custom_knn_predict(row, df_train_knn, df_train_knn['mood'], predictors, k=3)
        mood_knn_predictions[idx] = pred
    # Update the dataframe with KNN predictions.
    for idx, val in mood_knn_predictions.items():
        df_knn.loc[idx, 'mood_knn'] = val
    df_knn['mood_knn'] = df_knn['mood_knn'].fillna(df_knn['mood'])
    
    # --------------------------
    # Calculate Accuracy Metrics for Imputation
    # --------------------------
    # Evaluate only on rows where mood was artificially set to NaN.
    if missing_mask.sum() > 0:
        interp_mae = np.mean(np.abs(df_interp.loc[missing_mask, 'mood_interp'] - df_wide.loc[missing_mask, 'mood_true']))
        interp_rmse = np.sqrt(np.mean((df_interp.loc[missing_mask, 'mood_interp'] - df_wide.loc[missing_mask, 'mood_true'])**2))
        
        regression_mae = np.mean(np.abs(df_reg.loc[missing_mask, 'mood_reg'] - df_wide.loc[missing_mask, 'mood_true']))
        regression_rmse = np.sqrt(np.mean((df_reg.loc[missing_mask, 'mood_reg'] - df_wide.loc[missing_mask, 'mood_true'])**2))
        
        knn_mae = np.mean(np.abs(df_knn.loc[missing_mask, 'mood_knn'] - df_wide.loc[missing_mask, 'mood_true']))
        knn_rmse = np.sqrt(np.mean((df_knn.loc[missing_mask, 'mood_knn'] - df_wide.loc[missing_mask, 'mood_true'])**2))
        
        print(f"\nAccuracy Metrics for Subject {subject}:")
        print(f"Time Interpolation: MAE = {interp_mae:.3f}, RMSE = {interp_rmse:.3f}")
        print(f"Regression Imputation: MAE = {regression_mae:.3f}, RMSE = {regression_rmse:.3f}")
        print(f"KNN Imputation: MAE = {knn_mae:.3f}, RMSE = {knn_rmse:.3f}")
        
        accuracy_results[subject] = {
            'interp_mae': interp_mae,
            'interp_rmse': interp_rmse,
            'regression_mae': regression_mae,
            'regression_rmse': regression_rmse,
            'knn_mae': knn_mae,
            'knn_rmse': knn_rmse
        }


if accuracy_results:
    overall_interp_mae = np.mean([res['interp_mae'] for res in accuracy_results.values()])
    overall_interp_rmse = np.mean([res['interp_rmse'] for res in accuracy_results.values()])
    overall_regression_mae = np.mean([res['regression_mae'] for res in accuracy_results.values()])
    overall_regression_rmse = np.mean([res['regression_rmse'] for res in accuracy_results.values()])
    overall_knn_mae = np.mean([res['knn_mae'] for res in accuracy_results.values()])
    overall_knn_rmse = np.mean([res['knn_rmse'] for res in accuracy_results.values()])
    
    print("\nOverall Accuracy Metrics Across Subjects:")
    print(f"Time Interpolation: MAE = {overall_interp_mae:.3f}, RMSE = {overall_interp_rmse:.3f}")
    print(f"Regression Imputation: MAE = {overall_regression_mae:.3f}, RMSE = {overall_regression_rmse:.3f}")
    print(f"KNN Imputation: MAE = {overall_knn_mae:.3f}, RMSE = {overall_knn_rmse:.3f}")
