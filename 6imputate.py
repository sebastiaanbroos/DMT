#!/usr/bin/env python3
import pandas as pd
import numpy as np
# Machine learning models
from sklearn.ensemble import RandomForestRegressor
# ----------------------------- PyTorch Imports -----------------------------
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------------------------------------------------
# Helper 1: Temporal Imputation Function
# -----------------------------------------------------------------------------
def temporal_impute(df, target):
    """
    Performs temporal imputation for each subject (grouped by 'id') in df.
    If at least 2 non-missing values exist, uses time interpolation;
    otherwise, falls back on forward/backward fill.
    Returns a Series with imputed values.
    """
    if "id" not in df.columns:
        return df[target].ffill().bfill().fillna(df[target].mean())
    result = pd.Series(index=df.index, dtype=float)
    for sid, group in df.groupby("id"):
        group_sorted = group.sort_values("date")
        if group_sorted[target].notna().sum() < 2:
            imputed = group_sorted[target].ffill().bfill()
        else:
            ser = pd.Series(data=group_sorted[target].values, index=group_sorted["date"])
            imputed = ser.interpolate(method="time").ffill().bfill()
            imputed = pd.Series(imputed.values, index=group_sorted.index)
        result.loc[group_sorted.index] = imputed.values
    return result

# -----------------------------------------------------------------------------
# Helper 2: Group Mean Imputation
# -----------------------------------------------------------------------------
def impute_group_mean(df, target):
    if "id" not in df.columns:
        return df[target].fillna(df[target].mean())
    imputed = df.groupby("id")[target].transform(lambda grp: grp.fillna(grp.mean()))
    imputed = imputed.fillna(df[target].mean())
    return imputed

# -----------------------------------------------------------------------------
# Helper 3: Group Median Imputation
# -----------------------------------------------------------------------------
def impute_group_median(df, target):
    if "id" not in df.columns:
        return df[target].fillna(df[target].median())
    imputed = df.groupby("id")[target].transform(lambda grp: grp.fillna(grp.median()))
    imputed = imputed.fillna(df[target].median())
    return imputed

# -----------------------------------------------------------------------------
# Helper 4: Random Forest Regression Imputation
# -----------------------------------------------------------------------------
def impute_random_forest(df, target, features):
    observed = df[df[target].notna()]
    missing = df[df[target].isna()]
    if missing.empty:
         return df[target]
    X_train = observed[features].copy().fillna(observed[features].mean())
    y_train = observed[target]
    X_missing = missing[features].copy().fillna(observed[features].mean())
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    predicted = rf.predict(X_missing)
    imputed = df[target].copy()
    imputed.loc[missing.index] = predicted
    return imputed

# -----------------------------------------------------------------------------
# Helper 5: Load Saved NN Model & Specific Value Imputation
# -----------------------------------------------------------------------------
def load_nn_model(model_path, original_dim, hidden_dim1=128, hidden_dim2=64):
    """
    Loads the saved NN model into an instance of the ImputationNN architecture.
    """
    class ImputationNN(nn.Module):
        def __init__(self, original_dim, hidden_dim1, hidden_dim2):
            super(ImputationNN, self).__init__()
            input_dim = original_dim * 2
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim1),
                nn.ReLU(),
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.ReLU(),
                nn.Linear(hidden_dim2, original_dim)
            )
        def forward(self, x):
            return self.net(x)
    model = ImputationNN(original_dim, hidden_dim1, hidden_dim2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def impute_specific_value(sample, missing_idx, model):
    """
    Imputes the value for one missing variable (at missing_idx) in a given sample.
    'sample' is a 1D numpy array (normalized) of length original_dim.
    Returns a copy of the sample with the missing value replaced by the model's prediction.
    """
    mask = np.ones_like(sample, dtype=np.float32)
    mask[missing_idx] = 0.0
    sample_corrupted = sample.copy()
    sample_corrupted[missing_idx] = 0.0
    input_vec = np.concatenate([sample_corrupted, mask])
    input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).numpy()
    imputed_sample = sample.copy()
    imputed_sample[missing_idx] = output[missing_idx]
    return imputed_sample

# -----------------------------------------------------------------------------
# BEST METHOD MAPPING
# -----------------------------------------------------------------------------
best_method_mapping = {
    "circumplex.arousal_avg": "group_mean",
    "circumplex.valence_avg": "random_forest",
    "mood_avg": "random_forest",
    "circumplex.arousal_std": "nn",              # NN wins (0.9596 vs 0.9769)
    "circumplex.valence_std": "random_forest",
    "mood_std": "random_forest",
    "activity": "group_median",
    "appCat.builtin": "random_forest",
    "appCat.communication": "random_forest",
    "appCat.entertainment": "group_mean",
    "appCat.finance": "nn",                      # NN wins (0.9297 vs 1.3637)
    "appCat.game": "group_mean",
    "appCat.office": "group_mean",
    "appCat.other": "group_median",
    "appCat.social": "group_mean",
    "appCat.travel": "group_median",
    "appCat.unknown": "group_mean",
    "appCat.utilities": "group_mean",
    "appCat.weather": "group_mean",
    "call": "nn",                                # NN wins (0.9392 vs 1.0298)
    "screen": "random_forest",
    "sms": "nn"                                  # NN wins (0.9461 vs 0.9935)
}

# -----------------------------------------------------------------------------
# 1. LOAD & PREPROCESS THE DATA
# -----------------------------------------------------------------------------
data_path = "/Users/s.broos/Documents/DMT/data/daily_removed_incomplete_moods_non-imputated.csv"
df = pd.read_csv(data_path, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Candidate variables: all columns except "date" and "id"
impute_cols = [col for col in df.columns if col not in ["date", "id"]]

# Save original missing mask for each candidate variable.
missing_mask = {col: df[col].isna() for col in impute_cols}

# Create a working copy for preprocessing.
df_data = df[impute_cols].copy()
for col in impute_cols:
    df_data[col] = df_data[col].fillna(df_data[col].mean())
    df_data[col] = (df_data[col] - df_data[col].mean()) / df_data[col].std()

# For imputation functions, restore NaNs on originally missing entries.
df_for_impute = df_data.copy()
for col in impute_cols:
    df_for_impute.loc[missing_mask[col], col] = np.nan

# IMPORTANT: Add back the "id" column to df_for_impute for group-based methods.
if "id" in df.columns:
    df_for_impute["id"] = df["id"]

# Compute original_dim (number of candidate variables).
data_array = df_data.values.astype(np.float32)
original_dim = data_array.shape[1]

# -----------------------------------------------------------------------------
# 2. IMPUTE MISSING VALUES USING THE BEST METHOD FOR EACH VARIABLE
# -----------------------------------------------------------------------------
# Load the saved NN model once (for variables using the "nn" method).
model_path = "/Users/s.broos/Documents/DMT/imputation_nn_model.pt"
nn_model = load_nn_model(model_path, original_dim, hidden_dim1=128, hidden_dim2=64)

# Initialize a DataFrame to record imputed values.
imputation_record = pd.DataFrame(np.nan, index=df.index, columns=impute_cols)

# For each candidate variable, apply the designated imputation method
# and replace only those values that were originally missing.
final_imputed = {}
for col in impute_cols:
    method = best_method_mapping.get(col, "group_mean")
    if method == "time_interpolation":
        imputed_series = temporal_impute(df_for_impute, col)
    elif method == "group_mean":
        imputed_series = impute_group_mean(df_for_impute, col)
    elif method == "group_median":
        imputed_series = impute_group_median(df_for_impute, col)
    elif method == "random_forest":
        features = [c for c in impute_cols if c != col]
        imputed_series = impute_random_forest(df_for_impute, col, features)
    elif method == "nn":
        missing_col_idx = impute_cols.index(col)
        imputed_vals = []
        for idx, row in df_data.iterrows():
            sample = row.values.copy()  # normalized sample
            if not missing_mask[col].iloc[idx]:
                imputed_vals.append(sample[missing_col_idx])
            else:
                imputed_sample = impute_specific_value(sample, missing_col_idx, nn_model)
                imputed_vals.append(imputed_sample[missing_col_idx])
        imputed_series = pd.Series(imputed_vals, index=df.index)
    else:
        imputed_series = df_for_impute.groupby("id")[col].transform(lambda grp: grp.ffill().bfill())
        imputed_series = imputed_series.fillna(df_for_impute[col].mean())
    
    # Replace only originally missing values.
    final_col = df_data[col].copy()
    final_col.loc[missing_mask[col]] = imputed_series.loc[missing_mask[col]]
    final_col = final_col.fillna(df_data[col])
    final_imputed[col] = final_col
    df_data[col] = final_col
    # Use single-step assignment with .loc to avoid chained warnings.
    imputation_record.loc[missing_mask[col], col] = imputed_series.loc[missing_mask[col]]

# Add back the id and date columns to the imputation record.
imputation_record["id"] = df["id"]
imputation_record["date"] = df["date"]

# Optionally reorder so that id and date are the first columns.
cols = ["id", "date"] + [col for col in imputation_record.columns if col not in ["id", "date"]]
imputation_record = imputation_record[cols]

# Replace candidate columns in the original DataFrame with the imputed values.
for col in impute_cols:
    df[col] = df_data[col]

# Save the fully imputed DataFrame and the imputation record.
output_csv = "daily_removed_incomplete_moods_imputated.csv"
df.to_csv(output_csv, index=False)
record_csv = "imputation_record.csv"
imputation_record.to_csv(record_csv, index=False)
print(f"Imputation completed and saved to {output_csv}")
print(f"Imputation record saved to {record_csv}")
