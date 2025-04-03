import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --------------------------
# Step 1: Load the Data and Mark Weird Values as Missing
# --------------------------
data = pd.read_csv(".//data/dataset_mood_smartphone.csv", parse_dates=["time"])

# Convert 'value' to numeric
data['value_numeric'] = pd.to_numeric(data['value'], errors='coerce')

# Expected domains for each variable.
expected_domains = {
    "mood": (1, 10),
    "circumplex.arousal": (-2, 2),
    "circumplex.valence": (-2, 2),
    "activity": (0, 1),
    "screen": (0, np.inf),
    "call": (0, 1),    # should be 0 or 1
    "sms": (0, 1),     # should be 0 or 1
    "appCat.builtin": (0, np.inf),
    "appCat.communication": (0, np.inf),
    "appCat.entertainment": (0, np.inf),
    "appCat.finance": (0, np.inf),
    "appCat.game": (0, np.inf),
    "appCat.office": (0, np.inf),
    "appCat.other": (0, np.inf),
    "appCat.social": (0, np.inf),
    "appCat.travel": (0, np.inf),
    "appCat.unknown": (0, np.inf),
    "appCat.utilities": (0, np.inf),
    "appCat.weather": (0, np.inf)
}

def mark_weird(row):
    var = row['variable']
    val = row['value_numeric']
    if var in expected_domains and pd.notnull(val):
        lower, upper = expected_domains[var]
        # For binary variables, only allow 0 or 1.
        if var in ['call', 'sms']:
            if val not in [0, 1]:
                return np.nan
        else:
            if (val < lower) or (val > upper):
                return np.nan
    return val

data['value_numeric'] = data.apply(mark_weird, axis=1)

# --------------------------
# Step 2: Pivot the Data to Wide Format
# --------------------------
df_wide = data.pivot_table(index=['id', 'time'], columns='variable', values='value_numeric')
df_wide = df_wide.sort_index()

# --------------------------
# Step 3: Compute Ranges and Binary Flags for Gower's Distance
# --------------------------
# For continuous variables, range = max - min.
# For binary variables ('call', 'sms'), we treat them as binary.
ranges = df_wide.max() - df_wide.min()
binary_flags = {col: (col in ['call', 'sms']) for col in df_wide.columns}

# --------------------------
# Step 4: Define Gower's Distance Function for a Single Missing Row
# --------------------------
def gower_distance_row(x, X, ranges, binary_flags):
    """
    Compute Gower distances between a single row x and each row in X.
    x: 1D numpy array for the missing row.
    X: 2D numpy array for available rows.
    ranges: Pandas Series with ranges for each column.
    binary_flags: Dictionary indicating if a column is binary.
    Returns: 1D numpy array of distances.
    """
    # Number of features
    n_features = len(x)
    # Initialize a distance matrix (n_samples,)
    dists = np.zeros(X.shape[0])
    
    # For each feature, compute contribution if both values are non-missing.
    for j in range(n_features):
        # Create mask: only where both x and X[:, j] are not NaN.
        mask = ~np.isnan(x[j]) & ~np.isnan(X[:, j])
        if np.sum(mask) == 0:
            # No available data for this feature; skip.
            continue
        if binary_flags[list(ranges.index)[j]]:
            # For binary, distance is 0 if equal, 1 if not.
            diff = (X[mask, j] != x[j]).astype(float)
        else:
            # For continuous, normalized absolute difference.
            # Avoid division by zero in case range is 0.
            r = ranges.iloc[j] if ranges.iloc[j] != 0 else 1
            diff = np.abs(X[mask, j] - x[j]) / r
        # For each row in X, add the difference if that feature is available.
        # We need to accumulate per row; use a temporary array for the j-th feature.
        temp = np.full(X.shape[0], np.nan)
        temp[mask] = diff
        # Replace NaN with 0 so they don't contribute.
        temp = np.nan_to_num(temp, nan=0.0)
        dists += temp
    # Count available features per row
    available_counts = (~np.isnan(X)).sum(axis=1)
    # Avoid division by zero; if no features available, set denominator to 1.
    available_counts[available_counts == 0] = 1
    # Normalize distance by number of available features.
    return dists / available_counts

# --------------------------
# Step 5: Fast Gower-based Imputation (Overview Logging)
# --------------------------
# We will process each variable (except 'mood') and for each missing value, find the
# five nearest neighbors (based on Gower's distance) and impute the missing value as
# the average of these neighbors.
imputation_summary = {}

for var in df_wide.columns:
    if var == "mood":
        continue
    # Identify rows with missing values in this variable.
    missing_mask = df_wide[var].isna()
    missing_indices = df_wide[missing_mask].index
    if len(missing_indices) == 0:
        continue
    
    # Identify rows where the variable is available.
    available_mask = df_wide[var].notna()
    available_indices = df_wide[available_mask].index
    # Convert available data to a NumPy array.
    X_available = df_wide.loc[available_indices].values
    
    # Prepare to record imputation details.
    imputed_count = 0
    
    # For each missing row, compute Gower distances to available rows.
    for idx in missing_indices:
        x = df_wide.loc[idx].values  # 1D array for the missing row.
        distances = gower_distance_row(x, X_available, ranges, binary_flags)
        # Get indices of the 5 nearest neighbors.
        if len(distances) < 5:
            k = len(distances)
        else:
            k = 5
        neighbor_idx = np.argsort(distances)[:k]
        neighbor_values = df_wide.loc[available_indices, var].values[neighbor_idx]
        imputed_value = np.mean(neighbor_values)
        # Impute the missing value.
        df_wide.loc[idx, var] = imputed_value
        imputed_count += 1
    imputation_summary[var] = imputed_count

# Print an overview summary.
print("\nGower-based imputation overview (excluding 'mood'):")
for var, count in imputation_summary.items():
    print(f"Variable '{var}': Imputed {count} missing values using the average of 5 nearest neighbors.")

# --------------------------
# Step 6: (Optional) Plot Distributions for Variables with Changes (Excluding 'mood')
# --------------------------
for var in df_wide.columns:
    if var == "mood":
        continue
    # Identify indices that were originally missing.
    original_missing = data.pivot_table(index=['id', 'time'], columns='variable', values='value_numeric')[var].isna()
    if original_missing.sum() == 0:
        continue
    imputed_values = df_wide.loc[original_missing, var]
    plt.figure(figsize=(8, 4))
    plt.hist(imputed_values, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of Imputed Values for '{var}'")
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.show()

print("\nGower-based KNN imputation is complete.")
