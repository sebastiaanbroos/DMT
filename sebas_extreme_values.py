import pandas as pd
import numpy as np

# --------------------------
# Step 1: Load the Data
# --------------------------
data = pd.read_csv("data/dataset_mood_smartphone.csv", parse_dates=["time"])

# Convert the 'value' column to numeric (if applicable)
data['value_numeric'] = pd.to_numeric(data['value'], errors='coerce')

# --------------------------
# Step 2: Check for Missing Values
# --------------------------
print("Missing Values per Column:")
print(data.isnull().sum())

# Missing values per variable (for 'value_numeric'):
missing_per_variable = data.groupby('variable')['value_numeric'].apply(lambda x: x.isnull().sum())
print("\nMissing Values per Variable:")
print(missing_per_variable)

# --------------------------
# Step 3: Define Expected Domains for Each Variable
# --------------------------
expected_domains = {
    "mood": (1, 10),
    "circumplex.arousal": (-2, 2),
    "circumplex.valence": (-2, 2),
    "activity": (0, 1),
    # For 'screen' and all appCat and time-related variables,
    # we assume that duration should be non-negative.
    "screen": (0, np.inf),
    "call": (0, 1),  # should be exactly 0 or 1; we'll check for allowed values separately.
    "sms": (0, 1),   # same as call.
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

# --------------------------
# Step 4: Validate Each Variable's Values Against the Expected Domain
# --------------------------
def check_domain(df, variable, expected_range):
    # Select the subset for the given variable
    subset = df[df['variable'] == variable].copy()
    # For call and sms, enforce that allowed values are only 0 or 1.
    if variable in ["call", "sms"]:
        invalid = subset[~subset['value_numeric'].isin([0, 1])]
    else:
        lower, upper = expected_range
        invalid = subset[(subset['value_numeric'] < lower) | (subset['value_numeric'] > upper)]
    count_invalid = invalid.shape[0]
    return count_invalid, invalid

# Check each variable against its expected domain
for var, exp_range in expected_domains.items():
    count_invalid, invalid_rows = check_domain(data, var, exp_range)
    print("\n" + "="*50)
    print(f"Variable: {var}")
    print(f"Expected domain: {exp_range}")
    print(f"Number of entries outside expected domain: {count_invalid}")
    if count_invalid > 0:
        print("Sample problematic rows:")
        print(invalid_rows[['id', 'time', 'value', 'value_numeric']].head())

def detect_extreme_iqr(df, variable, expected_range):
    # Filter data for the given variable and exclude missing values
    valid = df[(df['variable'] == variable) & (df['value_numeric'].notnull())].copy()
    
    # Further filter based on expected domain
    if variable in ["call", "sms"]:
        valid = valid[valid['value_numeric'].isin([0, 1])]
    else:
        lower_valid, upper_valid = expected_range
        valid = valid[(valid['value_numeric'] >= lower_valid) & (valid['value_numeric'] <= upper_valid)]
    
    # Calculate the first (Q1) and third (Q3) quartiles on valid values only
    Q1 = valid['value_numeric'].quantile(0.01)
    Q3 = valid['value_numeric'].quantile(0.99)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for extreme values
    lower_bound = Q1 - 1 * IQR
    upper_bound = Q3 + 1 * IQR

    # Identify extreme values from the valid subset
    extremes = valid[(valid['value_numeric'] < lower_bound) | (valid['value_numeric'] > upper_bound)]
    return extremes, lower_bound, upper_bound

# Apply the IQR approach for each variable in expected_domains
for var, exp_range in expected_domains.items():
    extremes, lb, ub = detect_extreme_iqr(data, var, exp_range)
    print("\n" + "-"*50)
    print(f"IQR Outlier Detection for Variable: {var}")
    print(f"Lower Bound: {lb:.2f}, Upper Bound: {ub:.2f}")
    print(f"Number of extreme values detected: {extremes.shape[0]}")