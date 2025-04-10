import pandas as pd
import numpy as np



# ---> changed "screen": (0, np.inf)
# --------------------------
# Expected Domains: Define allowed ranges for each variable.
# --------------------------
expected_domains = {
    "mood": (1, 10),
    "circumplex.arousal": (-2, 2),
    "circumplex.valence": (-2, 2),
    "activity": (0, 1),
    # For 'screen' and all appCat variables,
    # we assume that duration should be non-negative.
    "screen": (0, 1440),  # because it shouldn't be more than 24 hours
    "call": (0, 1),    # binary: exactly 0 or 1.
    "sms": (0, 1),     # binary: exactly 0 or 1.
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
# Define Aggregation Functions for Each Variable
# --------------------------
# We want to use:
# - np.sum for durations (where the upper bound is np.inf)
# - np.max for binary variables (call, sms)
# - np.mean for the remaining (like mood, circumplex, activity)
binary_vars = ["call", "sms", "activity"]
agg_funcs = {}
for var, domain in expected_domains.items():
    if var in binary_vars:
        agg_funcs[var] = np.max      # if any record is 1, then the daily value is 1.
    elif domain[1] == np.inf:
        agg_funcs[var] = np.sum      # sum durations.
    else:
        agg_funcs[var] = np.mean     # otherwise, average the values.

# --------------------------
# Load the Data
# --------------------------
data = pd.read_csv("data/dataset_mood_smartphone.csv", parse_dates=["time"])

# Convert the 'value' column to numeric.
data['value_numeric'] = pd.to_numeric(data['value'], errors='coerce')

# --------------------------
# Remove Rows with Missing value_numeric
# --------------------------
num_nas = data['value_numeric'].isnull().sum()
print(f"Number of missing values in 'value_numeric': {num_nas}")
data = data.dropna(subset=['value_numeric'])

# --------------------------
# Remove Records Falling Outside the Expected Domain
# --------------------------
def within_domain(row):
    var = row['variable']
    if var not in expected_domains:
        return True  # keep if the variable is not defined in expected_domains.
    lo, hi = expected_domains[var]
    # For domains with hi == np.inf, only check the lower bound.
    if hi == np.inf:
        return row['value_numeric'] >= lo
    else:
        return lo <= row['value_numeric'] <= hi

data = data[data.apply(within_domain, axis=1)].copy()

# --------------------------
# Add a 'date' Column for Daily Grouping
# --------------------------
data['date'] = data['time'].dt.date

# --------------------------
# Aggregate the Data by (id, date, variable)
# --------------------------
# For each group (subject, day, variable) apply the appropriate function from agg_funcs.
def aggregate_daily(group):
    var = group['variable'].iloc[0]
    func = agg_funcs.get(var, np.mean)
    # Aggregate the 'value_numeric' column.
    return func(group['value_numeric'])

daily_agg = data.groupby(['id', 'date', 'variable'], group_keys=False)\
                .apply(aggregate_daily)\
                .reset_index(name='daily_value')

print("Daily aggregation (pre-pivot) shape:", daily_agg.shape)
print(daily_agg.head(10))

# --------------------------
# Pivot the Aggregated Data: one row per subject and date
# --------------------------
df_daily = daily_agg.pivot_table(index=['id', 'date'], columns='variable', values='daily_value')
df_daily = df_daily.reset_index()

print("Pivoted daily data shape:", df_daily.shape)
print("Missing values per variable in daily data:")
print(df_daily.isnull().sum())

# --------------------------
# Remove Rows where 'mood' is not Present
# --------------------------
df_daily = df_daily.dropna(subset=['mood'])
print("Pivoted daily data shape after dropping rows without mood:", df_daily.shape)

# --------------------------
# Save the Daily Aggregated Data to CSV
# --------------------------
output_csv = "data/daily_aggregated_imputed_nika.csv"
df_daily.to_csv(output_csv, index=False)
print(f"Saved daily aggregated data to {output_csv}")
