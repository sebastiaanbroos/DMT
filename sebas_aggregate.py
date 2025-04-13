import pandas as pd
import numpy as np

# --------------------------
# Updated Expected Domains: Define allowed ranges for each variable.
# --------------------------
expected_domains = {
    "mood": (1, 10),
    "circumplex.arousal": (-2, 2),
    "circumplex.valence": (-2, 2),
    "activity": (0, 1),
    "screen": (0, 24000),            # updated max value
    "call": (0, 1),                  # binary
    "sms": (0, 1),                   # binary
    "appCat.builtin": (0, 7500),       # updated max value
    "appCat.communication": (0, 11000),# updated max value
    "appCat.entertainment": (0, 9000), # updated max value
    "appCat.finance": (0, 600),     # no max provided, remain unchanged
    "appCat.game": (0, 4000),        # unchanged
    "appCat.office": (0, 3000),        # updated max value
    "appCat.other": (0, 1250),       # unchanged
    "appCat.social": (0, 10000),       # updated max value
    "appCat.travel": (0, 2100),        # updated max value
    "appCat.unknown": (0, 600),        # updated max value
    "appCat.utilities": (0, 750),      # updated max value
    "appCat.weather": (0, 500)         # updated max value
}

# --------------------------
# Load the Data
# --------------------------
data = pd.read_csv("/Users/s.broos/Documents/DMT/data/dataset_mood_smartphone.csv", parse_dates=["time"])

# Convert the 'value' column to numeric.
data['value_numeric'] = pd.to_numeric(data['value'], errors='coerce')

# --------------------------
# Remove Rows with Missing value_numeric
# --------------------------
data = data.dropna(subset=['value_numeric'])

# --------------------------
# Cap Values Above Maximum Rather than Dropping Them
# --------------------------
def cap_value(row):
    var = row['variable']
    # Only modify values if the variable is defined in expected_domains.
    if var not in expected_domains:
        return row['value_numeric']
    lo, hi = expected_domains[var]
    # Replace high values with the maximum allowed.
    if hi != np.inf and row['value_numeric'] > hi:
        return hi
    else:
        return row['value_numeric']

# Apply the cap_value function to update the value_numeric column.
data['value_numeric'] = data.apply(cap_value, axis=1)

# --------------------------
# Add a 'date' Column for Daily Grouping
# --------------------------
data['date'] = data['time'].dt.date

# --------------------------
# Define Variables for Special Aggregation and the Normal ones
# --------------------------
special_vars = ['mood', 'circumplex.arousal', 'circumplex.valence']

# --------------------------
# Special Aggregation: For mood, arousal, and valence calculate the mean and standard deviation
# --------------------------
special_data = data[data['variable'].isin(special_vars)]
special_agg = special_data.groupby(['id', 'date', 'variable'])['value_numeric'].agg(['mean', 'std']).reset_index()

# Pivot so that each variable produces two columns: one for average and one for standard deviation.
special_avg = special_agg.pivot_table(index=['id', 'date'], columns='variable', values='mean')
special_avg.columns = [f"{col}_avg" for col in special_avg.columns]

special_std = special_agg.pivot_table(index=['id', 'date'], columns='variable', values='std')
special_std.columns = [f"{col}_std" for col in special_std.columns]

# Merge average and standard deviation results for the special variables.
special_pivot = special_avg.join(special_std, how='outer').reset_index()

# --------------------------
# Normal Aggregation: For all other variables, sum the values.
# --------------------------
normal_data = data[~data['variable'].isin(special_vars)]
normal_agg = normal_data.groupby(['id', 'date', 'variable'])['value_numeric'].sum().reset_index()

# Pivot the normal aggregated values: one row per id and date.
normal_pivot = normal_agg.pivot(index=['id', 'date'], columns='variable', values='value_numeric').reset_index()

# --------------------------
# Combine the Special and Normal Aggregations
# --------------------------
df_daily = pd.merge(special_pivot, normal_pivot, on=['id', 'date'], how='outer')

# --------------------------
# Remove Rows where 'mood' is not Present.
# Note: Since 'mood' has been aggregated to 'mood_avg', we drop rows where that is missing.
# --------------------------
before_drop = df_daily.shape[0]
df_daily = df_daily.dropna(subset=['mood_avg'])
after_drop = df_daily.shape[0]
dropped_rows = before_drop - after_drop
print(f"Number of rows dropped due to missing 'mood_avg': {dropped_rows}")

# --------------------------
# Save the Daily Aggregated Data to CSV
# --------------------------
output_csv = "/Users/s.broos/Documents/DMT/data/daily_removed_incomplete_moods_non-imputated.csv"
df_daily.to_csv(output_csv, index=False)
print(f"Saved daily aggregated data to {output_csv}")
