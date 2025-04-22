import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('/Users/s.broos/Documents/DMT_data/training_set_VU_DM.csv', parse_dates=['date_time'])

# 1. Basic overview
print("Shape:", df.shape)
print("\nColumns and dtypes:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())

# 2. Missing values analysis
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
print("\nMissing values (%):")
print(missing_df)

# 3. Summary statistics for numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nSummary statistics:")
print(df[num_cols].describe())

# 8. Grouped analysis: average ADR and rating by country
grouped = df.groupby('visitor_location_country_id')[['visitor_hist_adr_usd', 'visitor_hist_starrating']].mean()
print("\nAverage ADR and Star Rating by Visitor Country:")
print(grouped.sort_values(by='visitor_hist_adr_usd', ascending=False).head())
