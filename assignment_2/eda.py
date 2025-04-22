import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('data.csv', parse_dates=['date_time'])

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

# 4. Distribution plots
for col in ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_starrating', 'prop_review_score']:
    plt.figure()
    df[col].dropna().hist()
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# 5. Time-based analysis: bookings over time
plt.figure()
bookings_by_date = df.set_index('date_time').resample('D').size()
bookings_by_date.plot()
plt.title('Bookings per Day')
plt.xlabel('Date')
plt.ylabel('Number of Bookings')
plt.tight_layout()
plt.show()

# 6. Correlation matrix
corr = df[num_cols].corr()
plt.figure(figsize=(8,6))
plt.imshow(corr, interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 7. Scatter plot of ADR vs Review score
good_data = df.dropna(subset=['visitor_hist_adr_usd', 'prop_review_score'])
plt.figure()
plt.scatter(good_data['visitor_hist_adr_usd'], good_data['prop_review_score'], alpha=0.5)
plt.title('Visitor ADR vs Property Review Score')
plt.xlabel('visitor_hist_adr_usd')
plt.ylabel('prop_review_score')
plt.tight_layout()
plt.show()

# 8. Grouped analysis: average ADR and rating by country
grouped = df.groupby('visitor_location_country_id')[['visitor_hist_adr_usd', 'visitor_hist_starrating']].mean()
print("\nAverage ADR and Star Rating by Visitor Country:")
print(grouped.sort_values(by='visitor_hist_adr_usd', ascending=False).head())
