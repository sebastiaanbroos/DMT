import pandas as pd

# Load the data with date parsing
df = pd.read_csv('data/daily_removed_incomplete_moods_imputated.csv', parse_dates=['date'])

def feature_engineering(df, target="mood_avg", history=3):
    """
    Create features using a history of past user data.
    Instead of replacing the average daily mood column, the function adds a 
    new column with the average mood over the past `history` days.
    """
    # Sort the DataFrame by user id and date
    df = df.sort_values(by=['id', 'date']).reset_index(drop=True)
    
    # For each user, compute the rolling average of the target over the past history days.
    # The .shift(1) ensures that the rolling average only uses past days and not the current day.
    df[f"{target}_hist"] = df.groupby("id")[target].transform(
        lambda x: x.rolling(window=history, min_periods=history).mean().shift(1)
    )
    
    # Remove rows that do not have sufficient history (rolling average will be NaN)
    df = df.dropna(subset=[f"{target}_hist"]).reset_index(drop=True)
    
    return df

# Apply feature engineering to add the 3-day historical average of mood_avg
data_fe = feature_engineering(df, target="mood_avg", history=3)

# Save the resulting DataFrame to CSV
output_csv = "data/data_after_fe.csv"
data_fe.to_csv(output_csv, index=False)
print("Feature-engineered data saved to", output_csv)
print("Shape of the dataset after feature engineering:", data_fe.shape)
