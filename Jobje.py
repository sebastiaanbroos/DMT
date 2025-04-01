import pandas as pd

# Load the CSV file
df = pd.read_csv('data/dataset_mood_smartphone.csv', parse_dates=['time'])

# Display the first few rows to verify
print(df.head())