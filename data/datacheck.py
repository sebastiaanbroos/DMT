import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load your CSV data
data_path = "/Users/s.broos/Documents/DMT/data/daily_removed_incomplete_moods_imputated.csv"
df = pd.read_csv(data_path)

# Identify numeric columns to normalize
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Normalize the numeric features using z-score normalization
df_normalized = df.copy()
df_normalized[numeric_cols] = df_normalized[numeric_cols].apply(zscore)

# Loop over each numeric column to plot its distribution
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    plt.hist(df_normalized[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of Normalized {col}')
    plt.xlabel(f'Normalized values of {col}')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
