import pandas as pd

# Load the mean and std files
means = pd.read_csv("/Users/s.broos/Documents/DMT/data/column_means.csv", index_col=0).squeeze()
stds = pd.read_csv("/Users/s.broos/Documents/DMT/data/column_stds.csv", index_col=0).squeeze()

# Extract mood_avg mean and std
mean_mood = means["mood_avg"]
std_mood = stds["mood_avg"]

# Compute real-value thresholds for the three classes
low_max = mean_mood - 0.5 * std_mood
neutral_min = low_max
neutral_max = mean_mood + 0.5 * std_mood
high_min = neutral_max

# Print results
print("Class ranges (real mood_avg values):")
print(f"Low mood:     mood_avg <  {low_max:.3f}")
print(f"Neutral mood: {neutral_min:.3f} ≤ mood_avg ≤ {neutral_max:.3f}")
print(f"High mood:    mood_avg >  {high_min:.3f}")
