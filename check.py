import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the normalized & imputed data
df = pd.read_csv("daily_removed_incomplete_moods_imputated.csv", parse_dates=["date"])

# 2. Load the saved normalization parameters
means_df = pd.read_csv("column_means.csv", index_col=0)
stds_df  = pd.read_csv("column_stds.csv",  index_col=0)

col_means = means_df.iloc[:, 0]   # extract the first (and only) column as a Series
col_stds  = stds_df.iloc[:, 0]

# 3. Invert z‑score normalization for mood_avg only
df["mood_avg_original"] = df["mood_avg"] * col_stds["mood_avg"] + col_means["mood_avg"]

# 4. Inspect the restored mood distribution
print(df["mood_avg_original"].describe())

# 5. Plot histogram of the original-scale mood_avg
plt.figure(figsize=(8, 5))
plt.hist(df["mood_avg_original"].dropna(), bins=50, edgecolor='black')
plt.title("Distribution of Restored Original mood_avg")
plt.xlabel("mood_avg (original scale)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
