import pandas as pd

# === Step 1: Load normalized dataset ===
df = pd.read_csv("data/daily_removed_incomplete_moods_imputated.csv", parse_dates=["date"])

# === Step 2: Load means and stds used for normalization ===
means = pd.read_csv("data/column_means.csv", index_col=0).squeeze("columns")
stds  = pd.read_csv("data/column_stds.csv", index_col=0).squeeze("columns")

# === Step 3: Select only the numeric columns that were normalized ===
# (You may need to adjust this based on what you normalized)
numeric_cols = means.index.tolist()

# === Step 4: Unnormalize the numeric columns ===
df_unnorm = df.copy()
for col in numeric_cols:
    if col in df_unnorm.columns:
        df_unnorm[col] = df_unnorm[col] * stds[col] + means[col]

# === Step 5: Save the unnormalized dataset ===
df_unnorm.to_csv("data_after_unnormalized.csv", index=False)
print("Unnormalized data saved to 'data_after_unnormalized.csv'")
