import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
# Load the CSV file
data = pd.read_csv('data/dataset_mood_smartphone.csv')

# converting time column to datetime format
data["time"] = pd.to_datetime(data['time'])
# group by date so I can summarize by day
data["date"] = data["time"].dt.date

#removing the first column from the dataset
data = data.drop(columns=['Unnamed: 0'])

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Create the figure
plt.figure(figsize=(10, 6))

# Filter just the mood data
mood_data = data[data['variable'] == 'mood']['value']

# Create custom bins from 1 to 10
bins = np.linspace(1, 10, 30)

# Create the histogram without KDE
ax = sns.histplot(mood_data, bins=bins, kde=False, color='#6BAED6')

# Calculate statistics
mean_val = mood_data.mean()
median_val = mood_data.median()
std_val = mood_data.std()
print(f"Mean: {mean_val:.2f}, Median: {median_val:.2f}, Standard Deviation: {std_val:.2f}")

# Create a normal distribution overlay using NumPy
x = np.linspace(1, 10, 1000)
# Using NumPy's normal PDF formula instead of scipy
y = 1/(std_val * np.sqrt(2 * np.pi)) * np.exp( - (x - mean_val)**2 / (2 * std_val**2))

# Scale the normal distribution to match the histogram height
hist_heights = [p.get_height() for p in ax.patches]
hist_max = max(hist_heights) if hist_heights else 1
pdf_max = max(y)
scale_factor = hist_max / pdf_max

# Plot the normal distribution curve
plt.plot(x, y * scale_factor, color='#08306B', linewidth=2, 
         label=f'Normal Distribution\nMean={mean_val:.2f}, SD={std_val:.2f}')

# Add title and labels with improved formatting
plt.title("Distribution of Mood Scores", fontweight='bold', pad=15)
plt.xlabel("Mood Value", fontweight='bold')
plt.ylabel("Frequency", fontweight='bold')

# Format y-axis to avoid scientific notation
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# Add vertical line for mean
plt.axvline(mean_val, color='#E41A1C', linestyle='-', alpha=0.7, 
           label=f'Mean: {mean_val:.2f}', linewidth=1.5)

# Add a legend
plt.legend(loc='upper right', framealpha=0.9)

# Add subtle grid but only on the y-axis
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set x-axis limits to match the shown range
plt.xlim(1, 10)

# Add a descriptive caption below the plot
plt.figtext(0.5, 0.01, 
           "Histogram of mood scores with normal distribution overlay.",
           ha="center", fontsize=12, style='italic')

# Tight layout to ensure everything fits
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

plt.show()


plt.figure(figsize=(12, 2.5))

# Only use screen values
screen_data = data[data['variable'] == 'screen']['value']

sns.stripplot(x=screen_data, color='#4292C6', size=4, jitter=True)

plt.title("All Screen Durations (Individual Data Points)", fontweight='bold', pad=10)
plt.xlabel("Screen Duration (seconds)", fontweight='bold')
plt.yticks([])  # No need for y-axis here

# Add mean line
mean_val = screen_data.mean()
plt.axvline(mean_val, color='#E41A1C', linestyle='--', linewidth=1.5,
            label=f'Mean: {mean_val:.2f}')

# Legend
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


# Step 1: Filter to only 'call' entries
call_data = data[data['variable'] == 'call'].copy()

# Step 2: Convert time to datetime and extract the date
call_data['date'] = pd.to_datetime(call_data['time']).dt.date

# Step 3: Group by user and date, and count
call_counts = call_data.groupby(['id', 'date']).size().reset_index(name='call_count')

# Step 4: Filter to only those with 2 or more calls
multiple_calls = call_counts[call_counts['call_count'] >= 2]

# Show the result
print(multiple_calls)


# Step 1: Filter to only 'sms' entries
sms_data = data[data['variable'] == 'sms'].copy()

# Step 2: Convert time to datetime and extract the date
sms_data['date'] = pd.to_datetime(sms_data['time']).dt.date

# Step 3: Group by user and date, and count
sms_counts = sms_data.groupby(['id', 'date']).size().reset_index(name='sms_count')

# Step 4: Filter to only those with 2 or more SMS events on the same day
multiple_sms = sms_counts[sms_counts['sms_count'] >= 2]

# Show the result
print(multiple_sms)
max_sms_count = sms_counts['sms_count'].max()
print(f"Highest number of SMS events by a user on a single day: {max_sms_count}")


app_categories = [
    "screen", "appCat.builtin", "appCat.communication", "appCat.entertainment",
    "appCat.office", "appCat.social", "appCat.travel",
    "appCat.unknown", "appCat.utilities", "appCat.weather"
]

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

for category in app_categories:
    # Filter op juiste categorie én geen negatieve waarden
    cat_data = data[(data["variable"] == category) & (data["value"] >= 0)].copy()

    plt.figure(figsize=(10, 4))
    ax = sns.scatterplot(
        x="time", y="value", data=cat_data,
        alpha=0.5, color="#3182bd", s=12
    )

    # IQR-berekening voor outlier threshold
    q1 = cat_data["value"].quantile(0.25)
    q3 = cat_data["value"].quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    # Threshold-lijn tekenen
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ≈ {threshold:.0f}")

    plt.title(f"App Usage: {category}", fontweight='bold')
    plt.xlabel("Time")
    plt.ylabel("Usage (seconds)")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

for category in app_categories:
    # Filter: alleen waarden tussen 0 en 200
    cat_data = data[
        (data["variable"] == category) &
        (data["value"] >= 0) &
        (data["value"] <= 200)
    ].copy()

    # Bereken threshold op basis van IQR
    q1 = cat_data["value"].quantile(0.25)
    q3 = cat_data["value"].quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    # Plot histogram
    plt.figure(figsize=(10, 4))
    sns.histplot(cat_data["value"], bins=40, color="#6baed6", edgecolor='black', alpha=0.8)

    # Verticale lijn bij threshold
    plt.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ≈ {threshold:.0f}")

    plt.title(f"App Usage Distribution: {category}", fontweight='bold')
    plt.xlabel("Usage (seconds)")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(axis="y", linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Laad de data
data = pd.read_csv("data/daily_aggregated_imputed_new.csv", parse_dates=["date"])

# App-categorieën (alle kolommen van interesse)
app_categories = [
    "screen", "appCat.builtin", "appCat.communication", "appCat.entertainment",
    "appCat.office", "appCat.social", "appCat.travel",
    "appCat.unknown", "appCat.utilities", "appCat.weather"
]

# Algemene stijlinstellingen
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# Scatterplots per categorie met IQR-threshold
for category in app_categories:
    cat_data = data[[category, "date"]].dropna()
    cat_data = cat_data[cat_data[category] >= 0]

    if cat_data.empty:
        continue

    # IQR-berekening
    q1 = cat_data[category].quantile(0.25)
    q3 = cat_data[category].quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    plt.figure(figsize=(10, 4))
    sns.scatterplot(
        x="date", y=category, data=cat_data,
        alpha=0.5, color="#3182bd", s=12
    )
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ≈ {threshold:.0f}")
    plt.title(f"App Usage Over Time: {category}", fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Usage (seconds)")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Histogrammen per categorie met threshold
for category in app_categories:
    cat_data = data[[category]].dropna()
    cat_data = cat_data[
        (cat_data[category] >= 0) &
        (cat_data[category] <= 1000000)
    ]

    if cat_data.empty:
        continue

    # Threshold berekening
    q1 = cat_data[category].quantile(0.25)
    q3 = cat_data[category].quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    plt.figure(figsize=(10, 4))
    sns.histplot(cat_data[category], bins=40, color="#6baed6", edgecolor='black', alpha=0.8)
    plt.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ≈ {threshold:.0f}")
    plt.title(f"App Usage Distribution: {category}", fontweight='bold')
    plt.xlabel("Usage (seconds)")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(axis="y", linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()