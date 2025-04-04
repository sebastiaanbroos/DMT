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
std_val = mood_data.std()

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