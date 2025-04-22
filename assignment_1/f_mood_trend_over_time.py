import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
So here I am representing a trend of averaged mood across all users daily over the course of the study.
"""
# Load the dataset
df = pd.read_csv('data/daily_removed_incomplete_moods_non-imputated.csv', parse_dates=['date'])

# Group by date to get the average normalized mood
daily_mood = df.groupby('date')['mood_avg'].mean()

# Calculate the overall mean mood
overall_mean = daily_mood.mean()

# Identify weekends
weekends = [d for d in daily_mood.index if d.weekday() >= 5]

# Plot setup
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 6))

# Plot mood trend
plt.plot(daily_mood.index, daily_mood.values, label='Average Mood', 
         color='#1f77b4', marker='o', markersize=3)

# Highlight max and min mood
max_day = daily_mood.idxmax()
min_day = daily_mood.idxmin()
plt.scatter(max_day, daily_mood[max_day], color='green', s=60, label=f'Max: {daily_mood[max_day]:.2f}')
plt.scatter(min_day, daily_mood[min_day], color='red', s=60, label=f'Min: {daily_mood[min_day]:.2f}')

# Add vertical dashed lines for weekends
for d in weekends:
    plt.axvline(x=d, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)


plt.title("Average Mood Over Time", fontsize=16, fontweight='bold')
plt.xlabel("Date", fontsize=13)
plt.ylabel(" Mood Score", fontsize=13)
plt.xticks(rotation=45)
plt.ylim(5, 9)  
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
plt.axhline(y=overall_mean, color='orange', linestyle='--', linewidth=2,
            label=f'Overall Mean: {overall_mean:.2f}')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# Comparing weekend and weekday moods
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = df['weekday'] >= 5

mean_weekend = df[df['is_weekend']]['mood_avg'].mean()
mean_weekday = df[~df['is_weekend']]['mood_avg'].mean()
print(f"Weekend avg: {mean_weekend:.2f}, Weekday avg: {mean_weekday:.2f}")