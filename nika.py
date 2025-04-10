# imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TASK 1
# EXPLORATORY DATA ANALYSIS

data = pd.read_csv('data/dataset_mood_smartphone.csv')

# converting time column to datetime format
data["time"] = pd.to_datetime(data['time'])
# group by date so I can summarize by day
data["date"] = data["time"].dt.date

#removing the first column from the dataset
data = data.drop(columns=['Unnamed: 0'])

# some information
print("Number of records:", len(data))
print("Number of unique users:", data['id'].nunique())
print("Number of unique variables:", data['variable'].nunique())
print("Variables:", data['variable'].unique())

# checking for missing values
missing = data.isnull().sum()
print("Missing values in each column: ", missing)
# missing values per variable type
missing_by_var = data[data['value'].isnull()]['variable'].value_counts()
print("Missing values per variable:\n", missing_by_var)

# ranges of values for each variable
range_summary = (
    data.groupby("variable")["value"]
    .agg(["count", "min", "max", "mean", "median"])
    .sort_values(by="count", ascending=False)
)
print("Range summary per variable: ", range_summary)

# relationships/correlation heatmap without dropping columns with missing values
data['date'] = data['time'].dt.date
pivot_df = (
    data.groupby(['id', 'date', 'variable'])['value']
    .mean()
    .unstack()
)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Between Daily-Aggregated Variables")
plt.tight_layout()
plt.show()

# number of records per variable
variable_counts = data['variable'].value_counts()
print("\nRecords per variable: ", variable_counts)

# records per user
user_counts = data['id'].value_counts()
print("\nRecords per user: ", user_counts)

# plotting number of entries per variable
plt.figure(figsize=(10, 4))
variable_counts.plot(kind='bar')
plt.title("Number of Entries per Variable")
plt.xlabel("Variable")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# --> screen has the most entries, followed by builtin and then communication
# --> it is not evenly distributed

# plotting distributions, not thaat pretty (Job has a really good one for mood)
for var in ['mood', 'screen', 'activity', 'circumplex.valence']:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[data['variable'] == var]['value'], bins=30, kde=True)
    plt.title(f"Distribution of {var}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

# plotting variables per user --> this is new, it doesnt work i have to debug it
for var in ['mood', 'screen', 'activity', 'circumplex.valence']:
    daily_avg = (
    var.groupby(["id", "date"])["value"]
    .mean()
    .reset_index()
    )

    # Plot: distribution of daily averages across users
    plt.figure(figsize=(10, 5))
    sns.histplot(daily_avg["value"], bins=30, kde=True)
    plt.title(f"Distribution of Daily Average {var.capitalize()} Usage Across Users")
    plt.xlabel(f"Average Daily {var.capitalize()}")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
"""
# plotting distribution of "mood" value
plt.figure(figsize=(8, 4))
sns.histplot(data[data['variable'] == 'mood']['value'], bins=20, kde=True)
plt.title("Distribution of Mood Scores")
plt.xlabel("Mood (1-10)")
plt.ylabel("Frequency")
plt.show()
# --> data is skewed to the right
"""

