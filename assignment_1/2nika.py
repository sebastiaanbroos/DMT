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

def user_with_most_entries(df):
    """
    Finds user and date with the most total entries in the dataset
    """
    
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date

    # group by user and date, count entries
    entry_counts = (
        df.groupby(['id', 'date'])
        .size()
        .reset_index(name='entry_count')
    )

    max_entry = entry_counts.loc[entry_counts['entry_count'].idxmax()]
    
    return max_entry

top_user = user_with_most_entries(data)
print("User with most entries on a single day:", top_user)

# date range
print("Date range:", data['time'].min(), "to", data['time'].max())

# creating a dataframe that contains the number of mood entries per user per day
def count_daily_mood_entries(df, output_path='daily_mood_counts.csv'):
    # Filter only mood entries
    mood_data = df[df['variable'] == 'mood'].copy()
    
    # Ensure datetime and extract date
    mood_data['time'] = pd.to_datetime(mood_data['time'])
    mood_data['date'] = mood_data['time'].dt.date

    # Count number of mood entries per user per day
    mood_counts = (
        mood_data.groupby(['id', 'date'])
        .size()
        .reset_index(name='mood_entry_count')
    )
    
    mood_counts.to_csv(output_path, index=False)
    
    print(f"Saved daily mood counts to '{output_path}'")
    return mood_counts

mood_daily_counts = count_daily_mood_entries(data, 'data/daily_mood_counts.csv')
print(mood_daily_counts.head())

overall_mood_mean = mood_daily_counts['mood_entry_count'].mean()
print(f"Average number of mood entries per user per day: {overall_mood_mean:.2f}")

# most active hour of the day for mood logging
data['hour'] = pd.to_datetime(data['time']).dt.hour
mood_hours = data[data['variable'] == 'mood']['hour'].value_counts().sort_index()
mood_hours.plot(kind='bar', title='Mood Entries by Hour of Day')

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
    
    var_data = data[data['variable'] == var].copy()
    
    daily_avg = (
    var_data.groupby(["id", "date"])["value"]
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


