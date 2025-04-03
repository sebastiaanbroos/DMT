# imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TASK 1
# EXPLORATORY DATA ANALYSIS

data = pd.read_csv('data/dataset_mood_smartphone.csv')

# converting time column to datetime format
data["time"] = pd.to_datetime(data['time'])

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

# number of records per variable
variable_counts = data['variable'].value_counts()
print("\nRecords per variable:\n", variable_counts)

# records per user
user_counts = data['id'].value_counts()
print("\nRecords per user:\n", user_counts)

# plotting distribution of "mood" value
plt.figure(figsize=(8, 4))
sns.histplot(data[data['variable'] == 'mood']['value'], bins=20, kde=True)
plt.title("Distribution of Mood Scores")
plt.xlabel("Mood (1-10)")
plt.ylabel("Frequency")
plt.show()
# --> data is skewed to the right

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
