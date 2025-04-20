import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("feature_ranks.csv", index_col=0)

# selecting top 10 features
top_10 = df.sort_values("Average_Rank").head(10)

# plotting
plt.figure(figsize=(9, 5))
sns.barplot(y=top_10.index, x=top_10["Average_Rank"], palette="crest")
plt.title("Top 10 Features by Average Rank Across Models", fontsize=14)
plt.xlabel("Average Rank (Lower = More Important)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
