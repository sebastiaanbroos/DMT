""""This is the part where we do Feature Engineering for Task 1C
I will need to create a predictive model for the data set I 
created during this part

- I will have to predict average mood (mood_avg) of teh day for each user
- I need to make use of average mood history during the last (3-5 days --> I need to decide)
"""

import pandas as pd

df = pd.read_csv('data/daily_removed_incomplete_moods_imputated.csv', parse_dates=['date'])


def feature_engineering(df, target = "mood_avg", history = 3):
    """
    I am creating feature set usding history of past user data to predict the target variable
    """
    # I need to sort the dataframe by user and date
    df = df.sort_values(by=['id', 'date']).reset_index(drop=True)
    
    # My features for prediction
    predictors = [col for col in df.columns if col not in ['id', 'date', 'mood_avg']]
    
    all_features = []
    
    for user_id, user_df in df.groupby('id'):
        user_df = user_df.sort_values("date").copy()
        
        # Finding a mean of the past history
        rolling_feats = (
            user_df[predictors]
            .rolling(window=history, min_periods=history)
            .mean()
            .shift(1)
        )
        
        rolling_feats["id"] = user_df["id"]
        rolling_feats["date"] = user_df["date"]
        rolling_feats[f"{target}_target"] = user_df[target]
        
        all_features.append(rolling_feats)
        
    # Removing missing rows --> check this part
    data_features = pd.concat(all_features).dropna().reset_index(drop=True)
    return data_features
        


data_fe = feature_engineering(df, target = "mood_avg", history = 3)

# Saving to csv
data_fe.to_csv("data/data_after_fe.csv", index=False)
print("Feature-engineered data saved to 'data_after_fe.csv'")
print("Shape of teh dataset after fe: ", data_fe.shape)