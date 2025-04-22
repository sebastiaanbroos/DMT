import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.decomposition import PCA

# ----------------------------
# 1. Data Loading and Preparation
# ----------------------------

# Load the data_after_fe.csv file (which already has feature-engineered columns)
# Expected columns include "id", "date", "mood_avg", and all other predictors.
df = pd.read_csv("/Users/s.broos/Documents/DMT/data/data_after_fe.csv", parse_dates=["date"])

# Drop non-feature columns (keeping the target "mood_avg")
df_features = df.drop(columns=["id", "date"])

# ----------------------------
# 2. Splitting Data into Features and Labels
# ----------------------------

# Separate features (all columns except "mood_avg") and the target variable "mood_avg"
X = df_features.drop(columns=["mood_avg"])
y = df_features["mood_avg"]

# Save feature names BEFORE scaling (to use later for feature importance)
feature_names = X.columns


X_scaled = X

# ----------------------------
# 3. Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ----------------------------
# 4. Define Models for Evaluation
# ----------------------------

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Lasso": Lasso(alpha=0.01, random_state=42, max_iter=10000),
    "XGBoost": XGBRegressor(use_label_encoder=False, random_state=42, eval_metric='rmse'),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42)
}

# Create a dictionary to store feature rankings from each model.
all_ranks = {}

# ----------------------------
# 5. Evaluate Each Model and Extract Top 20 Features
# ----------------------------

results = []

for name, model in models.items():
    print(f"\n========== {name} ==========")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.4f} | MSE: {mse:.4f} | R2 Score: {r2:.4f}")
    
    # Save performance metrics for comparison.
    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    })
    
    # Depending on the model, extract feature importances (for tree-based models)
    # or absolute coefficients (for Lasso). Then, rank the features (lower rank = more important).
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        rank_series = pd.Series(importances, index=feature_names).rank(ascending=False, method='min')
        print("\nTop 20 Feature Importances (Rank):")
        print(rank_series.sort_values().head(20))
        all_ranks[name] = rank_series  # store entire series of ranks
        # Also plot the top 20 features (by raw importance)
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        feat_imp.head(20).plot(kind='bar')
        plt.title(f"Top 20 Important Features: {name}")
        plt.ylabel("Importance Score")
        plt.xlabel("Feature")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        
    elif hasattr(model, 'coef_'):
        coeffs = np.abs(model.coef_)
        rank_series = pd.Series(coeffs, index=feature_names).rank(ascending=False, method='min')
        print("\nTop 20 Feature Coefficients (Rank):")
        print(rank_series.sort_values().head(20))
        all_ranks[name] = rank_series  # store the ranking for Lasso
        # Plot the top 20 features by absolute coefficient value.
        coeff_series = pd.Series(coeffs, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        coeff_series.head(20).plot(kind='bar')
        plt.title(f"Top 20 Feature Coefficients: {name}")
        plt.ylabel("Absolute Coefficient Value")
        plt.xlabel("Feature")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print(f"{name} does not provide feature importance directly.")

# Create a DataFrame of evaluation results.
results_df = pd.DataFrame(results)
print("\n========== Summary of Model Performance ==========")
print(results_df)

# ----------------------------
# 6. Hyperparameter Tuning with GridSearchCV for RandomForestRegressor
# ----------------------------

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}

grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)

print("\nGridSearchCV Best Params for RandomForest:")
print(grid.best_params_)
print("Best R2 Score on CV:", grid.best_score_)


# ----------------------------
# 8. Combine Feature Ranks Over All Models and Save to CSV
# ----------------------------

# Combine the rankings from all models into a single DataFrame.
ranks_df = pd.DataFrame(all_ranks)
# Compute the average rank (lower values mean more important).
ranks_df["Average_Rank"] = ranks_df.mean(axis=1)
# Sort the features by their average rank.
ranks_df = ranks_df.sort_values("Average_Rank")

# Save the feature ranking DataFrame to a CSV file.
ranks_df.to_csv("feature_ranks.csv")
print("\nFeature ranks saved to 'feature_ranks.csv'.")
