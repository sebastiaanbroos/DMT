#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =============================================================================
# 1. DATA LOADING, INITIAL IMPUTATION, AND NORMALIZATION
# =============================================================================

# Load the CSV file (assumes columns: id, date, plus candidate variables)
data_path = "/Users/s.broos/Documents/DMT/data/daily_removed_incomplete_moods_non-imputated.csv"
df = pd.read_csv(data_path, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Select candidate columns (all columns except "id" and "date")
impute_cols = [col for col in df.columns if col not in ["id", "date"]]
df_data = df[impute_cols].copy()

# Fill naturally missing values with column mean and then normalize (z-score)
for col in impute_cols:
    df_data[col].fillna(df_data[col].mean(), inplace=True)
    df_data[col] = (df_data[col] - df_data[col].mean()) / df_data[col].std()

# Convert complete data to a NumPy array (float32 for PyTorch)
data_array = df_data.values.astype(np.float32)

# =============================================================================
# 2. PYTORCH DATASET: A DENOSING AUTOENCODER FRAMEWORK
# =============================================================================

class ImputationDataset(Dataset):
    def __init__(self, data, missing_prob=0.2):
        """
        data: complete data as a numpy array.
        missing_prob: probability with which a feature is masked (simulating missingness).
        """
        self.data = data
        self.missing_prob = missing_prob
        self.num_features = data.shape[1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get a complete sample
        x_complete = self.data[idx].copy()
        # Generate a binary mask: 1 (observed) with probability (1 - missing_prob), 0 (masked) otherwise.
        mask = (np.random.rand(self.num_features) > self.missing_prob).astype(np.float32)
        # Create the corrupted input: for missing entries, set the value to 0 (0 is the mean after normalization)
        x_corrupt = x_complete * mask
        # Concatenate the corrupted data and the mask to form the input vector
        input_vec = np.concatenate([x_corrupt, mask])
        target = x_complete  # The complete, true data as the target
        return input_vec, target

# Create dataset and dataloader (shuffling training samples)
missing_probability = 0.2
dataset = ImputationDataset(data_array, missing_prob=missing_probability)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# =============================================================================
# 3. NEURAL NETWORK: A FEEDFORWARD AUTOENCODER
# =============================================================================

class ImputationNN(nn.Module):
    def __init__(self, original_dim, hidden_dim1=128, hidden_dim2=64):
        """
        original_dim: number of features (before appending mask).
        The input dimension is twice the original_dim.
        """
        super(ImputationNN, self).__init__()
        input_dim = original_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, original_dim)  # Reconstruct the complete data
        )
    
    def forward(self, x):
        return self.net(x)

# Initialize the network using the number of candidate variables.
original_dim = data_array.shape[1]
model = ImputationNN(original_dim)
model.train()

# =============================================================================
# 4. TRAINING SETUP
# =============================================================================

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()
num_epochs = 500  # Train longer

# =============================================================================
# 5. TRAINING LOOP WITH PROGRESS PER EPOCH
# =============================================================================

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_inputs.size(0)
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch+1:03d}/{num_epochs} - Loss: {epoch_loss:.4f}")

# -----------------------------------------------------------------------------
# 5.1 SAVE THE MODEL
# -----------------------------------------------------------------------------
model_save_path = "imputation_nn_model.pt"
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to {model_save_path}.")

# =============================================================================
# 6. EVALUATION: COMPUTE MAE & RMSE PER VARIABLE
# =============================================================================

model.eval()
np.random.seed(42)  # For reproducible evaluation missing pattern

# Simulate a fixed missingness pattern on the entire dataset for evaluation
masks = (np.random.rand(data_array.shape[0], data_array.shape[1]) > missing_probability).astype(np.float32)
x_corrupt = data_array * masks
# Prepare the input: concatenate the corrupted data with the mask
input_eval = np.concatenate([x_corrupt, masks], axis=1)
input_eval_tensor = torch.tensor(input_eval)
with torch.no_grad():
    output_eval = model(input_eval_tensor).numpy()

# Compute MAE and RMSE for each variable only on the imputed (masked) entries
evaluation_results = {}
for i, col in enumerate(impute_cols):
    # Consider only indices where the entry was masked (i.e., missing simulation)
    missing_indices = np.where(masks[:, i] == 0)[0]
    if len(missing_indices) == 0:
        continue  # Skip if none were masked (unlikely with 20% missing)
    true_vals = data_array[missing_indices, i]
    pred_vals = output_eval[missing_indices, i]
    mae = np.mean(np.abs(pred_vals - true_vals))
    rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
    evaluation_results[col] = {"MAE": mae, "RMSE": rmse}

# Print the per-variable imputation performance
print("\n====== Evaluation Results per Variable ======")
for col, metrics in evaluation_results.items():
    print(f"{col} -> MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
