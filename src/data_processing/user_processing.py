"""
    This script will handle the processing and saving of the user-specific dataset and model.
    It will be called by the NewFolderHandler in update_base_dataset.py when a new user folder is detected.
"""

import os
import torch
from datetime import datetime

USER_FOLDER_PATH = 'users'
DATA_FOLDER_PATH = 'data/base'
BASE_DATASET_PREFIX = 'base_dataset_'
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

def get_latest_base_dataset_path():
    base_datasets = [
        f for f in os.listdir(DATA_FOLDER_PATH)
        if f.startswith(BASE_DATASET_PREFIX) and f.endswith('.pt')
    ]
    
    if not base_datasets:
        raise FileNotFoundError("No base dataset found.")
    
    # Extract timestamps and sort to get the latest file
    base_datasets.sort(key=lambda x: datetime.strptime(x[len(BASE_DATASET_PREFIX):-3], TIMESTAMP_FORMAT), reverse=True)
    latest_base_dataset = base_datasets[0]
    return os.path.join(DATA_FOLDER_PATH, latest_base_dataset)

def process_user_dataset(user_name, user_dataset):
    user_folder_path = os.path.join(USER_FOLDER_PATH, user_name)
    os.makedirs(user_folder_path, exist_ok=True)
    
    # Save user-specific dataset
    user_dataset_path = os.path.join(user_folder_path, f"{user_name}_dataset.pt")
    torch.save(user_dataset, user_dataset_path)
    
    # Train and save user-specific model
    train_user_model(user_name, user_dataset_path)

def train_user_model(user_name, user_dataset_path):
    latest_base_dataset_path = get_latest_base_dataset_path()
    base_dataset = torch.load(latest_base_dataset_path)
    user_dataset = torch.load(user_dataset_path)
    
    # Combine datasets with higher weight on user dataset
    combined_dataset = torch.cat((base_dataset, user_dataset * 2))  # Example: Double the user dataset weight

    # Placeholder for model training (replace with actual model training code)
    user_model = train_model(combined_dataset)
    
    # Save the trained model
    user_model_path = os.path.join(USER_FOLDER_PATH, user_name, f"{user_name}_model.pt")
    torch.save(user_model, user_model_path)
    print(f"User-specific model for {user_name} saved at {user_model_path}")

def train_model(dataset):
    # Placeholder for model training (replace with actual model training code)
    model = torch.nn.Sequential(torch.nn.Linear(dataset.shape[1], 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
    return model

if __name__ == "__main__":
    # Example usage
    user_name = "new_user"
    user_dataset = torch.randn(100, 10)  # Example user dataset
    process_user_dataset(user_name, user_dataset)
