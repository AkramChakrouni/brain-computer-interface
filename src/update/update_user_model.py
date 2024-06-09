"""
    This script will prompt existing users to update their model if the base dataset has grown.
    It will be called by the NewFolderHandler in update_base_dataset.py when a new user folder is detected.
"""

import os
import torch
import shutil
from datetime import datetime

USER_FOLDER_PATH = 'users'
DATA_FOLDER_PATH = 'data/base'
BASE_DATASET_PREFIX = 'baseline_dataset_'
TIMESTAMP_FORMAT = '%Y-%m-%d_%H-%M-%S'

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

def prompt_user_to_update(user_name):
    response = input(f"New update available. Update your model, {user_name}? (yes/no): ")
    if response.lower() == 'yes':
        update_user_model(user_name)

def update_user_model(user_name):
    latest_base_dataset_path = get_latest_base_dataset_path()
    base_dataset = torch.load(latest_base_dataset_path)
    
    user_folder_path = os.path.join(USER_FOLDER_PATH, user_name)
    user_dataset_path = os.path.join(user_folder_path, f"{user_name}_dataset.pt")
    user_dataset = torch.load(user_dataset_path)
    
    # Combine datasets with higher weight on user dataset
    combined_dataset = torch.cat((base_dataset, user_dataset * 2))  # Example: Double the user dataset weight

    # Placeholder for model training (replace with actual model training code)
    user_model = train_model(combined_dataset)
    
    # Save the updated model
    user_model_path = os.path.join(USER_FOLDER_PATH, user_name, f"{user_name}_model.pt")
    torch.save(user_model, user_model_path)
    print(f"User-specific model for {user_name} updated and saved at {user_model_path}")

def train_model(dataset):
    # Placeholder for model training (replace with actual model training code)
    model = torch.nn.Sequential(torch.nn.Linear(dataset.shape[1], 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
    return model

if __name__ == "__main__":
    # Example usage
    existing_user_name = "existing_user"
    prompt_user_to_update(existing_user_name)
