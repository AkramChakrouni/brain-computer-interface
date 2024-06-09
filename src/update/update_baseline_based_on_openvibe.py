"""
    This script will monitor the users folder for new subfolders 
    and update the base dataset in data/base by adding the new user's dataset.
"""
import os
import time
import datetime
import torch
import sys

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from 'data_processing' module
from make_dataset.make_openvibe_dataset import make_openvibe_dataset

RAW_DIR = 'data/openvibe/raw'
PROCESSED_DIR = 'data/openvibe/processed'
INTERIM_DIR = 'data/openvibe/interim'
BASE_DIR = 'data/base'

def get_latest_baseline():
    subfolders = [f.path for f in os.scandir(BASE_DIR) if f.is_dir()]
    if not subfolders:
        print("No baseline subfolders found.")
        return None, None, None
    
    latest_subfolder = max(subfolders, key=os.path.getmtime)
    timestamp = os.path.basename(latest_subfolder)
    try:
        data_tensor = torch.load(os.path.join(latest_subfolder, f'dataset_{timestamp}.pt'))
        labels_tensor = torch.load(os.path.join(latest_subfolder, f'labels_{timestamp}.pt'))
    except FileNotFoundError as e:
        print(f"Error loading tensors from {latest_subfolder}: {e}")
        return None, None, latest_subfolder

    return data_tensor, labels_tensor, latest_subfolder

def update_baseline(new_data, new_labels):
    data_tensor, labels_tensor, latest_subfolder = get_latest_baseline()
    if data_tensor is None or labels_tensor is None:
        updated_data = new_data
        updated_labels = new_labels
    else:
        updated_data = torch.cat((data_tensor, new_data), dim=0)
        updated_labels = torch.cat((labels_tensor, new_labels), dim=0)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_baseline_subfolder = os.path.join(BASE_DIR, timestamp)
    os.makedirs(new_baseline_subfolder, exist_ok=True)
    
    torch.save(updated_data, os.path.join(new_baseline_subfolder, f'dataset_{timestamp}.pt'))
    torch.save(updated_labels, os.path.join(new_baseline_subfolder, f'labels_{timestamp}.pt'))

def monitor_directory():
    processed_files = set()
    
    while True:
        current_files = set(os.listdir(RAW_DIR))
        new_files = current_files - processed_files
        
        for file in new_files:
            if file.endswith('.edf'):
                file_path = os.path.join(RAW_DIR, file)
                new_data, new_labels = make_openvibe_dataset(RAW_DIR, INTERIM_DIR, PROCESSED_DIR)
                
                if new_data is not None and new_labels is not None:
                    update_baseline(new_data, new_labels)
                    processed_files.add(file)
                else:
                    print(f"Processing failed for file: {file_path}")
                
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    monitor_directory()