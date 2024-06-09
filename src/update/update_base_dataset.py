"""
    This script will monitor the users folder for new subfolders 
    and update the base dataset in data/base by adding the new user's dataset.

"""

import os
import time
import torch
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DATA_FOLDER_PATH = 'data/base'
USER_FOLDER_PATH = 'users'
BASE_DATASET_PREFIX = 'base_dataset_'
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'  # Assuming the format is like base_dataset_20230101_123456.pt

class NewFolderHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            user_folder_path = event.src_path
            print(f"New folder detected: {user_folder_path}")
            self.update_base_dataset(user_folder_path)

    def get_latest_base_dataset_path(self):
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

    def update_base_dataset(self, user_folder_path):
        try:
            latest_base_dataset_path = self.get_latest_base_dataset_path()
            base_dataset = torch.load(latest_base_dataset_path)
            
            # Assuming the user dataset is a single .pt file in the new folder
            for filename in os.listdir(user_folder_path):
                if filename.endswith(".pt"):
                    user_dataset_path = os.path.join(user_folder_path, filename)
                    user_dataset = torch.load(user_dataset_path)
                    
                    # Append the user dataset to the base dataset
                    updated_dataset = torch.cat((base_dataset, user_dataset))
                    
                    # Create new base dataset filename with current timestamp
                    new_timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
                    new_base_dataset_path = os.path.join(DATA_FOLDER_PATH, f"{BASE_DATASET_PREFIX}{new_timestamp}.pt")
                    
                    # Save the updated base dataset
                    torch.save(updated_dataset, new_base_dataset_path)
                    print(f"Base dataset updated with {user_dataset_path} and saved as {new_base_dataset_path}")
                    break
        except Exception as e:
            print(f"Error updating base dataset: {e}")

if __name__ == "__main__":
    event_handler = NewFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, path=USER_FOLDER_PATH, recursive=False)
    observer.start()
    print("Monitoring started...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

