"""
    This module creates the filtered files and the combined TensorDataset from the OpenVibe EEG data.
"""

import os
import sys
import logging
import contextlib
import mne

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from 'data_processing' module
from data_processing.openvibe_processing import file_to_tensor

# Suppress `mne` library logging
mne.set_log_level('CRITICAL')

# Suppress stdout and stderr
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def make_openvibe_dataset(input_folder, output_folder_filtered_files, output_folder_tensor_dataset):
    """
    Function to convert raw EEG data files to filtered files and then to a combined TensorDataset.
    """
    # Log the start of the process
    logger.info("Loading...")
    
    # Ensure output directories exist
    os.makedirs(output_folder_filtered_files, exist_ok=True)
    os.makedirs(output_folder_tensor_dataset, exist_ok=True)
    
    try:
        tensor, label = file_to_tensor(input_folder, output_folder_filtered_files, output_folder_tensor_dataset)
        logger.info("Dataset updated successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        
    return tensor, label

if __name__ == "__main__":
    # Input and output folders
    input_folder = "data/openvibe/raw"
    output_folder_filtered_files = "data/openvibe/interim"
    output_folder_tensor_dataset = "data/openvibe/processed"
    make_openvibe_dataset(input_folder, output_folder_filtered_files, output_folder_tensor_dataset)