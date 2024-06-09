"""
    This script is used to convert the raw EEG data files downloaded to 
    the folder data/external/raw to filtered files in the folder data/external/interim, 
    and then to a combined TensorDataset in the folder data/base.
"""

import os
import sys
import logging
import contextlib
import mne

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from 'data_processing' module
from data_processing.external_processing import file_to_tensor

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

def make_baseline_dataset():
    """
    Function to convert raw EEG data files to filtered files and then to a combined TensorDataset.
    """
    # Log the start of the process
    logger.info("Loading...")
    
    # Input and output folders
    input_folder = "data/external/raw"
    output_folder_filtered_files = "data/external/processed"
    output_folder_tensor_dataset = "data/base"
    
    # Ensure output directories exist
    os.makedirs(output_folder_filtered_files, exist_ok=True)
    os.makedirs(output_folder_tensor_dataset, exist_ok=True)
    
    try:
        file_to_tensor(input_folder, output_folder_filtered_files, output_folder_tensor_dataset)
        logger.info("Dataset creation successful.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    make_baseline_dataset()
