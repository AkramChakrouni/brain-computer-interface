"""
    This script creates a user specific dataset from raw EEG data files,
    and converts it to a tensor dataset.
"""

import os
import sys
import logging
import contextlib
import mne

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from 'data_processing' module
from data_processing.user_processing2 import process_user_dataset

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

def make_use_dataset(input_folder, set):
    """
    Function to convert raw EEG data files to filtered files and then to a combined TensorDataset.
    """

    if (set == 1):
        
        # Log the start of the process
        logger.info("Creating your dataset...")
        
        try:
            process_user_dataset(input_folder)
            logger.info("Your dataset has been successfully created.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    
    input_folder = "data/calibration/raw/training_right.csv"

    make_use_dataset(input_folder, set = 1)
