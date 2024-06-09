"""
    This module contains functions that are specific to processing external EEG data.
"""

import os
from datetime import datetime
import numpy as np
import mne
import torch
from torch.utils.data import TensorDataset
import logging
import sys

# Add the root directory of the project to the PYTHONPATH, to import shared functions
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(root_dir)

from src.data_processing.shared_processing import z_score, apply_wavelet_transform

# Configure logger for this module
logger = logging.getLogger(__name__)

def filter_eeg_files(input_folder, output_folder):
    """
    Process all EDF files ending with 'R03', 'R07', or 'R11' in the input folder.
    Apply a 5th order IIR Butterworth filter, resample the data to 250 Hz, rename the channels to standard names,
    and save each filtered file with a '_filtered' suffix in the output folder.

    Arguments:
        - Input_folder (str): Path to the folder containing the EDF files to be filtered.
        - Output_folder (str): Path to the folder where the filtered files will be saved.
        
    Returns:
        - FIF files: Filtered EEG data files saved in the output folder.
        - filtered_files_list (list): List of paths to the filtered files.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define the filter parameters
    l_freq=8 
    h_freq=30
    order=5

    # Define IIR filter parameters
    iir_params = dict(order=order, ftype='butter')
    
    # List to collect paths of the filtered files
    filtered_files_list = []

    # Iterate through all subfolders in the output folder
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith('.edf'):
                # Check if file ends with R03, R07, or R11
                if file_name.endswith(('R03.edf', 'R07.edf', 'R11.edf')):
                    # Construct full file path
                    file_path = os.path.join(root, file_name)
                    
                    # Load the EDF file
                    raw = mne.io.read_raw_edf(file_path, preload=True)
                    
                    # Resample the data to 250 Hz
                    raw.resample(250)
                    
                    # Select specific channels
                    raw.pick_channels(['Fp1.', 'Fp2.', 'F2..', 'F4..', 'C3..', 'C4..', 'P3..', 'P4..'], ordered=True)
                    
                    # Rename the channels
                    raw.rename_channels(mapping={'Fp1.': 'FP1', 'Fp2.': 'FP2', 'F2..': 'F2', 'F4..': 'F4', 
                                                 'C3..': 'C3', 'C4..': 'C4', 'P3..': 'P3', 'P4..': 'P4'})
                    
                    # Apply the 5th order IIR Butterworth filter
                    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=iir_params)
                    
                    # Define the output file path
                    output_file_name = f"{os.path.splitext(file_name)[0]}_filtered.fif"
                    output_file_path = os.path.join(output_folder, output_file_name)
                    
                    # Save the filtered data
                    raw.save(output_file_path, overwrite=True)  
                    
                    # Add the output file path to the list, for return
                    filtered_files_list.append(output_file_path)
    
    # Return the list of filtered file paths
    return filtered_files_list

def get_epochs_and_labels_external(fif_file):
    """
    Load the EEG data from a FIF file, extract the epochs and labels, and return them.

    Arguments:
        - FIF_file (str): Path to the FIF file containing the EEG data.

    Returns:
        - Epochs (mne.Epochs): EEG epochs extracted from the FIF file.
        - Labels (numpy.ndarray): Labels corresponding to the epochs.
    """
    # Load your EEG data
    raw = mne.io.read_raw_fif(fif_file, preload=True)
    
    # Get the events from the annotations
    events, _ = mne.events_from_annotations(raw)

    # T1 is left hand, T2 is right hand
    event_id = {'T1': 2, 'T2': 3}

    # Epochs start 0s before the trigger and end 0.5s after
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=0.5, baseline=None, preload=True)

    # Get the labels of the epochs
    labels = epochs.events[:, -1]

    # Change the labels to 0 and 1
    labels[labels == 2] = 0
    labels[labels == 3] = 1
    
    return epochs, labels

def file_to_tensor(input_folder, output_folder_filtered_files, output_folder_tensor_dataset):
    """
    Load the EEG data from a list of FIF files, apply z-score normalization and wavelet transform to each epoch.
    
    Arguments:
        - Input_folder (str): Path to the folder containing the FIF files.
        - Output_folder_filtered_files (str): Path to the folder where the filtered files will be saved.
        - Output_folder_tensor_dataset (str): Path to the folder where the combined dataset will be saved.
    """
    # Set logger level to CRITICAL to suppress logs
    logger.setLevel(logging.CRITICAL)
    
    # Create a list to hold the transformed epochs for all files
    all_transformed_epochs = []
    
    # Create a list to hold the corresponding labels for all epochs
    all_labels = []
    
    # Filter the EEG files
    filtered_files = filter_eeg_files(input_folder, output_folder_filtered_files)
    
    # Process each filtered file
    for fif_file in filtered_files:
        # Load the EEG data from the FIF file and extract the epochs and labels
        epochs, labels = get_epochs_and_labels_external(fif_file)
        
        # Process each epoch
        for epoch, label in zip(epochs, labels):
            # Z-score each epoch
            epoch_norm = z_score(epoch)
            
            # Apply wavelet transformation
            epoch_wavelet = apply_wavelet_transform(epoch_norm)
            
            # Append the transformed epoch and its label to the lists
            all_transformed_epochs.append(epoch_wavelet)
            all_labels.append(label)

    # Convert lists to NumPy arrays before creating tensors
    all_transformed_epochs = np.array(all_transformed_epochs)
    all_labels = np.array(all_labels)

    # Convert the NumPy arrays to tensors
    tensor_dataset = torch.tensor(all_transformed_epochs, dtype=torch.float)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    # Combine the tensor dataset and labels tensor into a TensorDataset
    dataset = TensorDataset(tensor_dataset, labels_tensor)
    
    # Get the current date and time
    current_datetime = datetime.now()
    
    # Format the date and time as a string for the dataset name
    dataset_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save the combined dataset to the output folder with the specified name
    output_file = os.path.join(output_folder_tensor_dataset, f"baseline_dataset_{dataset_name}.pt")
    torch.save(dataset, output_file)
    
    return None