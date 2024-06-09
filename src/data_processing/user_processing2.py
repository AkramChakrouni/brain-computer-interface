"""
    This script contains functions to process a user specific dataset, 
    to create a tensor dataset and labels tensor.
"""

import numpy as np
import pandas as pd
import torch
import os
from datetime import datetime
from scipy.signal import filtfilt, butter
import sys

# Add the root directory of the project to the PYTHONPATH, to import shared functions
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(root_dir)
from src.data_processing.shared_processing import z_score, apply_wavelet_transform

def load_dataset(file_path):
    """
    Load the dataset from the given file path, and manipulate to the required format.
    
    Arguments:
        - file_path -- path to the input file.
        
    Returns:
        - df_imm -- the manipulated DataFrame. 
    """
    # Read the input file
    df_in = pd.read_csv(file_path)
    
    # Get the header row
    header_row = list(df_in.columns)

    # Create a DataFrame from the header row
    header_df = pd.DataFrame([header_row], columns=df_in.columns)

    # Concatenate the header row with the original DataFrame
    df_imm = pd.concat([header_df, df_in], ignore_index=True)

    # Reset column names
    df_imm.columns = range(df_imm.shape[1])

    # Remove the first column
    df_imm = df_imm.drop(0, axis=1)

    # Shape is (n_timepoints, n_channels + target)
    return df_imm

def apply_bandpass_filter(signal, b, a):
    """
    Apply a bandpass filter to the given signal using the provided filter coefficients.

    Arguments:
        - signal -- the input signal.
        - b -- the numerator coefficients of the filter.
        - a -- the denominator coefficients of the filter.

    Returns:
        - filtered_signal -- the filtered signal.
    """
    return filtfilt(b, a, signal)

def filter_data(df_in):
    """
    Apply a bandpass filter to the input DataFrame.
    
    Arguments:
        - df_in -- the input DataFrame.
        
    Returns:
        - df_filtered -- the filtered DataFrame.
    """
    # Remove the last column, to shape (n_channels, n_timepoints)
    df_imm = df_in.iloc[:, :-1].T
    
    # Last column is the target_column
    target_column = df_in.iloc[:, -1]
    
    # Change to float32
    df_imm = df_imm.astype(np.float32)
    
    # Filter parameters
    l_freq = 8
    h_freq = 30
    order = 5
    fs = 250

    nyquist = 0.5 * fs
    low = l_freq / nyquist
    high = h_freq / nyquist

    # Calculate filter coefficients
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter to each row independently
    for i in range(df_imm.shape[0]):
        df_imm.iloc[i] = apply_bandpass_filter(df_imm.iloc[i], b, a)
        
    # Add the target_column column back, with no column name
    df_filtered = df_imm.T
    df_filtered[''] = target_column
    
    # reset column names
    df_filtered.columns = range(df_filtered.shape[1])
    
    # Shape is (n_timepoints, n_channels + target_column)
    return df_filtered   

def split_dataset(df):
    """
        Split the dataset into multiple datasets based on the target column.
        
        Arguments:
            - df -- the input DataFrame.
            
        Returns:
            - datasets -- a list of DataFrames, each containing a unique target value.
    """
    current_sequence = df.iloc[0, -1]  # Get the first value in the last column
    current_dataset = df.iloc[[0]]  # Initialize the first dataset
    datasets = []

    for i in range(1, len(df)):
        if df.iloc[i, -1] == current_sequence:  # If the sequence continues
            current_dataset = pd.concat([current_dataset, df.iloc[[i]]])
        else:  # If the sequence changes
            datasets.append(current_dataset)
            current_sequence = df.iloc[i, -1]
            current_dataset = df.iloc[[i]]

    datasets.append(current_dataset)  # Append the last dataset
    return datasets

def get_epochs_and_labels(dataset):
    """
    Convert the dataset to epochs and labels.
    
    Arguments:
        - dataset -- the input DataFrame.
        
    Returns:
        - epochs -- the EEG data in the form of epochs.
        - labels -- the corresponding labels for each epoch.
    """
    # Get only the data, by removing the last column
    data = dataset.drop(dataset.columns[-1], axis=1).T

    # Reset column names
    data.columns = range(data.shape[1])

    # Split data in epochs of 0.5 second, sampled at 250 Hz, so 125 samples per epoch
    epoch_length = 126  # 0.5 seconds * 250 Hz
    n_samples = data.shape[1]

    # Calculate the number of full epochs
    n_epochs = n_samples // epoch_length

    # Discard remaining samples that do not fit into a full epoch
    data = data.iloc[:, :n_epochs * epoch_length]

    # Properly slice the data into epochs
    epochs = [data.iloc[:, i*epoch_length:(i+1)*epoch_length] for i in range(n_epochs)]

    # Convert list of DataFrames to 3D numpy array
    epochs = np.stack([epoch.values for epoch in epochs], axis=0)

    # Get the category column
    category = dataset[dataset.columns[-1]]

    # Label is 0 is the first index of category 'L' and 1 if the first index of category 'R'
    label = 0 if category.iloc[0] == 'L' else 1

    # Create a label array with the same length as the number of epochs
    labels = np.full((n_epochs,), label)

    return epochs, labels

def process_user_dataset(file_path):
    """
    Process a user specific dataset to create a tensor dataset and labels tensor.

    Arguments:
        - file_path (str): The path to the input file.
    """
    # Load the dataset
    df_in = load_dataset(file_path)

    # Filter the data
    data_filtered = filter_data(df_in)

    # Split the dataset into sequences
    datasets = split_dataset(data_filtered)

    # Create empty lists to store the transformed epochs and their labels
    all_transformed_epochs = []
    all_labels = []

    # Process each dataset
    for dataset in datasets:
        # Get the epochs and labels for the current dataset
        epochs, labels = get_epochs_and_labels(dataset)
        
        # Process each epoch
        for epoch, label in zip(epochs, labels):
            
            # Z-score normalize each epoch
            epoch_norm = z_score(epoch)
        
            # Apply wavelet transformation, on the normalized epoch
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
        
    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time as a string for the dataset name
    time_stamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Specify the output folder
    output_folder = "users"

    # Get the filename without the extension
    user_name = os.path.splitext(os.path.basename(file_path))[0]

    # Make a subfolder for the tensor dataset in the output folder, name it with the current date and time
    subfolder = os.path.join(output_folder, user_name)    

    # Create the subfolder if it does not exist
    os.makedirs(subfolder, exist_ok=True)

    # Save the combined dataset to the subfolder with the specified name
    tensor_dataset_file_name = os.path.join(subfolder, f"dataset_{time_stamp}.pt")
    torch.save(tensor_dataset, tensor_dataset_file_name)

    labels_tensor_file_name = os.path.join(subfolder, f"labels_{time_stamp}.pt")
    torch.save(labels_tensor, labels_tensor_file_name)
    
    return None