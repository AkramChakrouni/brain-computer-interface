"""
    This module contains functions that are specific to processing the livestreamed data that is received from the OpenBCI 
    headset.

    Returns:
        - livestreamed_tensor (torch.Tensor): 4D tensor containing the wavelet transformed and z-score normalized EEG data,
          shaped as (n_epochs, n_channels, n_scales, n_time_points).
"""

import pandas as pd
import torch
from scipy.signal import butter, filtfilt
import sys
import os

# Add the root directory of the project to the PYTHONPATH, to import shared functions
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(root_dir)
from src.data_processing.shared_processing import z_score, apply_wavelet_transform

def csv_to_df(file_path):
    # Read the csv file
    df_in = pd.read_csv(file_path)

    # Get the header row
    header_row = list(df_in.columns)

    # Convert the header row to integer values
    header_row = list(map(int, header_row))

    # Create a DataFrame from the header row
    header_df = pd.DataFrame([header_row], columns=df_in.columns)

    # Concatenate the header row with the original DataFrame
    df_imm = pd.concat([header_df, df_in], ignore_index=True)

    # Reset column names
    df_imm.columns = range(df_imm.shape[1])

    # Remove the first column
    df_imm = df_imm.drop(0, axis=1)

    # Transpose the DataFrame, to have the channels as rows and time points as columns
    df_out = df_imm.T
    
    return df_out

def apply_bandpass_filter(signal, b, a):
    return filtfilt(b, a, signal)

def processing_livestreamed_signal(file_path):
    # Load the data, in dataframe format
    data_raw = csv_to_df(file_path)
    
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
    for i in range(data_raw.shape[0]):
        data_raw.iloc[i] = apply_bandpass_filter(data_raw.iloc[i], b, a)
        
    # Convert filtered data back to DataFrame
    data_filtered = pd.DataFrame(data_raw)
    
    # Split data into segments of 0.5 seconds, for epoching
    epoch_size = int(fs * 0.5)  # Number of samples in 0.5 seconds
    epochs = [data_filtered.iloc[:, i:i+epoch_size] for i in range(0, len(data_filtered.columns), epoch_size)]
    
    # Create an empty list to store the transformed epochs
    transformed_epochs = []
    
    # Z-score normalization and wavelet transform for each segment and stack them into a tensor (4D array), 
    # Loop through each epoch
    for epoch in epochs:
        # Z-score each epoch
        epoch_norm = z_score(epoch)
        
        # Apply wavelet transformation
        wavelet_tensor = apply_wavelet_transform(epoch_norm)
        
        # Append the transformed epoch to the list
        transformed_epochs.append(wavelet_tensor)
        
    # Convert the list of epochs into a tensor dataset
    livestream_data_tensor = torch.tensor(transformed_epochs, dtype=torch.float32)  # Ensure the tensor is float32
     
    return livestream_data_tensor