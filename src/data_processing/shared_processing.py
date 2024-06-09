"""
    This module contains functions that are shared across different data processing scripts.
"""

import numpy as np
import pywt

def z_score(epoch):
    """
    Apply z-score normalization to each channel of the EEG data.
    
    Arguments:
        - Epoch (numpy.ndarray): EEG data to be normalized.
        
    Returns:
        - Z-scored epoch (numpy.ndarray): Normalized EEG data.
    """    
    # Apply z-score normalization to each channel, saved in epoch
    for i in range(epoch.shape[0]):
        channel_epoch = epoch[i, :]
        mean = np.mean(channel_epoch)
        std = np.std(channel_epoch)
        z_scored_epoch = (channel_epoch - mean) / std
        epoch[i, :] = z_scored_epoch
    
    return epoch

def frequency_to_scale(freq, wavelet='morl', sampling_rate=250):
    """
    Convert frequency values to scales for continuous wavelet transform (CWT).

    Arguments:
        = freq (array): Array of frequency values.
        wavelet (str, optional): Type of wavelet to use. Defaults to 'morl'.
        sampling_rate (int): Sampling rate of the EEG data. Defaults to 250 Hz.

    Returns:
        - scales (array): Array of scales corresponding to the input frequencies.
    """
    # For the Morlet wavelet, scales are inversely proportional to frequency
    center_freq = pywt.central_frequency(wavelet)
    return center_freq / (freq / sampling_rate)

def apply_wavelet_transform(data_norm, wavelet='morl', freq_range=(8, 30), sampling_rate=250):
    """
    Apply wavelet transform to EEG data.
    
    Arguments:
        - data_norm (ndarray): 2D array with shape (n_channels, n_time_points)
        - wavelet (str): Wavelet type (default 'morl')
        - freq_range (tuple): Frequency range for the CWT (default (8, 30) Hz)
        - sampling_rate (int): Sampling rate of the EEG data (default 250 Hz)
    
    Returns:
    ndarray: 3D array with shape (n_channels, n_scales, n_time_points)
    """
    n_channels, n_times = data_norm.shape
    # Define scales based on the desired frequency range
    scales = frequency_to_scale(np.arange(freq_range[0], freq_range[1]+1), wavelet=wavelet, sampling_rate=sampling_rate)
    
    coeffs = []
    for i in range(n_channels):
        # Compute the wavelet transform coefficients
        coef, _ = pywt.cwt(data_norm[i], scales=scales, wavelet=wavelet)
        coeffs.append(coef)
    
    # Stack coefficients to form a 3D tensor
    coeffs_done = np.stack(coeffs, axis=0)
    
    return coeffs_done
