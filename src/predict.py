"""
This script classifies the unseen data from the headset during real-time EEG data acquisition.

Input:
    - Real-time EEG data from the headset
    
Output:
    - Predicted class label for the real-time EEG data
    - Confidence score of the prediction
    - Visual feedback for the user

"""

from data_processing.livestream_processing import processing_livestreamed_signal

def predict(dataset_path, model_path):
    """
    Function to classify the unseen data from the headset during real-time EEG data acquisition.
    """
    # Log the start of the process
    logger.info("Predicting...")
    
    try:
        data_to_predict = processing_livestreamed_signal(input_folder, output_folder_filtered_files, output_folder_tensor_dataset)
        logger.info("Prediction successful.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")