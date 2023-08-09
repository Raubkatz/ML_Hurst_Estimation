import sys
import numpy as np
from scipy.linalg import svd
from scipy.stats import entropy
import pickle
import os
from copy import deepcopy as dc

def calculate_hurst_exponent_CatBoost(time_series, stochastic_process='fBm'):
    """
    Calculate the Hurst exponent from an input time series using pre-trained machine learning models.

    This function finds the appropriate model based on the length of your time series (considered as the window size),
    divides your time series into the appropriate windows, and then feeds each window into the model to get a Hurst
    exponent prediction for each window. If the window size is above 100, the model trained with a window size of 100 is used.
    For a window size between 51 and 100, the model trained with a window size of 50 is used, and so on.

    Parameters:
    time_series (numpy array): The input time series data.
    stochastic_process (string): on which stochastic process the model was trained, allows for "both", "fLm" and "fBm"

    Returns:
    float: The average Hurst exponent of the time series.
    """

    def min_max_scale(data):
        """
        Scale the data to the unit interval [0,1].
        """
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)

    def load_model(window_size):
        """
        Load the pre-trained model based on window size.
        """
        MODELS_FOLDER = 'ML_MODELS_HURST'
        if stochastic_process == "fLm":
            if window_size >= 100:
                model_folder = 'CatBoost_final_100_fLm'
                window_size = 100
            elif window_size >= 50:
                model_folder = 'CatBoost_final_50_fLm'
                window_size = 50
            elif window_size >= 25:
                model_folder = 'CatBoost_final_25_fLm'
                window_size = 25
            elif window_size >= 10:
                window_size = 10
                model_folder = 'CatBoost_final_10_fLm'
            else:
                print('No fitting model available for this window size... exiting')
                sys.exit()
        elif stochastic_process == "fBm":
            if window_size >= 100:
                window_size = 100
                model_folder = 'CatBoost_final_100_fBm'
            elif window_size >= 50:
                window_size = 50
                model_folder = 'CatBoost_final_50_fBm'
            elif window_size >= 25:
                window_size = 25
                model_folder = 'CatBoost_final_25_fBm'
            elif window_size >= 10:
                window_size = 10
                model_folder = 'CatBoost_final_10_fBm'
            else:
                print('No fitting model available for this window size... exiting')
                sys.exit()
        else:
            if window_size >= 100:
                window_size = 100
                model_folder = 'CatBoost_final_100_both'
            elif window_size >= 50:
                window_size = 50
                model_folder = 'CatBoost_final_50_both'
            elif window_size >= 25:
                window_size = 25
                model_folder = 'CatBoost_final_25_both'
            elif window_size >= 10:
                window_size = 10
                model_folder = 'CatBoost_final_10_both'
            else:
                print('No fitting model available for this window size... exiting')
                sys.exit()

        model_path = os.path.join(MODELS_FOLDER, model_folder, 'CatBoost.clf')
        model = pickle.load(open(model_path, 'rb'))

        return model, window_size

    def get_windows(time_series, window_size):
        """
        Split the time series into windows of the specified size.
        """
        return [time_series[i: i + window_size] for i in range(len(time_series) - window_size + 1)]

    # Load the model based on window size
    model, window_size = load_model(len(time_series))

    # Get windows from the time series
    windows = get_windows(time_series, window_size)

    # Calculate Hurst exponent for each window and average them
    hurst_exponents = []

    for window in windows:

        scaled_window = min_max_scale(window)

        # Model expects a 2D array, so we need to add an extra dimension to the window
        #window2d = dc(scaled_window)
        window_2d = np.reshape(scaled_window, (1, -1))

        # Use the model to predict the Hurst exponent for this window
        hurst_exponent = model.predict(window_2d)

        hurst_exponents.append(hurst_exponent)

    # Calculate the average Hurst exponent
    avg_hurst_exponent = np.mean(hurst_exponents)

    return avg_hurst_exponent


def calculate_hurst_exponent_CatBoost_sliding_window(time_series, window_size, stochastic_process='fBm'):

    def get_windows(time_series, window_size):
        """
        Split the time series into windows of the specified size.
        """
        return [time_series[i: i + window_size] for i in range(len(time_series) - window_size + 1)]

    if window_size<10:
        print(f'Window size of {window_size} too small, 10 data points is the minimum.')

    # Get windows from the time series
    windows = get_windows(time_series, window_size)

    # Calculate Hurst exponent for each window
    hurst_exponents = []

    for window in windows:
        hurst_exponent = calculate_hurst_exponent_CatBoost(window, stochastic_process=stochastic_process)
        hurst_exponents.append(hurst_exponent)

    return hurst_exponents