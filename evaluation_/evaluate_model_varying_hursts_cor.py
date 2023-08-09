import pickle
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import preprocessing
import hurst
import pandas as pd
from copy import deepcopy as dc
import numpy as np
from matplotlib import pyplot as plt
import nolds
import flm
import neurokit

from datetime import datetime
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn

import numpy as np
import os
import random
#import tensorflow as tf


def reset_seeds(seed,reset_graph_with_backend=None):
    #if reset_graph_with_backend is not None:
    #    K = reset_graph_with_backend
    #    K.clear_session()
    #    tf.compat.v1.reset_default_graph()
    #    print("KERAS AND TENSORFLOW GRAPHS RESET")  # optional
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    #tf.compat.v1.set_random_seed(seed)
    print("RANDOM SEEDS RESET")  # optional



def hurst_stackoverflow(ts):
    """Returns the Hurst Exponent of the time series vector ts
    Code from https://stackoverflow.com/questions/39488806/hurst-exponent-in-python

    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses
    # standard deviation and then make a root of it?
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


def tensorize_2(array):
    tensor = list()
    inner_tensor = list()
    for i in range(len(array)):
        inner_tensor.append(array[i])
    tensor.append(inner_tensor)
    return tensor


def tensorize(array):
    tensor = list()
    for i in range(len(array)):
        sublist = list()
        sublist.append(array[i])
        tensor.append(sublist)
    return np.array(tensor)

def clean_mean(np_array):
    """
    :param np_array: array to be averaged
    :return: average, but without values above or below zero
    """
    out_list = list()
    for i in range(len(np_array)):
        if np_array[i] != np_array[i]:
            print('nan encountered')
        else:
            if ((np_array[i] <= 1.0) and (np_array[i] >= 0.0)):
                print("correct value " + str(np_array[i]))
                out_list.append(np_array[i])
            else:
                if np_array[i] > 1.0: out_list.append(1.0)
                elif np_array[i] < 0.0: out_list.append(0.0)
    return dc(np.mean(np.array(out_list, dtype=float)))

def clean_std(np_array):
    """
    :param np_array: array to be averaged
    :return: standard deviation, but without values above or below zero
    """
    out_list = list()
    for i in range(len(np_array)):
        if np_array[i] != np_array[i]:
            print('nan encountered')
        else:
            if ((np_array[i] <= 1.0) and (np_array[i] >= 0.0)):
                print("correct value " + str(np_array[i]))
                out_list.append(np_array[i])
            else:
                if np_array[i] > 1.0: out_list.append(1.0)
                elif np_array[i] < 0.0: out_list.append(0.0)
    return dc(np.std(np.array(out_list, dtype=float)))

def correct_val(H_est):
    if H_est<0:
        return 0.0
    if H_est>1:
        return 1.0
    else: return H_est

model_input = 25
stat_count = 1000
step_size = 10 #default = 10

fLm = False
alpha = 0.5 #0.5 missing
cor = True
rand_seed = 42

reset_seeds(rand_seed)


#missing 25, 10
#0.5 1.0 alpha missing
import sys


MLP = True
LightGBM = True
CatBoost = True
AdaBoost = True
Lasso = True
Ridge = True
XGBoost = False # XGBoost is not working due to an error in the implementation when loading a pretrained model.




#add is to label the output files
add = "_" + str(model_input) + "_" + str(stat_count)
if MLP:
    add = add + "_MLP"
if LightGBM:
    add = add + "_LightGBM"
if CatBoost:
    add = add + "_CatBoost"
if AdaBoost:
    add = add + "_AdaBoost"
if Lasso:
    add = add + "_Lasso"
if Ridge:
    add = add + "_Ridge"
if XGBoost:
    add = add + "_XGBoost"

if fLm:
    add = add + "_fLm" + "_" + str(alpha)

if cor: add = add + "_cor_final_" + str(rand_seed)

if MLP:
    if model_input >= 100:
        loaded_model_MLP_100_fLm = pickle.load(open("./models/MLP_final_100_fLm/MLP.clf", 'rb'))
        loaded_model_MLP_100_both = pickle.load(open("./models/MLP_final_100_both/MLP.clf", 'rb'))
        loaded_model_MLP_100_fBm = pickle.load(open("./models/MLP_final_100_fBm/MLP.clf", 'rb'))
    if model_input == 50:
        loaded_model_MLP_50_both = pickle.load(open("./models/MLP_final_50_both/MLP.clf", 'rb'))
        loaded_model_MLP_50_fBm = pickle.load(open("./models/MLP_final_50_fBm/MLP.clf", 'rb'))
        loaded_model_MLP_50_fLm = pickle.load(open("./models/MLP_final_50_fLm/MLP.clf", 'rb'))
    if model_input == 25:
        loaded_model_MLP_25_fLm = pickle.load(open("./models/MLP_final_25_fLm/MLP.clf", 'rb'))
        loaded_model_MLP_25_both = pickle.load(open("./models/MLP_final_25_both/MLP.clf", 'rb'))
        loaded_model_MLP_25_fBm = pickle.load(open("./models/MLP_final_25_fBm/MLP.clf", 'rb'))
    if model_input == 10:
        loaded_model_MLP_10_fBm = pickle.load(open("./models/MLP_final_10_fBm/MLP.clf", 'rb'))
        loaded_model_MLP_10_both = pickle.load(open("./models/MLP_final_10_both/MLP.clf", 'rb'))
        loaded_model_MLP_10_fLm = pickle.load(open("./models/MLP_final_10_fLm/MLP.clf", 'rb'))

if CatBoost:
    if model_input >= 100:
        loaded_model_CatBoost_100_fLm = pickle.load(open("./models/CatBoost_final_100_fLm/CatBoost.clf", 'rb'))
        loaded_model_CatBoost_100_both = pickle.load(open("./models/CatBoost_final_100_both/CatBoost.clf", 'rb'))
        loaded_model_CatBoost_100_fBm = pickle.load(open("./models/CatBoost_final_100_fBm/CatBoost.clf", 'rb'))
    if model_input == 50:
        loaded_model_CatBoost_50_both = pickle.load(open("./models/CatBoost_final_50_both/CatBoost.clf", 'rb'))
        loaded_model_CatBoost_50_fBm = pickle.load(open("./models/CatBoost_final_50_fBm/CatBoost.clf", 'rb'))
        loaded_model_CatBoost_50_fLm = pickle.load(open("./models/CatBoost_final_50_fLm/CatBoost.clf", 'rb'))
    if model_input == 25:
        loaded_model_CatBoost_25_fLm = pickle.load(open("./models/CatBoost_final_25_fLm/CatBoost.clf", 'rb'))
        loaded_model_CatBoost_25_both = pickle.load(open("./models/CatBoost_final_25_both/CatBoost.clf", 'rb'))
        loaded_model_CatBoost_25_fBm = pickle.load(open("./models/CatBoost_final_25_fBm/CatBoost.clf", 'rb'))
    if model_input == 10:
        loaded_model_CatBoost_10_fBm = pickle.load(open("./models/CatBoost_final_10_fBm/CatBoost.clf", 'rb'))
        loaded_model_CatBoost_10_both = pickle.load(open("./models/CatBoost_final_10_both/CatBoost.clf", 'rb'))
        loaded_model_CatBoost_10_fLm = pickle.load(open("./models/CatBoost_final_10_fLm/CatBoost.clf", 'rb'))

if AdaBoost:
    if model_input >= 100:
        loaded_model_AdaBoost_100_fLm = pickle.load(open("./models/AdaBoost_final_100_fLm/AdaBoost.clf", 'rb'))
        loaded_model_AdaBoost_100_both = pickle.load(open("./models/AdaBoost_final_100_both/AdaBoost.clf", 'rb'))
        loaded_model_AdaBoost_100_fBm = pickle.load(open("./models/AdaBoost_final_100_fBm/AdaBoost.clf", 'rb'))
    if model_input == 50:
        loaded_model_AdaBoost_50_both = pickle.load(open("./models/AdaBoost_final_50_both/AdaBoost.clf", 'rb'))
        loaded_model_AdaBoost_50_fBm = pickle.load(open("./models/AdaBoost_final_50_fBm/AdaBoost.clf", 'rb'))
        loaded_model_AdaBoost_50_fLm = pickle.load(open("./models/AdaBoost_final_50_fLm/AdaBoost.clf", 'rb'))
    if model_input == 25:
        loaded_model_AdaBoost_25_fLm = pickle.load(open("./models/AdaBoost_final_25_fLm/AdaBoost.clf", 'rb'))
        loaded_model_AdaBoost_25_both = pickle.load(open("./models/AdaBoost_final_25_both/AdaBoost.clf", 'rb'))
        loaded_model_AdaBoost_25_fBm = pickle.load(open("./models/AdaBoost_final_25_fBm/AdaBoost.clf", 'rb'))
    if model_input == 10:
        loaded_model_AdaBoost_10_fBm = pickle.load(open("./models/AdaBoost_final_10_fBm/AdaBoost.clf", 'rb'))
        loaded_model_AdaBoost_10_both = pickle.load(open("./models/AdaBoost_final_10_both/AdaBoost.clf", 'rb'))
        loaded_model_AdaBoost_10_fLm = pickle.load(open("./models/AdaBoost_final_10_fLm/AdaBoost.clf", 'rb'))

if LightGBM:
    if model_input >= 100:
        loaded_model_LightGBM_100_fLm = pickle.load(open("./models/LightGBM_final_100_fLm/LightGBM.clf", 'rb'))
        loaded_model_LightGBM_100_both = pickle.load(open("./models/LightGBM_final_100_both/LightGBM.clf", 'rb'))
        loaded_model_LightGBM_100_fBm = pickle.load(open("./models/LightGBM_final_100_fBm/LightGBM.clf", 'rb'))
    if model_input == 50:
        loaded_model_LightGBM_50_both = pickle.load(open("./models/LightGBM_final_50_both/LightGBM.clf", 'rb'))
        loaded_model_LightGBM_50_fBm = pickle.load(open("./models/LightGBM_final_50_fBm/LightGBM.clf", 'rb'))
        loaded_model_LightGBM_50_fLm = pickle.load(open("./models/LightGBM_final_50_fLm/LightGBM.clf", 'rb'))
    if model_input == 25:
        loaded_model_LightGBM_25_fLm = pickle.load(open("./models/LightGBM_final_25_fLm/LightGBM.clf", 'rb'))
        loaded_model_LightGBM_25_both = pickle.load(open("./models/LightGBM_final_25_both/LightGBM.clf", 'rb'))
        loaded_model_LightGBM_25_fBm = pickle.load(open("./models/LightGBM_final_25_fBm/LightGBM.clf", 'rb'))
    if model_input == 10:
        loaded_model_LightGBM_10_fBm = pickle.load(open("./models/LightGBM_final_10_fBm/LightGBM.clf", 'rb'))
        loaded_model_LightGBM_10_both = pickle.load(open("./models/LightGBM_final_10_both/LightGBM.clf", 'rb'))
        loaded_model_LightGBM_10_fLm = pickle.load(open("./models/LightGBM_final_10_fLm/LightGBM.clf", 'rb'))

if Ridge:
    if model_input >= 100:
        loaded_model_Ridge_100_fLm = pickle.load(open("./models/Ridge_final_100_fLm/Ridge.clf", 'rb'))
        loaded_model_Ridge_100_both = pickle.load(open("./models/Ridge_final_100_both/Ridge.clf", 'rb'))
        loaded_model_Ridge_100_fBm = pickle.load(open("./models/Ridge_final_100_fBm/Ridge.clf", 'rb'))
    if model_input == 50:
        loaded_model_Ridge_50_both = pickle.load(open("./models/Ridge_final_50_both/Ridge.clf", 'rb'))
        loaded_model_Ridge_50_fBm = pickle.load(open("./models/Ridge_final_50_fBm/Ridge.clf", 'rb'))
        loaded_model_Ridge_50_fLm = pickle.load(open("./models/Ridge_final_50_fLm/Ridge.clf", 'rb'))
    if model_input == 25:
        loaded_model_Ridge_25_fLm = pickle.load(open("./models/Ridge_final_25_fLm/Ridge.clf", 'rb'))
        loaded_model_Ridge_25_both = pickle.load(open("./models/Ridge_final_25_both/Ridge.clf", 'rb'))
        loaded_model_Ridge_25_fBm = pickle.load(open("./models/Ridge_final_25_fBm/Ridge.clf", 'rb'))
    if model_input == 10:
        loaded_model_Ridge_10_fBm = pickle.load(open("./models/Ridge_final_10_fBm/Ridge.clf", 'rb'))
        loaded_model_Ridge_10_both = pickle.load(open("./models/Ridge_final_10_both/Ridge.clf", 'rb'))
        loaded_model_Ridge_10_fLm = pickle.load(open("./models/Ridge_final_10_fLm/Ridge.clf", 'rb'))

if Lasso:
    if model_input >= 100:
        loaded_model_Lasso_100_fLm = pickle.load(open("./models/Lasso_final_100_fLm/Lasso.clf", 'rb'))
        loaded_model_Lasso_100_both = pickle.load(open("./models/Lasso_final_100_both/Lasso.clf", 'rb'))
        loaded_model_Lasso_100_fBm = pickle.load(open("./models/Lasso_final_100_fBm/Lasso.clf", 'rb'))
    if model_input == 50:
        loaded_model_Lasso_50_both = pickle.load(open("./models/Lasso_final_50_both/Lasso.clf", 'rb'))
        loaded_model_Lasso_50_fBm = pickle.load(open("./models/Lasso_final_50_fBm/Lasso.clf", 'rb'))
        loaded_model_Lasso_50_fLm = pickle.load(open("./models/Lasso_final_50_fLm/Lasso.clf", 'rb'))
    if model_input == 25:
        loaded_model_Lasso_25_fLm = pickle.load(open("./models/Lasso_final_25_fLm/Lasso.clf", 'rb'))
        loaded_model_Lasso_25_both = pickle.load(open("./models/Lasso_final_25_both/Lasso.clf", 'rb'))
        loaded_model_Lasso_25_fBm = pickle.load(open("./models/Lasso_final_25_fBm/Lasso.clf", 'rb'))
    if model_input == 10:
        loaded_model_Lasso_10_fBm = pickle.load(open("./models/Lasso_final_10_fBm/Lasso.clf", 'rb'))
        loaded_model_Lasso_10_both = pickle.load(open("./models/Lasso_final_10_both/Lasso.clf", 'rb'))
        loaded_model_Lasso_10_fLm = pickle.load(open("./models/Lasso_final_10_fLm/Lasso.clf", 'rb'))

if XGBoost:
    if model_input >= 100:
        loaded_model_XGBoost_100_fLm = pickle.load(open("./models/XGBoost_final_100_fLm/XGBoost.clf", 'rb'))
        loaded_model_XGBoost_100_both = pickle.load(open("./models/XGBoost_final_100_both/XGBoost.clf", 'rb'))
        loaded_model_XGBoost_100_fBm = pickle.load(open("./models/XGBoost_final_100_fBm/XGBoost.clf", 'rb'))
    if model_input == 50:
        loaded_model_XGBoost_50_both = pickle.load(open("./models/XGBoost_final_50_both/XGBoost.clf", 'rb'))
        loaded_model_XGBoost_50_fBm = pickle.load(open("./models/XGBoost_final_50_fBm/XGBoost.clf", 'rb'))
        loaded_model_XGBoost_50_fLm = pickle.load(open("./models/XGBoost_final_50_fLm/XGBoost.clf", 'rb'))
    if model_input == 25:
        loaded_model_XGBoost_25_fLm = pickle.load(open("./models/XGBoost_final_25_fLm/XGBoost.clf", 'rb'))
        loaded_model_XGBoost_25_both = pickle.load(open("./models/XGBoost_final_25_both/XGBoost.clf", 'rb'))
        loaded_model_XGBoost_25_fBm = pickle.load(open("./models/XGBoost_final_25_fBm/XGBoost.clf", 'rb'))
    if model_input == 10:
        loaded_model_XGBoost_10_fBm = pickle.load(open("./models/XGBoost_final_10_fBm/XGBoost.clf", 'rb'))
        loaded_model_XGBoost_10_both = pickle.load(open("./models/XGBoost_final_10_both/XGBoost.clf", 'rb'))
        loaded_model_XGBoost_10_fLm = pickle.load(open("./models/XGBoost_final_10_fLm/XGBoost.clf", 'rb'))

#hurst_exps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
hurst_exps = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]
#hurst_exps = [0.975]

if MLP:
    MLP_array_both = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    MLP_array_fLm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    MLP_array_fBm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    MLP_array_ensemble = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev

if XGBoost:
    XGBoost_array_both = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    XGBoost_array_fLm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    XGBoost_array_fBm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    XGBoost_array_ensemble = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev

if AdaBoost:
    AdaBoost_array_both = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    AdaBoost_array_fLm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    AdaBoost_array_fBm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    AdaBoost_array_ensemble = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev

if LightGBM:
    LightGBM_array_both = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    LightGBM_array_fLm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    LightGBM_array_fBm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    LightGBM_array_ensemble = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev

if Lasso:
    Lasso_array_both = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    Lasso_array_fLm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    Lasso_array_fBm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    Lasso_array_ensemble = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev

if Ridge:
    Ridge_array_both = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    Ridge_array_fLm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    Ridge_array_fBm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    Ridge_array_ensemble = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev

if CatBoost:
    CatBoost_array_both = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    CatBoost_array_fLm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    CatBoost_array_fBm = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev
    CatBoost_array_ensemble = np.empty((stat_count, len(hurst_exps)), dtype=float) #the array will be filled with withthe +2 is for the average and the std_dev

hurst_nolds_array = np.empty((stat_count, len(hurst_exps)), dtype=float)
hurst_hurst_array = np.empty((stat_count, len(hurst_exps)), dtype=float)
hurst_hurst_simplified_array = np.empty((stat_count, len(hurst_exps)), dtype=float)
dfa_nolds_array = np.empty((stat_count, len(hurst_exps)), dtype=float)
higuchi_nolds_array = np.empty((stat_count, len(hurst_exps)), dtype=float)

hurst_nolds_switch = True
hurst_hurst_switch = True
hurst_hurst_simplified_switch = True
dfa_nolds_switch = True
higuchi_nolds_switch = True

for i in range(len(hurst_exps)):
    # print(detrended_list)
    if MLP:
        pred_list_fBm_MLP = list()
        pred_list_fLm_MLP = list()
        pred_list_both_MLP = list()
        pred_list_ensemble_MLP = list()

    if CatBoost:
        pred_list_fBm_CatBoost = list()
        pred_list_fLm_CatBoost = list()
        pred_list_both_CatBoost = list()
        pred_list_ensemble_CatBoost = list()

    if AdaBoost:
        pred_list_fBm_AdaBoost = list()
        pred_list_fLm_AdaBoost = list()
        pred_list_both_AdaBoost = list()
        pred_list_ensemble_AdaBoost = list()

    if XGBoost:
        pred_list_fBm_XGBoost = list()
        pred_list_fLm_XGBoost = list()
        pred_list_both_XGBoost = list()
        pred_list_ensemble_XGBoost = list()

    if LightGBM:
        pred_list_fBm_LightGBM = list()
        pred_list_fLm_LightGBM = list()
        pred_list_both_LightGBM = list()
        pred_list_ensemble_LightGBM = list()

    if Lasso:
        pred_list_fBm_Lasso = list()
        pred_list_fLm_Lasso = list()
        pred_list_both_Lasso = list()
        pred_list_ensemble_Lasso = list()

    if Ridge:
        pred_list_fBm_Ridge = list()
        pred_list_fLm_Ridge = list()
        pred_list_both_Ridge = list()
        pred_list_ensemble_Ridge = list()

    hurst_nolds_list = list()
    hurst_stackoverflow_list = list()
    hurst_hurst_list = list()
    hurst_hurst_list_simplified = list()
    dfa_nolds_list = list()
    higuchi_nolds_list = list()
    petrosian_nolds_list = list()

    if fLm:
        #alpha = round(np.random.uniform(0.1, 1.999), 3)
        signal = dc(flm.flm(alpha=alpha, H=hurst_exps[i], n=18))
    else:
        signal = dc(hurst.random_walk(200000, hurst_exps[i]))

    j = 0
    print(len(signal))

    for ii in range(stat_count):
        print('Hurst: ' + str(hurst_exps[i]))
        print('Stat Count: ' + str(ii))
        step_size = random.randint(10, model_input)
        j = j + step_size

        cut_signal = dc(preprocessing.de_no(signal[j:j + model_input], tensor=False))
        #cut_signal_alg = dc(preprocessing.de_no(signal[j:j + model_input], tensor=False))


        hurst_nolds_list.append(nolds.hurst_rs(cut_signal))
        # hurst_stackoverflow_list.append(hurst_stackoverflow(cut_signal))
        try:
            hurst_hurst_list.append(hurst.compute_Hc(cut_signal, simplified=False)[0])
        except:
            hurst_hurst_list.append(-200)
            hurst_hurst_switch = False
        try:
            hurst_hurst_list_simplified.append(hurst.compute_Hc(cut_signal, simplified=True)[0])
        except:
            hurst_hurst_list_simplified.append(-200)
            hurst_hurst_simplified_switch = False
        try:
            dfa_nolds_list.append(nolds.dfa(cut_signal))
        except:
            dfa_nolds_list.append(-200)
            dfa_nolds_switch = False
        try:
            higuchi_nolds_list.append(neurokit.complexity_fd_higushi(cut_signal, k_max=model_input))
            if higuchi_nolds_list[-1]>2:
                higuchi_nolds_switch = False
        except:
            higuchi_nolds_list.append(-200)
            higuchi_nolds_switch = False

        print("Nolds Hurst: " + str(hurst_nolds_list[-1]))
        # print("Stackoverflow Hurst: " + str(hurst_stackoverflow_list[-1]))
        print("Hurst Hurst: " + str(hurst_hurst_list[-1]))
        print("Hurst Hurst Simplified: " + str(hurst_hurst_list_simplified[-1]))
        print("DFA Nolds: " + str(dfa_nolds_list[-1]))
        print("Higuchi Nolds: " + str(higuchi_nolds_list[-1]))
        #fill stat arrays
        hurst_nolds_array[ii,i] = dc(hurst_nolds_list[-1])
        hurst_hurst_array[ii,i] = dc(hurst_hurst_list[-1])
        hurst_hurst_simplified_array[ii,i] = dc(hurst_hurst_list_simplified[-1])
        dfa_nolds_array[ii,i] = dc(dfa_nolds_list[-1])
        higuchi_nolds_array[ii,i] = dc(higuchi_nolds_list[-1])

        # pred_list.append(loaded_model_RandomForest_5.predict(tensorize_2(X)))
        if MLP:
            if (model_input == 10):
                pred_list_fBm_MLP.append(correct_val(loaded_model_MLP_10_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fBm_MLP.append(correct_val(loaded_model_MLP_25_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fBm_MLP.append(correct_val(loaded_model_MLP_50_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fBm_MLP.append(correct_val(loaded_model_MLP_100_fBm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    #print(ikk)
                    if ikk+100 >= len(cut_signal):
                        pred_list_fBm_MLP.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_MLP_100_fBm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            if (model_input == 10):
                pred_list_both_MLP.append(correct_val(loaded_model_MLP_10_both.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_both_MLP.append(correct_val(loaded_model_MLP_25_both.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_both_MLP.append(correct_val(loaded_model_MLP_50_both.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_both_MLP.append(correct_val(loaded_model_MLP_100_both.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_both_MLP.append(np.mean(np.array(long_signal_aux_list)))
                        #print(pred_list_both_MLP)
                        #print(cut_signal)
                        #print(long_signal_aux_list)
                        #print(len(cut_signal))
                        #sys.exit()
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_MLP_100_both.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))
                    #print(long_signal_aux_list)

            if (model_input == 10):
                pred_list_fLm_MLP.append(correct_val(loaded_model_MLP_10_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fLm_MLP.append(correct_val(loaded_model_MLP_25_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fLm_MLP.append(correct_val(loaded_model_MLP_50_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fLm_MLP.append(correct_val(loaded_model_MLP_100_fLm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_fLm_MLP.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_MLP_100_fLm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            pred_list_ensemble_MLP.append(
                ((pred_list_fBm_MLP[-1] + pred_list_fLm_MLP[-1] + pred_list_both_MLP[-1]) / 3))
            print("Y_pred fBm MLP: " + str(pred_list_fBm_MLP[-1]))
            print("Y_pred fLm MLP: " + str(pred_list_fLm_MLP[-1]))
            print("Y_pred both MLP: " + str(pred_list_both_MLP[-1]))
            print("Y_pred ensemble MLP: " + str(pred_list_ensemble_MLP[-1]))
            print("")
            MLP_array_both[ii,i] = dc(pred_list_both_MLP[-1])
            MLP_array_fLm[ii,i] = dc(pred_list_fLm_MLP[-1])
            MLP_array_fBm[ii,i] = dc(pred_list_fBm_MLP[-1])
            MLP_array_ensemble[ii,i] = dc(pred_list_ensemble_MLP[-1])

        if LightGBM:
            if (model_input == 10):
                pred_list_fBm_LightGBM.append(correct_val(loaded_model_LightGBM_10_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fBm_LightGBM.append(correct_val(loaded_model_LightGBM_25_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fBm_LightGBM.append(correct_val(loaded_model_LightGBM_50_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fBm_LightGBM.append(correct_val(loaded_model_LightGBM_100_fBm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    #print(ikk)
                    if ikk+100 >= len(cut_signal):
                        pred_list_fBm_LightGBM.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_LightGBM_100_fBm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            if (model_input == 10):
                pred_list_both_LightGBM.append(correct_val(loaded_model_LightGBM_10_both.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_both_LightGBM.append(correct_val(loaded_model_LightGBM_25_both.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_both_LightGBM.append(correct_val(loaded_model_LightGBM_50_both.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_both_LightGBM.append(correct_val(loaded_model_LightGBM_100_both.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_both_LightGBM.append(np.mean(np.array(long_signal_aux_list)))
                        #print(pred_list_both_LightGBM)
                        #print(cut_signal)
                        #print(long_signal_aux_list)
                        #print(len(cut_signal))
                        #sys.exit()
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_LightGBM_100_both.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))
                    #print(long_signal_aux_list)

            if (model_input == 10):
                pred_list_fLm_LightGBM.append(correct_val(loaded_model_LightGBM_10_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fLm_LightGBM.append(correct_val(loaded_model_LightGBM_25_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fLm_LightGBM.append(correct_val(loaded_model_LightGBM_50_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fLm_LightGBM.append(correct_val(loaded_model_LightGBM_100_fLm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_fLm_LightGBM.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_LightGBM_100_fLm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            pred_list_ensemble_LightGBM.append(
                ((pred_list_fBm_LightGBM[-1] + pred_list_fLm_LightGBM[-1] + pred_list_both_LightGBM[-1]) / 3))
            print("Y_pred fBm LightGBM: " + str(pred_list_fBm_LightGBM[-1]))
            print("Y_pred fLm LightGBM: " + str(pred_list_fLm_LightGBM[-1]))
            print("Y_pred both LightGBM: " + str(pred_list_both_LightGBM[-1]))
            print("Y_pred ensemble LightGBM: " + str(pred_list_ensemble_LightGBM[-1]))
            print("")
            LightGBM_array_both[ii,i] = dc(pred_list_both_LightGBM[-1])
            LightGBM_array_fLm[ii,i] = dc(pred_list_fLm_LightGBM[-1])
            LightGBM_array_fBm[ii,i] = dc(pred_list_fBm_LightGBM[-1])
            LightGBM_array_ensemble[ii,i] = dc(pred_list_ensemble_LightGBM[-1])

        if CatBoost:
            if (model_input == 10):
                pred_list_fBm_CatBoost.append(correct_val(loaded_model_CatBoost_10_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fBm_CatBoost.append(correct_val(loaded_model_CatBoost_25_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fBm_CatBoost.append(correct_val(loaded_model_CatBoost_50_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fBm_CatBoost.append(correct_val(loaded_model_CatBoost_100_fBm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    #print(ikk)
                    if ikk+100 >= len(cut_signal):
                        pred_list_fBm_CatBoost.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_CatBoost_100_fBm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))
                    if long_signal_aux_list[-1] != long_signal_aux_list[-1]:
                        print("HEre error")
                        break

            if (model_input == 10):
                pred_list_both_CatBoost.append(correct_val(loaded_model_CatBoost_10_both.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_both_CatBoost.append(correct_val(loaded_model_CatBoost_25_both.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_both_CatBoost.append(correct_val(loaded_model_CatBoost_50_both.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_both_CatBoost.append(correct_val(loaded_model_CatBoost_100_both.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_both_CatBoost.append(np.mean(np.array(long_signal_aux_list)))
                        #print(pred_list_both_CatBoost)
                        #print(cut_signal)
                        #print(long_signal_aux_list)
                        #print(len(cut_signal))
                        #sys.exit()
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_CatBoost_100_both.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))
                    #print(long_signal_aux_list)


            if (model_input == 10):
                pred_list_fLm_CatBoost.append(correct_val(loaded_model_CatBoost_10_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fLm_CatBoost.append(correct_val(loaded_model_CatBoost_25_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fLm_CatBoost.append(correct_val(loaded_model_CatBoost_50_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fLm_CatBoost.append(correct_val(loaded_model_CatBoost_100_fLm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_fLm_CatBoost.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_CatBoost_100_fLm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            pred_list_ensemble_CatBoost.append(
                ((pred_list_fBm_CatBoost[-1] + pred_list_fLm_CatBoost[-1] + pred_list_both_CatBoost[-1]) / 3))
            print("Y_pred fBm CatBoost: " + str(pred_list_fBm_CatBoost[-1]))
            print("Y_pred fLm CatBoost: " + str(pred_list_fLm_CatBoost[-1]))
            print("Y_pred both CatBoost: " + str(pred_list_both_CatBoost[-1]))
            print("Y_pred ensemble CatBoost: " + str(pred_list_ensemble_CatBoost[-1]))
            print("")
            CatBoost_array_both[ii, i] = dc(pred_list_both_CatBoost[-1])
            CatBoost_array_fLm[ii, i] = dc(pred_list_fLm_CatBoost[-1])
            CatBoost_array_fBm[ii, i] = dc(pred_list_fBm_CatBoost[-1])
            CatBoost_array_ensemble[ii, i] = dc(pred_list_ensemble_CatBoost[-1])

        if AdaBoost:
            if (model_input == 10):
                pred_list_fBm_AdaBoost.append(correct_val(loaded_model_AdaBoost_10_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fBm_AdaBoost.append(correct_val(loaded_model_AdaBoost_25_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fBm_AdaBoost.append(correct_val(loaded_model_AdaBoost_50_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fBm_AdaBoost.append(correct_val(loaded_model_AdaBoost_100_fBm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    #print(ikk)
                    if ikk+100 >= len(cut_signal):
                        pred_list_fBm_AdaBoost.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_AdaBoost_100_fBm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            if (model_input == 10):
                pred_list_both_AdaBoost.append(correct_val(loaded_model_AdaBoost_10_both.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_both_AdaBoost.append(correct_val(loaded_model_AdaBoost_25_both.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_both_AdaBoost.append(correct_val(loaded_model_AdaBoost_50_both.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_both_AdaBoost.append(correct_val(loaded_model_AdaBoost_100_both.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_both_AdaBoost.append(np.mean(np.array(long_signal_aux_list)))
                        #print(pred_list_both_AdaBoost)
                        #print(cut_signal)
                        #print(long_signal_aux_list)
                        #print(len(cut_signal))
                        #sys.exit()
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_AdaBoost_100_both.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))
                    #print(long_signal_aux_list)


            if (model_input == 10):
                pred_list_fLm_AdaBoost.append(correct_val(loaded_model_AdaBoost_10_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fLm_AdaBoost.append(correct_val(loaded_model_AdaBoost_25_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fLm_AdaBoost.append(correct_val(loaded_model_AdaBoost_50_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fLm_AdaBoost.append(correct_val(loaded_model_AdaBoost_100_fLm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_fLm_AdaBoost.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_AdaBoost_100_fLm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            pred_list_ensemble_AdaBoost.append(
                ((pred_list_fBm_AdaBoost[-1] + pred_list_fLm_AdaBoost[-1] + pred_list_both_AdaBoost[-1]) / 3))
            print("Y_pred fBm AdaBoost: " + str(pred_list_fBm_AdaBoost[-1]))
            print("Y_pred fLm AdaBoost: " + str(pred_list_fLm_AdaBoost[-1]))
            print("Y_pred both AdaBoost: " + str(pred_list_both_AdaBoost[-1]))
            print("Y_pred ensemble AdaBoost: " + str(pred_list_ensemble_AdaBoost[-1]))
            print("")
            AdaBoost_array_both[ii, i] = dc(pred_list_both_AdaBoost[-1])
            AdaBoost_array_fLm[ii, i] = dc(pred_list_fLm_AdaBoost[-1])
            AdaBoost_array_fBm[ii, i] = dc(pred_list_fBm_AdaBoost[-1])
            AdaBoost_array_ensemble[ii, i] = dc(pred_list_ensemble_CatBoost[-1])

        if Ridge:
            if (model_input == 10):
                pred_list_fBm_Ridge.append(correct_val(loaded_model_Ridge_10_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fBm_Ridge.append(correct_val(loaded_model_Ridge_25_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fBm_Ridge.append(correct_val(loaded_model_Ridge_50_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fBm_Ridge.append(correct_val(loaded_model_Ridge_100_fBm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    #print(ikk)
                    if ikk+100 >= len(cut_signal):
                        pred_list_fBm_Ridge.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_Ridge_100_fBm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            if (model_input == 10):
                pred_list_both_Ridge.append(correct_val(loaded_model_Ridge_10_both.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_both_Ridge.append(correct_val(loaded_model_Ridge_25_both.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_both_Ridge.append(correct_val(loaded_model_Ridge_50_both.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_both_Ridge.append(correct_val(loaded_model_Ridge_100_both.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_both_Ridge.append(np.mean(np.array(long_signal_aux_list)))
                        #print(pred_list_both_Ridge)
                        #print(cut_signal)
                        #print(long_signal_aux_list)
                        #print(len(cut_signal))
                        #sys.exit()
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_Ridge_100_both.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))
                    #print(long_signal_aux_list)


            if (model_input == 10):
                pred_list_fLm_Ridge.append(correct_val(loaded_model_Ridge_10_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fLm_Ridge.append(correct_val(loaded_model_Ridge_25_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fLm_Ridge.append(correct_val(loaded_model_Ridge_50_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fLm_Ridge.append(correct_val(loaded_model_Ridge_100_fLm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_fLm_Ridge.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_Ridge_100_fLm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            pred_list_ensemble_Ridge.append(
                ((pred_list_fBm_Ridge[-1] + pred_list_fLm_Ridge[-1] + pred_list_both_Ridge[-1]) / 3))
            print("Y_pred fBm Ridge: " + str(pred_list_fBm_Ridge[-1]))
            print("Y_pred fLm Ridge: " + str(pred_list_fLm_Ridge[-1]))
            print("Y_pred both Ridge: " + str(pred_list_both_Ridge[-1]))
            print("Y_pred ensemble Ridge: " + str(pred_list_ensemble_Ridge[-1]))
            print("")
            Ridge_array_both[ii, i] = dc(pred_list_both_Ridge[-1])
            Ridge_array_fLm[ii, i] = dc(pred_list_fLm_Ridge[-1])
            Ridge_array_fBm[ii, i] = dc(pred_list_fBm_Ridge[-1])
            Ridge_array_ensemble[ii, i] = dc(pred_list_ensemble_Ridge[-1])

        if Lasso:
            if (model_input == 10):
                pred_list_fBm_Lasso.append(correct_val(loaded_model_Lasso_10_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fBm_Lasso.append(correct_val(loaded_model_Lasso_25_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fBm_Lasso.append(correct_val(loaded_model_Lasso_50_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fBm_Lasso.append(correct_val(loaded_model_Lasso_100_fBm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    #print(ikk)
                    if ikk+100 >= len(cut_signal):
                        pred_list_fBm_Lasso.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_Lasso_100_fBm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            if (model_input == 10):
                pred_list_both_Lasso.append(correct_val(loaded_model_Lasso_10_both.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_both_Lasso.append(correct_val(loaded_model_Lasso_25_both.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_both_Lasso.append(correct_val(loaded_model_Lasso_50_both.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_both_Lasso.append(correct_val(loaded_model_Lasso_100_both.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_both_Lasso.append(np.mean(np.array(long_signal_aux_list)))
                        #print(pred_list_both_Lasso)
                        #print(cut_signal)
                        #print(long_signal_aux_list)
                        #print(len(cut_signal))
                        #sys.exit()
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_Lasso_100_both.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))
                    #print(long_signal_aux_list)


            if (model_input == 10):
                pred_list_fLm_Lasso.append(correct_val(loaded_model_Lasso_10_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fLm_Lasso.append(correct_val(loaded_model_Lasso_25_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fLm_Lasso.append(correct_val(loaded_model_Lasso_50_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fLm_Lasso.append(correct_val(loaded_model_Lasso_100_fLm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_fLm_Lasso.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_Lasso_100_fLm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            pred_list_ensemble_Lasso.append(
                ((pred_list_fBm_Lasso[-1] + pred_list_fLm_Lasso[-1] + pred_list_both_Lasso[-1]) / 3))
            print("Y_pred fBm Lasso: " + str(pred_list_fBm_Lasso[-1]))
            print("Y_pred fLm Lasso: " + str(pred_list_fLm_Lasso[-1]))
            print("Y_pred both Lasso: " + str(pred_list_both_Lasso[-1]))
            print("Y_pred ensemble Lasso: " + str(pred_list_ensemble_Lasso[-1]))
            print("")
            Lasso_array_both[ii, i] = dc(pred_list_both_Lasso[-1])
            Lasso_array_fLm[ii, i] = dc(pred_list_fLm_Lasso[-1])
            Lasso_array_fBm[ii, i] = dc(pred_list_fBm_Lasso[-1])
            Lasso_array_ensemble[ii, i] = dc(pred_list_ensemble_Lasso[-1])

        if XGBoost:
            if (model_input == 10):
                pred_list_fBm_XGBoost.append(correct_val(loaded_model_XGBoost_10_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fBm_XGBoost.append(correct_val(loaded_model_XGBoost_25_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fBm_XGBoost.append(correct_val(loaded_model_XGBoost_50_fBm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fBm_XGBoost.append(correct_val(loaded_model_XGBoost_100_fBm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    #print(ikk)
                    if ikk+100 >= len(cut_signal):
                        pred_list_fBm_XGBoost.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_XGBoost_100_fBm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            if (model_input == 10):
                pred_list_both_XGBoost.append(correct_val(loaded_model_XGBoost_10_both.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_both_XGBoost.append(correct_val(loaded_model_XGBoost_25_both.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_both_XGBoost.append(correct_val(loaded_model_XGBoost_50_both.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_both_XGBoost.append(correct_val(loaded_model_XGBoost_100_both.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_both_XGBoost.append(np.mean(np.array(long_signal_aux_list)))
                        #print(pred_list_both_XGBoost)
                        #print(cut_signal)
                        #print(long_signal_aux_list)
                        #print(len(cut_signal))
                        #sys.exit()
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_XGBoost_100_both.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))
                    #print(long_signal_aux_list)


            if (model_input == 10):
                pred_list_fLm_XGBoost.append(correct_val(loaded_model_XGBoost_10_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 25):
                pred_list_fLm_XGBoost.append(correct_val(loaded_model_XGBoost_25_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 50):
                pred_list_fLm_XGBoost.append(correct_val(loaded_model_XGBoost_50_fLm.predict(tensorize_2(cut_signal))))
            if (model_input == 100):
                pred_list_fLm_XGBoost.append(correct_val(loaded_model_XGBoost_100_fLm.predict(tensorize_2(cut_signal))))
            if model_input > 100:
                long_signal_aux_list = dc(list())
                for ikk in range(len(cut_signal)):
                    if ikk+100 >= len(cut_signal):
                        pred_list_fLm_XGBoost.append(np.mean(np.array(long_signal_aux_list)))
                        break
                    long_signal_aux_list.append(correct_val(loaded_model_XGBoost_100_fLm.predict(tensorize_2(preprocessing.de_no(cut_signal[ikk:(ikk+100)], tensor=False)))))

            pred_list_ensemble_XGBoost.append(
                ((pred_list_fBm_XGBoost[-1] + pred_list_fLm_XGBoost[-1] + pred_list_both_XGBoost[-1]) / 3))
            print("Y_pred fBm XGBoost: " + str(pred_list_fBm_XGBoost[-1]))
            print("Y_pred fLm XGBoost: " + str(pred_list_fLm_XGBoost[-1]))
            print("Y_pred both XGBoost: " + str(pred_list_both_XGBoost[-1]))
            print("Y_pred ensemble XGBoost: " + str(pred_list_ensemble_XGBoost[-1]))
            print("")
            XGBoost_array_both[ii, i] = dc(pred_list_both_XGBoost[-1])
            XGBoost_array_fLm[ii, i] = dc(pred_list_fLm_XGBoost[-1])
            XGBoost_array_fBm[ii, i] = dc(pred_list_fBm_XGBoost[-1])
            XGBoost_array_ensemble[ii, i] = dc(pred_list_ensemble_XGBoost[-1])





print('Build output array')
header = list()
header.append("hurst_exp")
header.append("alg_nolds_hurst")
if hurst_hurst_switch:
    header.append("alg_hurst_hurst")
if hurst_hurst_simplified_switch:
    header.append("alg_hurst_hurst_simplified")
if dfa_nolds_switch:
    header.append("alg_dfa_nolds")
if higuchi_nolds_switch:
    header.append("alg_higuchi_nolds")

if MLP:
    header.append("MLP_both")
    header.append("MLP_fLm")
    header.append("MLP_fBm")
    header.append("MLP_ensemble")
if CatBoost:
    header.append("CatBoost_both")
    header.append("CatBoost_fLm")
    header.append("CatBoost_fBm")
    header.append("CatBoost_ensemble")
if LightGBM:
    header.append("LightGBM_both")
    header.append("LightGBM_fLm")
    header.append("LightGBM_fBm")
    header.append("LightGBM_ensemble")
if AdaBoost:
    header.append("AdaBoost_both")
    header.append("AdaBoost_fLm")
    header.append("AdaBoost_fBm")
    header.append("AdaBoost_ensemble")
if Ridge:
    header.append("Ridge_both")
    header.append("Ridge_fLm")
    header.append("Ridge_fBm")
    header.append("Ridge_ensemble")
if Lasso:
    header.append("Lasso_both")
    header.append("Lasso_fLm")
    header.append("Lasso_fBm")
    header.append("Lasso_ensemble")
if XGBoost:
    header.append("XGBoost_both")
    header.append("XGBoost_fLm")
    header.append("XGBoost_fBm")
    header.append("XGBoost_ensemble")







out_array_mean = np.empty((len(hurst_exps)+1, len(header)), dtype=object)
out_array_mean[0,:] = dc(header)
out_array_mean[1:,0] = dc(hurst_exps)
for i in range(len(hurst_exps)):
    k=0

    k = k + 1
    out_array_mean[i+1, k] = np.mean(hurst_nolds_array[:,i])
    if hurst_hurst_switch:
        k=k+1
        out_array_mean[i + 1, k] = np.mean(hurst_hurst_array[:, i])
    if hurst_hurst_simplified_switch:
        k=k+1
        out_array_mean[i + 1, k] = np.mean(hurst_hurst_simplified_array[:, i])
    if dfa_nolds_switch:
        k=k+1
        out_array_mean[i + 1, k]  = np.mean(dfa_nolds_array[:, i] - 1)
    if higuchi_nolds_switch:
        k=k+1
        out_array_mean[i + 1, k] = np.mean(higuchi_nolds_array[:, i])

    if MLP:
        k=k+1
        out_array_mean[i+1, k] = clean_mean(MLP_array_both[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(MLP_array_fLm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(MLP_array_fBm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(MLP_array_ensemble[:,i])
    if CatBoost:
        k=k+1
        out_array_mean[i+1, k] = clean_mean(CatBoost_array_both[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(CatBoost_array_fLm[:,i])
        k=k+1
        #print('pizza')
        #print(CatBoost_array_fBm)
        #print('pizza')

        #sys.exit()
        out_array_mean[i+1, k] = clean_mean(CatBoost_array_fBm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(CatBoost_array_ensemble[:,i])
    if LightGBM:
        k=k+1
        out_array_mean[i+1, k] = clean_mean(LightGBM_array_both[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(LightGBM_array_fLm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(LightGBM_array_fBm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(LightGBM_array_ensemble[:,i])
    if AdaBoost:
        k=k+1
        out_array_mean[i+1, k] = clean_mean(AdaBoost_array_both[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(AdaBoost_array_fLm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(AdaBoost_array_fBm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(AdaBoost_array_ensemble[:,i])
    if Ridge:
        k=k+1
        out_array_mean[i+1, k] = clean_mean(Ridge_array_both[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(Ridge_array_fLm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(Ridge_array_fBm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(Ridge_array_ensemble[:,i])
    if Lasso:
        k=k+1
        out_array_mean[i+1, k] = clean_mean(Lasso_array_both[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(Lasso_array_fLm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(Lasso_array_fBm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(Lasso_array_ensemble[:,i])
    if XGBoost:
        k=k+1
        out_array_mean[i+1, k] = clean_mean(XGBoost_array_both[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(XGBoost_array_fLm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(XGBoost_array_fBm[:,i])
        k=k+1
        out_array_mean[i+1, k] = clean_mean(XGBoost_array_ensemble[:,i])



out_array_std = np.empty((len(hurst_exps)+1, len(header)), dtype=object)
out_array_std[0,:] = dc(header)
out_array_std[1:,0] = dc(hurst_exps)

for i in range(len(hurst_exps)):
    k=0
    k = k + 1
    out_array_std[i+1, k] = np.std(hurst_nolds_array[:,i])
    if hurst_hurst_switch:
        k=k+1
        out_array_std[i + 1, k] = np.std(hurst_hurst_array[:, i])
    if hurst_hurst_simplified_switch:
        k=k+1
        out_array_std[i + 1, k] = np.std(hurst_hurst_simplified_array[:, i])
    if dfa_nolds_switch:
        k=k+1
        out_array_std[i + 1, k]  = np.std(dfa_nolds_array[:, i] - 1)
    if higuchi_nolds_switch:
        k=k+1
        out_array_std[i + 1, k] = np.std(higuchi_nolds_array[:, i])

    if MLP:
        k=k+1
        out_array_std[i+1, k] = clean_std(MLP_array_both[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(MLP_array_fLm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(MLP_array_fBm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(MLP_array_ensemble[:,i])
    if CatBoost:
        k=k+1
        out_array_std[i+1, k] = clean_std(CatBoost_array_both[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(CatBoost_array_fLm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(CatBoost_array_fBm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(CatBoost_array_ensemble[:,i])
    if LightGBM:
        k=k+1
        out_array_std[i+1, k] = clean_std(LightGBM_array_both[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(LightGBM_array_fLm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(LightGBM_array_fBm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(LightGBM_array_ensemble[:,i])
    if AdaBoost:
        k=k+1
        out_array_std[i+1, k] = clean_std(AdaBoost_array_both[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(AdaBoost_array_fLm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(AdaBoost_array_fBm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(AdaBoost_array_ensemble[:,i])
    if Ridge:
        k=k+1
        out_array_std[i+1, k] = clean_std(Ridge_array_both[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(Ridge_array_fLm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(Ridge_array_fBm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(Ridge_array_ensemble[:,i])
    if Lasso:
        k=k+1
        out_array_std[i+1, k] = clean_std(Lasso_array_both[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(Lasso_array_fLm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(Lasso_array_fBm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(Lasso_array_ensemble[:,i])
    if XGBoost:
        k=k+1
        out_array_std[i+1, k] = clean_std(XGBoost_array_both[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(XGBoost_array_fLm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(XGBoost_array_fBm[:,i])
        k=k+1
        out_array_std[i+1, k] = clean_std(XGBoost_array_ensemble[:,i])



print('Saving Results')
np.savetxt("./COMP_DATA//out_average_" + add + ".csv", out_array_mean, fmt='%s', delimiter=",")
np.savetxt("./COMP_DATA//out_std_" + add + ".csv", out_array_std, fmt='%s', delimiter=",")
print('Results saved')

plt.figure(figsize=(20, 10), dpi=100)
for i in range(len(header)):
    aux_str = dc(str(header[i]))
    print(aux_str)
    print(aux_str[:3])



    x_plot = dc(np.array(out_array_mean[1:,0], dtype=float))
    y_plot = dc(np.array(out_array_mean[1:,i], dtype=float))
    print(x_plot)
    print(y_plot)
    if aux_str[:3] == "alg":
        plt.plot(x_plot, y_plot, "-.", label=str(header[i]), linewidth=1.6)
        print('plotting algorithm data')
    else:
        plt.plot(x_plot, y_plot, label=str(header[i]), linewidth=0.7)
        print('plotting ML data')

    #plt.show()
plt.legend(bbox_to_anchor=(1.01, 1.01), loc='upper left', borderaxespad=0.)
plt.xlabel("H true")
plt.ylabel("H predicted")
plt.tight_layout()
plt.savefig('./plots/corr_plot_hurst' + add + '.png')  # , dpi=150)
plt.savefig('./plots/corr_plot_hurst' + add + '.eps')  # , dpi=150)
plt.ion()
plt.show(block=False)
plt.pause(0.01)
plt.close()



plt.figure(figsize=(20, 10), dpi=100)
for i in range(len(header)-1):
    aux_str = dc(str(header[i+1]))
    print(aux_str)
    print(aux_str[:3])



    x_plot = dc(np.array(out_array_std[1:,0], dtype=float))
    y_plot = dc(np.array(out_array_std[1:,i+1], dtype=float))
    print(x_plot)
    print(y_plot)
    if aux_str[:3] == "alg":
        plt.plot(x_plot, y_plot, "-.", label=str(header[i+1]), linewidth=1.6)
        print('plotting algorithm data')
    else:
        plt.plot(x_plot, y_plot, label=str(header[i+1]), linewidth=0.7)
        print('plotting ML data')

    #plt.show()
plt.legend(bbox_to_anchor=(1.01, 1.01), loc='upper left', borderaxespad=0.)
plt.xlabel("H true")
plt.ylabel("H predicted")
plt.tight_layout()
plt.savefig('./plots/std_plot_hurst' + add + '.png')  # , dpi=150)
plt.savefig('./plots/std_plot_hurst' + add + '.eps')  # , dpi=150)
plt.ion()
plt.show(block=False)
plt.pause(0.01)
plt.close()

import sys
sys.exit()

