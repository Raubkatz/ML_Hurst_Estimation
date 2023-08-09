import pickle
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import detrend as lin_detrend
import hurst
from copy import deepcopy as dc
import numpy as np

def re_brace(dataset): # adds additional brackets to the inner values, for ML and sklearn stuff, tensorization
    Out_Arr = []
    for i in range(len(dataset)):
        Out_Arr.append(dataset[(i):(i+1)])
    return np.array(Out_Arr)

def un_brace(dataset): # removes the inner bracket, for ML and sklearn stuff, untensorization
    Out_Arr = np.empty([len(dataset)])
    for i in range(len(dataset)):
        Out_Arr[i] = dataset[i,0]
    return Out_Arr


#signal = dc(hurst.random_walk(len_brown, H))  # generate fractional Brownian motion

#signal_de_no = dc(lin_detrend(signal))
#scaler = MinMaxScaler(feature_range=(min_scale, max_scale))
#signal_de_no = dc(un_brace(scaler.fit_transform(re_brace(signal_de_no))))

def de_no(signal, tensor=True, list=True, min_scale=0, max_scale=1):
    signal = dc(lin_detrend(signal))
    scaler = dc(MinMaxScaler(feature_range=(min_scale, max_scale)))
    if tensor:
        return dc(scaler.fit_transform(re_brace(signal)))
    else:
        return dc(un_brace(scaler.fit_transform(re_brace(signal))))