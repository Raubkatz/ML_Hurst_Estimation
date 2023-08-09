"""
25.01.2023, Sebastian Raubitzek
Program to generate a dataset for training a ML algorithm to calculate a range of signal complexity metrics:
I.e. Hurst exponent, largest lyapunov exponent, sample entropy and approximate entropy.

"""

import warnings #ignore warning, can be ignored XD
warnings.filterwarnings('ignore')
import flm_bencoscia
import antropy
import nolds
from random import randint
import hurst
import sys
from copy import deepcopy as dc
from scipy.signal import detrend as lin_detrend
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import random
import tensorflow as tf
import flm #fractional levy motion
from datetime import datetime
from matplotlib import pyplot as plt
rand_seed = 0 #2 for trainnig data set, 3 for test data set
random_hurst_dataset = True # generate just a random hurst dataset, overwrites all other complexity metrics
de_no = True #detrend and normalize signal output
detrend = False
gen_hurst = True #hurst yes/no
gen_samp_en = False #samp_en yes/no
gen_app_en = False #app_en yes/no
gen_lyap_r = False #lyap_r yes/no
gen_lyap_e = False #lyap_e yes/no
min_scale = 0 # scale to (min,max)
max_scale = 1
test_plots = False
fbm_y_flm = True
len_brown = 1000 #length of the brownian motion which is then to be seperated into smaller signals
n_monte_carlo = 1 # pumping up the the amount of data, i.e. generating X different brownian motions for each hurst and length, drastically increases computation time
hurst_len = 100000 #only applies if random_hurst_dataset =True
sub_prob = 0.15 #probabilty to take a sub signal as a training/test sample

if random_hurst_dataset:
    gen_hurst = False
    gen_samp_en = False
    gen_app_en = False
    gen_lyap_e = False
    gen_lyap_r = False


#label output
add = "_"

if de_no:
    add = add + "de_no_"
if gen_hurst:
    add = add + "gen_hurst_"
if gen_lyap_e:
    add = add + "gen_lyap_e_"
if gen_lyap_r:
    add = add + "gen_lyap_r_"
if gen_app_en:
    add = add + "gen_app_en_"
if gen_samp_en:
    add = add + "gen_samp_en_"

add = add + "_"
add = add + "lb" + str(len_brown) + "_nmc" + str(n_monte_carlo)

if random_hurst_dataset:
    add = "rand_hurst_" + "lb" + str(len_brown) + "_nmc" + str(n_monte_carlo)
if fbm_y_flm:
    add = add + "_flm"

#add = dc("_test_1000_flm")
add = dc("train_100k_fLm")

#list of hurst exponents and lengths
#for testing:
#hurst_list = [0.3, 0.5, 0.7]
#len_list = [50, 100, 500]
#len_list = [50, 100, 200, 500]
#len_list = [10,25,50,75,100]
#len_list = [5,10,25,50,75,100]
#len_list = [10,25,50,100]
len_list = [10,25,50,100]
#len_list = [5]



#actual lists
#hurst_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95] #the actual training list
#len_list = [10,20,25,30,40,50,60,70,80,90,100,150,200,250,500,750,1000,1500,2000,5000,10000]
#also, for the actual experiements, all complexity metrics = True, len_brown = 200000, n_monte_carlo=20
#some other test ideas
#hurst_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,] #some other test ideas
#len_list = [25,50,100,200,500,1000] #some other test ideas


def rand_bool(prob):
    s=str(prob)
    p=s.index('.')
    d=10**(len(s)-p)
    return randint(0,d*d-1)%d<int(s[p+1:])


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

def generate_random_hurst_list(hurst_len=1000):
    hurst_list = list()
    for i in range(hurst_len):
        H = round(np.random.uniform(0.001, 0.999),3)
        hurst_list.append(H)
    return np.array(hurst_list, dtype=float)

def generate_header(length=100, random_hurst_dataset=random_hurst_dataset ,gen_hurst=False, gen_lyap_e=False, gen_lyap_r=False, gen_app_en=False, gen_samp_en=False):
    """
    requires:
    length: integer; to be the length of the signal.
    gen_hurst: boolean; do we calculate the hurst exponent?
    gen_lyap_e: boolean; Do we claculate the lyapunov expoennt using the algorithm by Eckmann?
    gen_lyap_e: boolean; Do we claculate the lyapunov expoennt using the algorithm by Rosenstein?
    gen_app_en: boolean; Do we calculate approximate entropy?
    gen_samp_en: boolean, Do we calculate sample entropy?
    """
    header = list()
    for i in range(length):
        header.append(("dp_" + str(i))) #data point 1, 2, 3, etc.
    if random_hurst_dataset:
        header.append("fbm_hurst")
    else:
        if gen_hurst: # adding the labels for the Hurst exponent
            header.append("fbm_hurst")
            header.append("nolds_hurst_long")
            header.append("hurst_hurst_long")
            header.append("hurst_hurst_simplified_long")
            header.append("DFA_long")
            header.append("nolds_hurst_long_de_no")
            header.append("hurst_hurst_long_de_no")
            header.append("hurst_hurst_simplified_long_de_no")
            header.append("DFA_long_de_no")
            header.append("nolds_hurst_short")
            header.append("hurst_hurst_short")
            header.append("hurst_hurst_simplified_short")
            header.append("DFA_short")
            header.append("nolds_hurst_short_de_no")
            header.append("hurst_hurst_short_de_no")
            header.append("hurst_hurst_simplified_short_de_no")
            header.append("DFA_short_de_no")
        if gen_lyap_e: # adding the labels for Eckmann lypunov exponent
            header.append("lyap_e")
            header.append("lyap_e_short")
            header.append("lyap_e_de_no")
            header.append("lyap_e_short_de_no")
        if gen_lyap_r: # adding the labels for Rosenstein lypunov exponent
            header.append("lyap_r")
            header.append("lyap_r_short")
            header.append("lyap_r_de_no")
            header.append("lyap_r_short_de_no")
        if gen_app_en: # adding the labels for the approximate Entropy
            header.append("app_en")
            header.append("app_en_short")
            header.append("app_en_de_no")
            header.append("app_en_short_de_no")
        if gen_samp_en: # adding the labels for the sample Entropy
            header.append("samp_en")
            header.append("samp_en_short")
            header.append("samp_en_de_no")
            header.append("samp_en_short_de_no")
    return dc(header)

def reset_seeds(seed,reset_graph_with_backend=None): #reset seeds, this is a generalized reset including tensorflow.
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("KERAS AND TENSORFLOW GRAPHS RESET")  # optional
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    print("RANDOM SEEDS RESET")  # optional

def de_no(time_series, min_scale=0.0, max_scale=1.0, detrend=True):
    try:
        if detrend:
            time_series = dc(lin_detrend(time_series))
        scaler = dc(MinMaxScaler(feature_range=(min_scale, max_scale)))
    except:
        print('something is wrong, probably wrong alpha or somehting check signal:')
        print(time_series)
        sys.exit()
    return dc(un_brace(scaler.fit_transform(re_brace(time_series))))


start_global = datetime.now()

for length in len_list: #run thourhg all lengths
    start_length = dc(datetime.now())
    print('length')
    print(length)
    len_out_list = dc(list()) # this list is the output array
    header = dc(generate_header(length=length, gen_hurst=gen_hurst, gen_lyap_e=gen_lyap_e, gen_lyap_r=gen_lyap_r, gen_app_en=gen_app_en, gen_samp_en=gen_samp_en))
    len_out_list.append(header) #get header on top
    if random_hurst_dataset:
        hurst_list = dc(generate_random_hurst_list(hurst_len=hurst_len))
    hurst_count = 0
    for H in hurst_list: #run through all hursts
        print('##########################################')
        print('##########################################')
        print('##########################################')
        print('##########################################')
        start_hurst = dc(datetime.now())
        print('Hurst #' + str(hurst_count))
        print("H=" + str(H))
        print('##########################################')
        print('##########################################')
        print('##########################################')
        print('##########################################')
        hurst_count = hurst_count + 1
        for i in range(n_monte_carlo): #how many times should it be repeated? I.e. generating more samples
            start_monte_carlo = dc(datetime.now())
            print('i_monte_carlo')
            print(i)
            print('Seed:')
            seed = int(hurst_count + rand_seed*hurst_len)
            print(seed)
            reset_seeds(seed) #always reset the seeds
            out_str = dc("")
            if fbm_y_flm:
                if (hurst_count+1)%2==0:
                    print('fLm')
                    out_str = dc("fLm")
                    alpha = round(np.random.uniform(0.1,1.999),3)
                    #FLM = dc(flm_bencoscia.FLM(len_brown, H, alpha))
                    #FLM.generate_realizations(1)
                    #signal = dc(FLM.realizations[0])
                    print(alpha)
                    print(H)
                    signal = dc(flm.flm(alpha=alpha, H=H, n=10))
                    signal = dc(signal[:len_brown])
                    print(np.shape(signal))
                else:
                    print("fBm")
                    out_str = "fBm"
                    signal = dc(hurst.random_walk(len_brown, H)) #generate fractional Brownian motion
            else:
                signal = dc(hurst.random_walk(len_brown, H))  # generate fractional Brownian motion

            fbm_hurst = H
            H_str = str(H)
            H_str = dc(H_str[2:])
            if hurst_count<20:
                np.savetxt("./COMP_DATA/ts_saves/" + str(length) + "signal_" + out_str + str(H_str) + ".csv", signal, fmt='%s', delimiter=",")
            #sub signal stuff
            for ii in range(len(signal)): #loop through the signal
                bool_switch = rand_bool(sub_prob)
                if bool_switch: #make a prompt every 10th signal partition
                    print("sub signal #" + str(ii)) #just for check the progress of the program
                    if (ii+length) >= len(signal): #if the subsignal exceeds the long signal -> break
                        break
                    ts_list = list()
                    short_signal = dc(signal[ii:(ii+length)])
                    if test_plots:
                        plt.plot(short_signal)
                        plt.show()

                    short_signal_de_no = dc(de_no(short_signal, detrend=detrend))
                    if test_plots:
                        plt.plot(short_signal_de_no)
                        plt.show()

                    for iii_de_no in range(length):
                        ts_list.append(short_signal_de_no[iii_de_no])
                    ts_list.append(fbm_hurst)
                    len_out_list.append(ts_list) #append datapoint to output array
            end_monte_carlo = dc(datetime.now())
            print("Calculation time Monte Carlo run  " + str(i) + ":\n" + str((end_monte_carlo - start_monte_carlo)))
        end_hurst = dc(datetime.now())
        print("Calculation time Hurst " + str(H) + ":\n" + str((end_hurst - start_hurst)))
    end_length = dc(datetime.now())
    print("Calculation time length " + str(length) + ":\n" + str((end_length - start_length)))
    print('Save results, length: ' + str(length))
    len_out_arr = dc(np.array(len_out_list, dtype=object))
    add_out = add + str(length) #label output correctly
    np.savetxt("./data_ML//" + add_out + ".csv", len_out_list, fmt='%s', delimiter=",") #save output for a certain signal length
    print('Saved')
end_global = datetime.now()
print("Calculation time global: " + str((end_global-start_global)))





