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
import neurokit

from datetime import datetime
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import sys


def re_brace(dataset): # adds an additional brackst to the inner values
    Out_Arr = []
    for i in range(len(dataset)):
        Out_Arr.append(dataset[(i):(i+1)])
    return np.array(Out_Arr)

def un_brace(dataset): # removes the inner bracket
    Out_Arr = np.empty([len(dataset)])
    for i in range(len(dataset)):
        Out_Arr[i] = dataset[i,0]
    return Out_Arr

def RMSE_Err_Prop(av, std, ground_truth):
    rmse_list = list()
    av = dc(un_brace(av))
    std = dc(un_brace(std))
    #print(av)
    #print(std)
    #sys.exit()
    for i in range(len(av)):
        rmse_list.append((av[i] - ground_truth[i]) * (av[i] - ground_truth[i]))
    rmse_list = dc(np.array(rmse_list))
    rmse = np.sqrt(np.sum(rmse_list) / len(av))

    rmse_err_nominator = list()
    rmse_err_denominator = list()

    for i in range(len(av)):
        rmse_err_nominator.append((av[i] - ground_truth[i]) * (av[i] - ground_truth[i]) * std[i] * std[i])
        rmse_err_denominator.append(len(av) * (av[i] - ground_truth[i]) * (av[i] - ground_truth[i]))
    rmse_err_nominator = dc(np.sum(np.array(rmse_err_nominator)))
    rmse_err_denominator = dc(np.sum(np.array(rmse_err_denominator)))
    rmse_err = np.sqrt(rmse_err_nominator / rmse_err_denominator)
    return rmse, rmse_err




hurst_exps = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425,
              0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85,
              0.875, 0.9, 0.925, 0.95, 0.975]
#out_array_mean = np.genfromtxt("./COMP_DATA//out_average_" + add + ".csv", delimiter=',', dtype=float)
#out_array_std = np.genfromtxt("./COMP_DATA//out_std_" + add + ".csv", delimiter=',', dtype=float)

# np.savetxt("./COMP_DATA//out_average_" + add + ".csv", out_array_mean, fmt='%s', delimiter=",")
# np.savetxt("./COMP_DATA//out_std_" + add + ".csv", out_array_std, fmt='%s', delimiter=",")

fLm = True
alpha = 1.5
rw_list = ['fBm', 'fLm', 'both']
model_input_list = [10,25,50,100,200,350]
print('Build output array')
header = list()
header.append("Random Walk for Training")
# header.append("alg. nolds Hurst")
# if hurst_hurst_switch:
#    header.append("alg. Hurst Hurst")
# if hurst_hurst_simplified_switch:
#    header.append("alg. Hurst Hurst simplified")
# if dfa_nolds_switch:
#    header.append("alg. nolds DFA")
# if higuchi_nolds_switch:
#    header.append("alg. nolds Higuchi")




header_alg = list()
header_alg.append('window length')
header_alg.append("alg. nolds Hurst")
header_alg.append("alg. nolds DFA")
header_alg.append("alg. Hurst Hurst")
header_alg.append("alg. Hurst Hurst simplified")

header_MLP = list()
header_MLP.append('window length')
header_MLP.append("MLP fBm")
header_MLP.append("MLP fLm")
header_MLP.append("MLP both")

header_AdaBoost = list()
header_AdaBoost.append('window length')
header_AdaBoost.append("AdaBoost fBm")
header_AdaBoost.append("AdaBoost fLm")
header_AdaBoost.append("AdaBoost both")

header_CatBoost = list()
header_CatBoost.append('window length')
header_CatBoost.append("CatBoost fBm")
header_CatBoost.append("CatBoost fLm")
header_CatBoost.append("CatBoost both")

header_LightGBM = list()
header_LightGBM.append('window length')
header_LightGBM.append("LightGBM fBm")
header_LightGBM.append("LightGBM fLm")
header_LightGBM.append("LightGBM both")

header_Lasso = list()
header_Lasso.append('window length')
header_Lasso.append("Lasso fBm")
header_Lasso.append("Lasso fLm")
header_Lasso.append("Lasso both")

header_Ridge = list()
header_Ridge.append('window length')
header_Ridge.append("Ridge fBm")
header_Ridge.append("Ridge fLm")
header_Ridge.append("Ridge both")

header_XGBoost = list()
header_XGBoost.append('window length')
header_XGBoost.append("XGBoost fBm")
header_XGBoost.append("XGBoost fLm")
header_XGBoost.append("XGBoost both")



MLP_fBm_list = list()
XGBoost_fBm_list = list()
AdaBoost_fBm_list = list()
CatBoost_fBm_list = list()
LightGBM_fBm_list = list()
Lasso_fBm_list = list()
Ridge_fBm_list = list()

MLP_both_list = list()
XGBoost_both_list = list()
AdaBoost_both_list = list()
CatBoost_both_list = list()
LightGBM_both_list = list()
Lasso_both_list = list()
Ridge_both_list = list()

MLP_fLm_list = list()
XGBoost_fLm_list = list()
AdaBoost_fLm_list = list()
CatBoost_fLm_list = list()
LightGBM_fLm_list = list()
Lasso_fLm_list = list()
Ridge_fLm_list = list()



MLP_out_array = np.empty((len(model_input_list) + 1, len(header_MLP)), dtype=object)
AdaBoost_out_array = np.empty((len(model_input_list) + 1, len(header_AdaBoost)), dtype=object)
CatBoost_out_array = np.empty((len(model_input_list) + 1, len(header_CatBoost)), dtype=object)
LightGBM_out_array = np.empty((len(model_input_list) + 1, len(header_LightGBM)), dtype=object)
Lasso_out_array = np.empty((len(model_input_list) + 1, len(header_Lasso)), dtype=object)
Ridge_out_array = np.empty((len(model_input_list) + 1, len(header_Ridge)), dtype=object)
XGBoost_out_array = np.empty((len(model_input_list) + 1, len(header_XGBoost)), dtype=object)


for i in range(len(model_input_list)):
    model_input = model_input_list[i]
    stat_count = 1000

    MLP = True
    LightGBM = True
    CatBoost = True
    AdaBoost = True
    Lasso = True
    Ridge = True
    XGBoost = False

    # add is to label the output files
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

    if model_input < 100:
        hurst_hurst_switch = False
        hurst_hurst_simplified_switch = False
        higuchi_nolds_switch = False
    else:
        hurst_hurst_switch = True
        hurst_hurst_simplified_switch = True
        higuchi_nolds_switch = True

    add = add + "_cor_final_42"

    data_av = dc(pd.read_csv("./COMP_DATA//out_average_" + add + ".csv"))
    data_std = dc(pd.read_csv("./COMP_DATA//out_std_" + add + ".csv"))

    if MLP:
        av_MLP_both = dc(data_av[['MLP_both']].to_numpy())
        std_MLP_both = dc(data_std[['MLP_both']].to_numpy())
        MLP_both_rmse, MLP_both_rmse_err = dc(RMSE_Err_Prop(av_MLP_both, std_MLP_both, hurst_exps))
        MLP_both_list.append(str(str(round(MLP_both_rmse,5)) + "±" + str(round(MLP_both_rmse_err,5))))

        av_MLP_fBm = dc(data_av[['MLP_fBm']].to_numpy())
        std_MLP_fBm = dc(data_std[['MLP_fBm']].to_numpy())
        MLP_fBm_rmse, MLP_fBm_rmse_err = dc(RMSE_Err_Prop(av_MLP_fBm, std_MLP_fBm, hurst_exps))
        MLP_fBm_list.append(str(str(round(MLP_fBm_rmse,5)) + "±" + str(round(MLP_fBm_rmse_err,5))))

        av_MLP_fLm = dc(data_av[['MLP_fLm']].to_numpy())
        std_MLP_fLm = dc(data_std[['MLP_fLm']].to_numpy())
        MLP_fLm_rmse, MLP_fLm_rmse_err = dc(RMSE_Err_Prop(av_MLP_fLm, std_MLP_fLm, hurst_exps))
        MLP_fLm_list.append(str(str(round(MLP_fLm_rmse,5)) + "±" + str(round(MLP_fLm_rmse_err,5))))
    if XGBoost:
        av_XGBoost_both = dc(data_av[['XGBoost_both']].to_numpy())
        std_XGBoost_both = dc(data_std[['XGBoost_both']].to_numpy())
        XGBoost_both_rmse, XGBoost_both_rmse_err = dc(RMSE_Err_Prop(av_XGBoost_both, std_XGBoost_both, hurst_exps))
        XGBoost_both_list.append(str(str(round(XGBoost_both_rmse, 5)) + "±" + str(round(XGBoost_both_rmse_err, 5))))

        av_XGBoost_fBm = dc(data_av[['XGBoost_fBm']].to_numpy())
        std_XGBoost_fBm = dc(data_std[['XGBoost_fBm']].to_numpy())
        XGBoost_fBm_rmse, XGBoost_fBm_rmse_err = dc(RMSE_Err_Prop(av_XGBoost_fBm, std_XGBoost_fBm, hurst_exps))
        XGBoost_fBm_list.append(str(str(round(XGBoost_fBm_rmse, 5)) + "±" + str(round(XGBoost_fBm_rmse_err, 5))))

        av_XGBoost_fLm = dc(data_av[['XGBoost_fLm']].to_numpy())
        std_XGBoost_fLm = dc(data_std[['XGBoost_fLm']].to_numpy())
        XGBoost_fLm_rmse, XGBoost_fLm_rmse_err = dc(RMSE_Err_Prop(av_XGBoost_fLm, std_XGBoost_fLm, hurst_exps))
        XGBoost_fLm_list.append(str(str(round(XGBoost_fLm_rmse, 5)) + "±" + str(round(XGBoost_fLm_rmse_err, 5))))
    if AdaBoost:
        av_AdaBoost_both = dc(data_av[['AdaBoost_both']].to_numpy())
        std_AdaBoost_both = dc(data_std[['AdaBoost_both']].to_numpy())
        AdaBoost_both_rmse, AdaBoost_both_rmse_err = dc(RMSE_Err_Prop(av_AdaBoost_both, std_AdaBoost_both, hurst_exps))
        AdaBoost_both_list.append(str(str(round(AdaBoost_both_rmse, 5)) + "±" + str(round(AdaBoost_both_rmse_err, 5))))

        av_AdaBoost_fBm = dc(data_av[['AdaBoost_fBm']].to_numpy())
        std_AdaBoost_fBm = dc(data_std[['AdaBoost_fBm']].to_numpy())
        AdaBoost_fBm_rmse, AdaBoost_fBm_rmse_err = dc(RMSE_Err_Prop(av_AdaBoost_fBm, std_AdaBoost_fBm, hurst_exps))
        AdaBoost_fBm_list.append(str(str(round(AdaBoost_fBm_rmse, 5)) + "±" + str(round(AdaBoost_fBm_rmse_err, 5))))

        av_AdaBoost_fLm = dc(data_av[['AdaBoost_fLm']].to_numpy())
        std_AdaBoost_fLm = dc(data_std[['AdaBoost_fLm']].to_numpy())
        AdaBoost_fLm_rmse, AdaBoost_fLm_rmse_err = dc(RMSE_Err_Prop(av_AdaBoost_fLm, std_AdaBoost_fLm, hurst_exps))
        AdaBoost_fLm_list.append(str(str(round(AdaBoost_fLm_rmse, 5)) + "±" + str(round(AdaBoost_fLm_rmse_err, 5))))
    if CatBoost:
        av_CatBoost_both = dc(data_av[['CatBoost_both']].to_numpy())
        std_CatBoost_both = dc(data_std[['CatBoost_both']].to_numpy())
        CatBoost_both_rmse, CatBoost_both_rmse_err = dc(RMSE_Err_Prop(av_CatBoost_both, std_CatBoost_both, hurst_exps))
        CatBoost_both_list.append(str(str(round(CatBoost_both_rmse, 5)) + "±" + str(round(CatBoost_both_rmse_err, 5))))

        av_CatBoost_fBm = dc(data_av[['CatBoost_fBm']].to_numpy())
        std_CatBoost_fBm = dc(data_std[['CatBoost_fBm']].to_numpy())
        CatBoost_fBm_rmse, CatBoost_fBm_rmse_err = dc(RMSE_Err_Prop(av_CatBoost_fBm, std_CatBoost_fBm, hurst_exps))
        CatBoost_fBm_list.append(str(str(round(CatBoost_fBm_rmse, 5)) + "±" + str(round(CatBoost_fBm_rmse_err, 5))))

        av_CatBoost_fLm = dc(data_av[['CatBoost_fLm']].to_numpy())
        std_CatBoost_fLm = dc(data_std[['CatBoost_fLm']].to_numpy())
        CatBoost_fLm_rmse, CatBoost_fLm_rmse_err = dc(RMSE_Err_Prop(av_CatBoost_fLm, std_CatBoost_fLm, hurst_exps))
        CatBoost_fLm_list.append(str(str(round(CatBoost_fLm_rmse, 5)) + "±" + str(round(CatBoost_fLm_rmse_err, 5))))
    if Lasso:
        av_Lasso_both = dc(data_av[['Lasso_both']].to_numpy())
        std_Lasso_both = dc(data_std[['Lasso_both']].to_numpy())
        Lasso_both_rmse, Lasso_both_rmse_err = dc(RMSE_Err_Prop(av_Lasso_both, std_Lasso_both, hurst_exps))
        Lasso_both_list.append(str(str(round(Lasso_both_rmse, 5)) + "±" + str(round(Lasso_both_rmse_err, 5))))

        av_Lasso_fBm = dc(data_av[['Lasso_fBm']].to_numpy())
        std_Lasso_fBm = dc(data_std[['Lasso_fBm']].to_numpy())
        Lasso_fBm_rmse, Lasso_fBm_rmse_err = dc(RMSE_Err_Prop(av_Lasso_fBm, std_Lasso_fBm, hurst_exps))
        Lasso_fBm_list.append(str(str(round(Lasso_fBm_rmse, 5)) + "±" + str(round(Lasso_fBm_rmse_err, 5))))

        av_Lasso_fLm = dc(data_av[['Lasso_fLm']].to_numpy())
        std_Lasso_fLm = dc(data_std[['Lasso_fLm']].to_numpy())
        Lasso_fLm_rmse, Lasso_fLm_rmse_err = dc(RMSE_Err_Prop(av_Lasso_fLm, std_Lasso_fLm, hurst_exps))
        Lasso_fLm_list.append(str(str(round(Lasso_fLm_rmse, 5)) + "±" + str(round(Lasso_fLm_rmse_err, 5))))
    if Ridge:
        av_Ridge_both = dc(data_av[['Ridge_both']].to_numpy())
        std_Ridge_both = dc(data_std[['Ridge_both']].to_numpy())
        Ridge_both_rmse, Ridge_both_rmse_err = dc(RMSE_Err_Prop(av_Ridge_both, std_Ridge_both, hurst_exps))
        Ridge_both_list.append(str(str(round(Ridge_both_rmse, 5)) + "±" + str(round(Ridge_both_rmse_err, 5))))

        av_Ridge_fBm = dc(data_av[['Ridge_fBm']].to_numpy())
        std_Ridge_fBm = dc(data_std[['Ridge_fBm']].to_numpy())
        Ridge_fBm_rmse, Ridge_fBm_rmse_err = dc(RMSE_Err_Prop(av_Ridge_fBm, std_Ridge_fBm, hurst_exps))
        Ridge_fBm_list.append(str(str(round(Ridge_fBm_rmse, 5)) + "±" + str(round(Ridge_fBm_rmse_err, 5))))

        av_Ridge_fLm = dc(data_av[['Ridge_fLm']].to_numpy())
        std_Ridge_fLm = dc(data_std[['Ridge_fLm']].to_numpy())
        Ridge_fLm_rmse, Ridge_fLm_rmse_err = dc(RMSE_Err_Prop(av_Ridge_fLm, std_Ridge_fLm, hurst_exps))
        Ridge_fLm_list.append(str(str(round(Ridge_fLm_rmse, 5)) + "±" + str(round(Ridge_fLm_rmse_err, 5))))
    if LightGBM:
        av_LightGBM_both = dc(data_av[['LightGBM_both']].to_numpy())
        std_LightGBM_both = dc(data_std[['LightGBM_both']].to_numpy())
        LightGBM_both_rmse, LightGBM_both_rmse_err = dc(RMSE_Err_Prop(av_LightGBM_both, std_LightGBM_both, hurst_exps))
        LightGBM_both_list.append(str(str(round(LightGBM_both_rmse, 5)) + "±" + str(round(LightGBM_both_rmse_err, 5))))

        av_LightGBM_fBm = dc(data_av[['LightGBM_fBm']].to_numpy())
        std_LightGBM_fBm = dc(data_std[['LightGBM_fBm']].to_numpy())
        LightGBM_fBm_rmse, LightGBM_fBm_rmse_err = dc(RMSE_Err_Prop(av_LightGBM_fBm, std_LightGBM_fBm, hurst_exps))
        LightGBM_fBm_list.append(str(str(round(LightGBM_fBm_rmse, 5)) + "±" + str(round(LightGBM_fBm_rmse_err, 5))))

        av_LightGBM_fLm = dc(data_av[['LightGBM_fLm']].to_numpy())
        std_LightGBM_fLm = dc(data_std[['LightGBM_fLm']].to_numpy())
        LightGBM_fLm_rmse, LightGBM_fLm_rmse_err = dc(RMSE_Err_Prop(av_LightGBM_fLm, std_LightGBM_fLm, hurst_exps))
        LightGBM_fLm_list.append(str(str(round(LightGBM_fLm_rmse, 5)) + "±" + str(round(LightGBM_fLm_rmse_err, 5))))


add_out = ""
if fLm:
    add_out = add_out + "_fLm_" + str(alpha)
else:
    add_out = "_fBm"


if MLP:
    MLP_out_array[0,:] = dc(header_MLP)
    MLP_out_array[1:,0] = dc(model_input_list)
    MLP_out_array[1:,1] = dc(MLP_fBm_list)
    MLP_out_array[1:,2] = dc(MLP_fLm_list)
    MLP_out_array[1:,3] = dc(MLP_both_list)
    np.savetxt("./COMP_DATA//out_MLP" + add_out + ".csv", MLP_out_array, fmt='%s', delimiter=",")
if LightGBM:
    LightGBM_out_array[0,:] = dc(header_LightGBM)
    LightGBM_out_array[1:,0] = dc(model_input_list)
    LightGBM_out_array[1:,1] = dc(LightGBM_fBm_list)
    LightGBM_out_array[1:,2] = dc(LightGBM_fLm_list)
    LightGBM_out_array[1:,3] = dc(LightGBM_both_list)
    np.savetxt("./COMP_DATA//out_LightGBM" + add_out + ".csv", LightGBM_out_array, fmt='%s', delimiter=",")
if CatBoost:
    CatBoost_out_array[0,:] = dc(header_CatBoost)
    CatBoost_out_array[1:,0] = dc(model_input_list)
    CatBoost_out_array[1:,1] = dc(CatBoost_fBm_list)
    CatBoost_out_array[1:,2] = dc(CatBoost_fLm_list)
    CatBoost_out_array[1:,3] = dc(CatBoost_both_list)
    np.savetxt("./COMP_DATA//out_CatBoost" + add_out + ".csv", CatBoost_out_array, fmt='%s', delimiter=",")
if Ridge:
    Ridge_out_array[0,:] = dc(header_Ridge)
    Ridge_out_array[1:,0] = dc(model_input_list)
    Ridge_out_array[1:,1] = dc(Ridge_fBm_list)
    Ridge_out_array[1:,2] = dc(Ridge_fLm_list)
    Ridge_out_array[1:,3] = dc(Ridge_both_list)
    np.savetxt("./COMP_DATA//out_Ridge" + add_out + ".csv", Ridge_out_array, fmt='%s', delimiter=",")
if Lasso:
    Lasso_out_array[0,:] = dc(header_Lasso)
    Lasso_out_array[1:,0] = dc(model_input_list)
    Lasso_out_array[1:,1] = dc(Lasso_fBm_list)
    Lasso_out_array[1:,2] = dc(Lasso_fLm_list)
    Lasso_out_array[1:,3] = dc(Lasso_both_list)
    np.savetxt("./COMP_DATA//out_Lasso" + add_out + ".csv", Lasso_out_array, fmt='%s', delimiter=",")
if XGBoost:
    XGBoost_out_array[0,:] = dc(header_XGBoost)
    XGBoost_out_array[1:,0] = dc(model_input_list)
    XGBoost_out_array[1:,1] = dc(XGBoost_fBm_list)
    XGBoost_out_array[1:,2] = dc(XGBoost_fLm_list)
    XGBoost_out_array[1:,3] = dc(XGBoost_both_list)
    np.savetxt("./COMP_DATA//out_XGBoost" + add_out + ".csv", XGBoost_out_array, fmt='%s', delimiter=",")
if AdaBoost:
    AdaBoost_out_array[0,:] = dc(header_AdaBoost)
    AdaBoost_out_array[1:,0] = dc(model_input_list)
    AdaBoost_out_array[1:,1] = dc(AdaBoost_fBm_list)
    AdaBoost_out_array[1:,2] = dc(AdaBoost_fLm_list)
    AdaBoost_out_array[1:,3] = dc(AdaBoost_both_list)
    np.savetxt("./COMP_DATA//out_AdaBoost" + add_out + ".csv", AdaBoost_out_array, fmt='%s', delimiter=",")


if MLP:
    MLP_df = pd.DataFrame(MLP_out_array[1:], columns=MLP_out_array[0])
    MLP_latex = MLP_df.to_latex(index=False)
    with open(f"./COMP_DATA/out_MLP{add_out}.tex", 'w') as file:
        file.write(MLP_latex)

if LightGBM:
    LightGBM_df = pd.DataFrame(LightGBM_out_array[1:], columns=LightGBM_out_array[0])
    LightGBM_latex = LightGBM_df.to_latex(index=False)
    with open(f"./COMP_DATA/out_LightGBM{add_out}.tex", 'w') as file:
        file.write(LightGBM_latex)

if CatBoost:
    CatBoost_df = pd.DataFrame(CatBoost_out_array[1:], columns=CatBoost_out_array[0])
    CatBoost_latex = CatBoost_df.to_latex(index=False)
    with open(f"./COMP_DATA/out_CatBoost{add_out}.tex", 'w') as file:
        file.write(CatBoost_latex)

if Ridge:
    Ridge_df = pd.DataFrame(Ridge_out_array[1:], columns=Ridge_out_array[0])
    Ridge_latex = Ridge_df.to_latex(index=False)
    with open(f"./COMP_DATA/out_Ridge{add_out}.tex", 'w') as file:
        file.write(Ridge_latex)

if Lasso:
    Lasso_df = pd.DataFrame(Lasso_out_array[1:], columns=Lasso_out_array[0])
    Lasso_latex = Lasso_df.to_latex(index=False)
    with open(f"./COMP_DATA/out_Lasso{add_out}.tex", 'w') as file:
        file.write(Lasso_latex)


if XGBoost:
    XGBoost_df = pd.DataFrame(XGBoost_out_array[1:], columns=XGBoost_out_array[0])
    XGBoost_latex = XGBoost_df.to_latex(index=False)
    with open(f"./COMP_DATA/out_XGBoost{add_out}.tex", 'w') as file:
        file.write(XGBoost_latex)

if AdaBoost:
    AdaBoost_df = pd.DataFrame(AdaBoost_out_array[1:], columns=AdaBoost_out_array[0])
    AdaBoost_latex = AdaBoost_df.to_latex(index=False)
    with open(f"./COMP_DATA/out_AdaBoost{add_out}.tex", 'w') as file:
        file.write(AdaBoost_latex)









