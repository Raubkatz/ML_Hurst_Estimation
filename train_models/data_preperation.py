import datetime
from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from pipelines import get_bikes_pipeline, get_superc_pipeline, get_air_pass_pipeline, get_CO_pred_pipeline, get_NOX_pred_pipeline, get_Lotto_pipeline, get_hurst_pipeline, get_hurst_min_max_pipeline, get_hurst_no_pp_pipeline
import numpy as np
from scipy import sparse as sp

datasets = {
    "bikes": "./data/london_merged.csv",
    "superc": "./data/superconductivity.csv",
    "air_pass": "./data/airline_passengers.csv",
    "lin_reg_test": "./data/lin_reg_test.csv",
    "multi_lin_reg_test": "./data/multi_lin_reg_test.csv",
    "CO_pred_11": "./data/pp_gas_emission/gt_2011.csv",
    "CO_pred_12": "./data/pp_gas_emission/gt_2012.csv",
    "CO_pred_13": "./data/pp_gas_emission/gt_2013.csv",
    "CO_pred_14": "./data/pp_gas_emission/gt_2014.csv",
    "CO_pred_15": "./data/pp_gas_emission/gt_2015.csv",
    "NOX_pred_11": "./data/pp_gas_emission/gt_2011.csv",
    "NOX_pred_12": "./data/pp_gas_emission/gt_2012.csv",
    "NOX_pred_13": "./data/pp_gas_emission/gt_2013.csv",
    "NOX_pred_14": "./data/pp_gas_emission/gt_2014.csv",
    "NOX_pred_15": "./data/pp_gas_emission/gt_2015.csv",
    "CO_pred_full": "./data/pp_gas_emission/gt_full.csv",
    "NOX_pred_full": "./data/pp_gas_emission/gt_full.csv",
    "Lotto1": "./EuroMilDat_Orig/EuroMillions_ML_1.csv",
    "Lotto2": "./EuroMilDat_Orig/EuroMillions_ML_2.csv",
    "Lotto3": "./EuroMilDat_Orig/EuroMillions_ML_3.csv",
    "Lotto4": "./EuroMilDat_Orig/EuroMillions_ML_4.csv",
    "Lotto5": "./EuroMilDat_Orig/EuroMillions_ML_5.csv",
    "Lotto6": "./EuroMilDat_Orig/EuroMillions_ML_6.csv",
    "Lotto7": "./EuroMilDat_Orig/EuroMillions_ML_7.csv",
    "Lotto8": "./EuroMilDat_Orig/EuroMillions_ML_8.csv",
    "Lotto9": "./EuroMilDat_Orig/EuroMillions_ML_9.csv",
    "Lotto10": "./EuroMilDat_Orig/EuroMillions_ML_10.csv",
    "OELotto1": "./OELotto/OE_nu_lotto_1.csv",
    "OELotto2": "./OELotto/OE_nu_lotto_2.csv",
    "OELotto3": "./OELotto/OE_nu_lotto_3.csv",
    "OELotto4": "./OELotto/OE_nu_lotto_4.csv",
    "OELotto5": "./OELotto/OE_nu_lotto_5.csv",
    "OELotto6": "./OELotto/OE_nu_lotto_6.csv",
    "OELotto7": "./OELotto/OE_nu_lotto_7.csv",
    "OELotto8": "./OELotto/OE_nu_lotto_8.csv",
    "OELotto9": "./OELotto/OE_nu_lotto_9.csv",
    "OELotto10": "./OELotto/OE_nu_lotto_10.csv",
    "hurst_100": "./train_datasets/train_hurst_100.csv",
    "hurst_5": "./train_20000_flmfbm/train_hurst_5.csv",
    "hurst_10": "./train_datasets/train_hurst_10.csv",
    "hurst_10_test": "./test_datasets/test_hurst_10.csv",
    "hurst_50": "./train_20000_flmfbm/train_hurst_50.csv",
    "hurst_75": "./train_20000_flmfbm/train_hurst_75.csv",
    "hurst_25": "./train_datasets/train_hurst_25.csv"
}

test_data = {
}


def get_pipeline(dataset: str, data: DataFrame, args):
    if dataset == "bikes":
        return get_bikes_pipeline()
    if dataset == "superc":
        return get_superc_pipeline()
    if dataset == "air_pass":
        return get_air_pass_pipeline()
    if dataset == "hurst_100":
        return get_hurst_pipeline()
    if dataset == "hurst_10":
        return get_hurst_no_pp_pipeline()
    if dataset == "hurst_10_test":
        return get_hurst_no_pp_pipeline()
    if dataset == "hurst_5":
        return get_hurst_no_pp_pipeline()
    if dataset == "hurst_25":
        return get_hurst_no_pp_pipeline()
    if dataset == "hurst_75":
        return get_hurst_no_pp_pipeline()
    if dataset == "hurst_50":
        return get_hurst_no_pp_pipeline()
    if dataset == "lin_reg_test":
        return get_air_pass_pipeline()
    if dataset == "multi_lin_reg_test":
        return get_air_pass_pipeline()
    if dataset == "CO_pred_11":
        return get_CO_pred_pipeline()
    if dataset == "CO_pred_12":
        return get_CO_pred_pipeline()
    if dataset == "CO_pred_13":
        return get_CO_pred_pipeline()
    if dataset == "CO_pred_14":
        return get_CO_pred_pipeline()
    if dataset == "CO_pred_15":
        return get_CO_pred_pipeline()
    if dataset == "CO_pred_full":
        return get_CO_pred_pipeline()
    if dataset == "NOX_pred_11":
        return get_NOX_pred_pipeline()
    if dataset == "NOX_pred_12":
        return get_NOX_pred_pipeline()
    if dataset == "NOX_pred_13":
        return get_NOX_pred_pipeline()
    if dataset == "NOX_pred_14":
        return get_NOX_pred_pipeline()
    if dataset == "NOX_pred_15":
        return get_NOX_pred_pipeline()
    if dataset == "NOX_pred_full":
        return get_NOX_pred_pipeline()
    if dataset == "Lotto":
        return get_Lotto_pipeline()
    if dataset == "Lotto1":
        return get_Lotto_pipeline()
    if dataset == "Lotto2":
        return get_Lotto_pipeline()
    if dataset == "Lotto3":
        return get_Lotto_pipeline()
    if dataset == "Lotto4":
        return get_Lotto_pipeline()
    if dataset == "Lotto5":
        return get_Lotto_pipeline()
    if dataset == "Lotto6":
        return get_Lotto_pipeline()
    if dataset == "Lotto7":
        return get_Lotto_pipeline()
    if dataset == "Lotto8":
        return get_Lotto_pipeline()
    if dataset == "Lotto9":
        return get_Lotto_pipeline()
    if dataset == "Lotto10":
        return get_Lotto_pipeline()
    if dataset == "Lotto":
        return get_Lotto_pipeline()
    if dataset == "OELotto1":
        return get_Lotto_pipeline()
    if dataset == "OELotto2":
        return get_Lotto_pipeline()
    if dataset == "OELotto3":
        return get_Lotto_pipeline()
    if dataset == "OELotto4":
        return get_Lotto_pipeline()
    if dataset == "OELotto5":
        return get_Lotto_pipeline()
    if dataset == "OELotto6":
        return get_Lotto_pipeline()
    if dataset == "OELotto7":
        return get_Lotto_pipeline()
    if dataset == "OELotto8":
        return get_Lotto_pipeline()
    if dataset == "OELotto9":
        return get_Lotto_pipeline()
    if dataset == "OELotto10":
        return get_Lotto_pipeline()



def get_response_column(dataset: str, data: DataFrame) -> Tuple[np.array, LabelEncoder]:
    """
    Returns the relevant response data with the corresponding encoder, so the inverse may be
    determined to create a solution file.
    """
    return data[response_column[dataset]]


response_column = {
    "bikes": "cnt",
    "superc": "critical_temp",
    "air_pass": "Passengers",
    "hurst_100": "fbm_hurst",
    "hurst_10": "fbm_hurst",
    "hurst_10_test": "fbm_hurst",
    "hurst_5": "fbm_hurst",
    "hurst_25": "fbm_hurst",
    "hurst_50": "fbm_hurst",
    "hurst_75": "fbm_hurst",
    "lin_reg_test": "Passengers",
    "multi_lin_reg_test": "Passengers",
    "CO_pred_11": "CO",
    "CO_pred_12": "CO",
    "CO_pred_13": "CO",
    "CO_pred_14": "CO",
    "CO_pred_15": "CO",
    "CO_pred_full": "CO",
    "NOX_pred_11": "NOX",
    "NOX_pred_12": "NOX",
    "NOX_pred_13": "NOX",
    "NOX_pred_14": "NOX",
    "NOX_pred_15": "NOX",
    "NOX_pred_full": "NOX",
    "Lotto": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "Lotto1": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "Lotto2": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "Lotto3": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "Lotto4": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "Lotto5": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "Lotto6": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "Lotto7": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "Lotto8": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "Lotto9": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "Lotto10": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "OELotto": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "OELotto1": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "OELotto2": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "OELotto3": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "OELotto4": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "OELotto5": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "OELotto6": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "OELotto7": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "OELotto8": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "OELotto9": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "OELotto10": ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"],
    "LottC0": "Class0",
    "LottC1": "Class1",
    "LottC2": "Class2",
    "LottC3": "Class3",
    "LottC4": "Class4",
    "LottC5": "Class5",
    "LottC6": "Class6"
}

single_response = {
    "LottC0": "Class0",
    "LottC1": "Class1",
    "LottC2": "Class2",
    "LottC3": "Class3",
    "LottC4": "Class4",
    "LottC5": "Class5",
    "LottC6": "Class6"
}

multi_lotto_response_columns = ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"]


def prepare_data(dataset: str, args, test_size=0.25):
    data: DataFrame = read_train_data(dataset, nrows=args.nrows)

    train_values = get_response_column(dataset, data)
    try: data.drop(columns=[response_column[dataset]], inplace=True)
    except:
        data.drop(columns=[multi_lotto_response_columns[0]], inplace=True)
        data.drop(columns=[multi_lotto_response_columns[1]], inplace=True)
        data.drop(columns=[multi_lotto_response_columns[2]], inplace=True)
        data.drop(columns=[multi_lotto_response_columns[3]], inplace=True)
        data.drop(columns=[multi_lotto_response_columns[4]], inplace=True)
        data.drop(columns=[multi_lotto_response_columns[5]], inplace=True)
        data.drop(columns=[multi_lotto_response_columns[6]], inplace=True)

        #data.drop(columns=[response_column[dataset]], inplace=True)
        #data.drop(columns=[response_column[dataset]], inplace=True)
        #data.drop(columns=[response_column[dataset]], inplace=True)
        #data.drop(columns=[response_column[dataset]], inplace=True)
        #data.drop(columns=[response_column[dataset]], inplace=True)
        #data.drop(columns=[response_column[dataset]], inplace=True)





    pre_processing_pipeline = get_pipeline(dataset, data, args)

    train_data = pre_processing_pipeline.fit_transform(data)

    # Replace missing values
    # num_cols = train_data.select_dtypes(include=['number']).columns.values
    # train_data[num_cols] = train_data[num_cols].fillna(-999, inplace=False, downcast=False)

    if args.reduce is not None:
        pca = PCA(n_components=args.reduce)
        pca.fit(train_data)
        train_data = pca.transform(train_data)
        if args.verbosity:
            print("Shape after PCA: ", train_data.shape)
    else:
        pca = None

    X_train, X_test, y_train, y_test = train_test_split(train_data, train_values, test_size=test_size, random_state=123)
    return X_train, X_test, y_train, y_test, pre_processing_pipeline, pca


def read_train_data(dataset: str, nrows: int = None) -> DataFrame:
    return pd.read_csv(datasets[dataset], header=0, nrows=nrows)


def read_test_data(dataset: str, nrows: int = None) -> DataFrame:
    return pd.read_csv(test_data[dataset], header=0, nrows=nrows)



