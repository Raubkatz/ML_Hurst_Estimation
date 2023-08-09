from pandas import DataFrame
import numpy as np
import pandas as pd
import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import util as util
import feature_engineering as fe





def extract_datetime(df: DataFrame) -> DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['dow'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    return df.drop(columns=["timestamp"])
    
def do_nuthin(df: DataFrame) -> DataFrame:
    return df

    


def get_bikes_pipeline():
    return Pipeline([
        ('feature_engineering', fe.GenericFeatureExtractor([extract_datetime])),
        ('one_hot', fe.OneHotEncoder(categorical_cols=['season', 'weather_code'])),
        ('scaling', MinMaxScaler())  # min-max scaling does not affect one hot encoded values
    ])


def get_superc_pipeline():
    return Pipeline([('scaling', StandardScaler())])

def get_air_pass_pipeline():
    return Pipeline([('scaling', StandardScaler())])

def lin_reg_test_pipeline():
    return Pipeline([('scaling', StandardScaler())])
    
def get_hurst_pipeline():
    return Pipeline([('scaling', StandardScaler())])

def get_hurst_min_max_pipeline():
    return Pipeline([('scaling', MinMaxScaler())])

def get_hurst_no_pp_pipeline():
    return Pipeline([('generic', fe.GenericFeatureExtractor([do_nuthin]))])

def multi_lin_reg_test_pipeline():
    return Pipeline([('scaling', StandardScaler())])

def get_CO_pred_pipeline():
    return Pipeline([('scaling', StandardScaler())])

def get_NOX_pred_pipeline():
    return Pipeline([('scaling', StandardScaler())])

def get_Lotto_pipeline():
    return Pipeline([('scaling', StandardScaler())])


