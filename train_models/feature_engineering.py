import itertools
from typing import List, Callable
import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
import util


class GenericFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, extractors: List[Callable[[DataFrame], DataFrame]]):
        self.extractors = extractors

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for extractor in self.extractors:
            X = extractor(X)
        return X


class ArrayIndexEncoder(BaseEstimator, TransformerMixin):
    """
    Data:
    id    present_media
    0     Video
    1     Picture<SEP>Picture
    2     Video<SEP>Picture
    #
    Result:
    id    present_media_Video present_media_Picture
    0     1                   0
    1     0                   2
    2     1                   1
    """
    def __init__(self, columns: List[str], sep='\t'):
        self.columns = columns
        self.sep = sep

    def fit(self, X, y=None):
        return self

    def transform(self, X: DataFrame):

        for col in self.columns:
            unique_values = self.__get_unique_values(X[col])

            for unique_value in unique_values:
                column_name = '{}_{}'.format(col, unique_value)

                tmp_series = X[col].apply(
                    lambda x: self.__cnt_hex_list(unique_value, x) if isinstance(x, str) else 0)
                tmp_series = tmp_series.rename(column_name)

                X = pd.concat([X, tmp_series], axis=1)

        return X

    def __cnt_hex_list(self, col, x):
        output = np.array(str(x).split(self.sep))
        cnt = np.count_nonzero(output == col)
        return cnt

    def __get_unique_values(self, df: pd.Series):
        values = df.dropna().unique()
        tmp_values = []
        for value in values:
            if self.sep in value:
                tmp_values.append(value.split(self.sep))
            else:
                tmp_values.append([value])

        unique_values = np.unique(list(itertools.chain(*tmp_values)))
        return unique_values


class ArraySplitEncoder(BaseEstimator, TransformerMixin):
    """
    Data:
    id    present_media
    0     Video
    1     Picture<SEP>Picture
    2     Video<SEP>Picture
    #
    Result:
    id    present_media
    0     [Video]
    1     [Picture, Picture]
    2     [Video, Picture]
    """
    def __init__(self, columns: List[str], sep='\t'):
        self.columns = columns
        self.sep = sep

    def fit(self, X, y=None):
        return self

    def transform(self, X: DataFrame):
        for column in self.columns:
            X[column] = X[column].str.split(self.sep)

        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency encoding:
    Data:
    0    orange
    1    apple
    2    orange
    3    apple
    4    pear
    5    apple
    
    Result:
    0    2.0
    1    3.0
    2    2.0
    3    3.0
    4    1.0
    5    3.0
    """
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.__frequency_encoding(X, self.cols)
    
    @staticmethod
    def __frequency_encoding(df: DataFrame, cols: List[str]) -> DataFrame:
        for col in cols:
            counts = df[col].value_counts().to_dict()
            name = col + '_counts'
            df[name] = df[col].map(counts)
            df[name] = df[name].astype('float32')
        return df


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Data:
    id    column1
    0     orange 
    1     apple  
    2     pear   
    
    Result:
    id    column1
    0     [1,0,0]
    1     [0,1,0]
    2     [0,0,1]
    """
    def __init__(self, categorical_cols: List[str], drop=True):
        self.categorical_cols = categorical_cols
        self.encoder = None  # preprocessing.OneHotEncoder()
        self.encoded_cols = None
        self.drop = drop

    def fit(self, X, y=None):
        self.encoder = preprocessing.OneHotEncoder().fit(X[self.categorical_cols])
        return self

    def transform(self, X):
        return self.__encode(X, self.categorical_cols)

    def __encode(self, df: DataFrame, categorical_cols: List[str]) -> DataFrame:
        # df[categorical_cols] = df[categorical_cols].apply(self.encoder.fit_transform)
        self.encoded_cols = self.encoder.get_feature_names(self.categorical_cols)
        encoded_df = DataFrame(self.encoder.transform(df[categorical_cols]).toarray(), columns=self.encoded_cols)
        df[self.encoded_cols] = encoded_df
        if self.drop:
            df.drop(columns=categorical_cols, inplace=True)
        # df[categorical_cols] = df[categorical_cols].astype('category')
        return df

class NoEncode(BaseEstimator, TransformerMixin):
    """
    Data:
    id    column1
    0     orange 
    1     apple  
    2     pear   
    
    Result:
    id    column1
    0     [1,0,0]
    1     [0,1,0]
    2     [0,0,1]
    """
    def __init__(self, drop=True):
        self.encoder = None  # preprocessing.OneHotEncoder()
        self.encoded_cols = None
        self.drop = drop


    def transform(self, X):
        return self.__encode(X)

    def __encode(self, df: DataFrame) -> DataFrame:
        return df

class FeatureAggregator(BaseEstimator, TransformerMixin):
    """
    Data:
    id    column1     column2
    0     orange      2
    1     apple       1
    2     orange      3
    
    Result with group_cols=["column1"] and variables=["column2"]:
    id    column1_column2_mean
    0     2.5
    1     1.0 
    2     2.5  
    """
    def __init__(self, group_cols: [str], variables: [str], measure: str, fillna=None):
        self.group_cols = group_cols
        self.variables = variables
        self.measure = measure
        self.fillna = fillna

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.__aggregate(X, self.group_cols, self.variables, self.measure, self.fillna)
    
    @staticmethod
    def __aggregate(self, df: DataFrame, group_cols: [str], variables: [str], measure: str, fillna=None) -> DataFrame:
        for col in group_cols:
            for variable in variables:
                agg = df.groupby(col)[variable].agg(measure)
                df[variable + "_" + col + "_" + measure] = agg
                if fillna is not None:
                    df[variable + "_" + col + "_" + measure].fillna(fillna)
        return df


class FeatureCombiner(BaseEstimator, TransformerMixin):
    """
    Data:
    id    column1     column2
    0     orange      US
    1     apple       EU
    2     orange      EU
    3     apple       EU

    Result:
    id    column1_column2
    0     0  (orange_US)
    1     1  (apple_EU)
    2     2  (orange_EU)
    3     1  (apple_EU)
    """
    def __init__(self, cols: List[str], name: str):
        self.cols = cols
        self.name = name
        self.label_encoder = preprocessing.LabelEncoder()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.__combine_features(X, self.cols, self.name)

    def __combine_features(self, df: DataFrame, cols: [str], name: str) -> DataFrame:
        df[name] = df[cols[0]].astype(str)
        for i in range(1, len(cols)):
            df[name] = df[name] + '_' + df[cols[i]].astype(str)
        return self.__label_encode(df, [name])

    def __label_encode(self, df: DataFrame, categorical_cols: List[str]) -> DataFrame:
        df[categorical_cols] = df[categorical_cols].apply(self.label_encoder.fit_transform)
        # df[categorical_cols] = df[categorical_cols].astype('category')
        return df


class FeatureScaler(BaseEstimator, TransformerMixin):

    def __init__(self, numerical_cols: List[str]):
        self.numerical_cols = numerical_cols
        self.scaler = preprocessing.MinMaxScaler()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.__scale(X, self.numerical_cols)

    def __scale(self, df: DataFrame, numerical_cols: List[str]) -> DataFrame:
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        return df


class NotNullEncoder(BaseEstimator, TransformerMixin):
    """
    Data:
    id    column1
    0     orange 
    1     NaN  
    2     pear   

    Result:
    id    column1
    0     1
    1     0
    2     1
    """
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.cols:
            mask = X[col].isna() & X[col].isnull()
            X.loc[mask, col] = 0
            X.loc[~mask, col] = 1
            X[col].astype(np.int8)
        return X


class CountEncoder(BaseEstimator, TransformerMixin):
    """
    Data:
    0   [1,2,3]
    1   [1]
    2   []
    
    Result:
    0   3
    1   1
    2   0
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in self.columns:
            X['{}_count'.format(column)] = X[column].map(lambda x: len(x) if not isinstance(x, float) else 0)

        return X


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    """
    LabelEncoder over multiple indices while keeping the index.
    The encoder is returned to later be able to inverse the IDs using
    train_data[column] = encoder.inverse_transform(train_data[column])

    Data:
    id    column1     column2
    0     orange      apple
    1     apple       orange
    2     pear        orange

    Result:
    id    column1     column2
    0     0           1
    1     1           0
    2     2           0
    """
    def __init__(self, columns):
        self.columns = columns
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = preprocessing.LabelEncoder().fit(np.unique(X[self.columns].values))
        return self

    def transform(self, X):
        output = X.copy()
        for column in self.columns:
            output[column] = self.encoder.transform(output[column])
        return output, self.encoder


class TimestampEncoder(BaseEstimator, TransformerMixin):
    """
    Splits a datetime string or actual datetime in its parts and stores it as separate columns.

    Data:
    id  column1
    0   '2010-01-01 01:00:00'
    1   '2010-02-01 01:00:00'
    2   '2010-03-01 01:00:00'

    Result:
    id  column1_year   column1_month  ...  column1_second
    0   2010           01                  00
    1   2010           02                  00
    2   2010           03                  00
    """
    def __init__(self, columns, format=None, include_time=True):
        self.columns = columns
        self.include_time = include_time
        self.format = format

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in self.columns:
            tmp_data: pd.Series = pd.to_datetime(X[column], format=self.format)
            X['{}_year'.format(column)] = tmp_data.dt.year
            X['{}_month'.format(column)] = tmp_data.dt.month
            X['{}_day'.format(column)] = tmp_data.dt.day
            X['{}_weekday'.format(column)] = tmp_data.dt.weekday
            if self.include_time:
                X['{}_hour'.format(column)] = tmp_data.dt.hour
                X['{}_minute'.format(column)] = tmp_data.dt.minute
                X['{}_second'.format(column)] = tmp_data.dt.second

        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols: [str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols, inplace=False)


class MemoryReducer(BaseEstimator, TransformerMixin):
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        (df, _) = util.reduce_mem_usage(X, self.verbose)
        return df

# todo: add binning transformer (values get binned to create labels)
