import sys

import numpy as np
import copy
from scipy import stats
from sklearn import neighbors
from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import pairwise_distances
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import catboost
import lightgbm

#SGDRegressor -> pre-implemented Linear SG
#MultiLinearSGD -> self-implemented Linear SGD

def get_estimator(name: str):
    class_obj = getattr(sys.modules[__name__], (name + "Helper"))
    return class_obj()


class BaseHelper:
    """
    This class implements a basic framework for training various classifiers.
    Any methods should be adapted in the subclasses to implement the correct behaviour for different models
    """
    def __init__(self, name: str):
        self.reg = None
        self.grid_parameters = None
        self.random_parameters = None
        self.name = name
        self.use_parameter_search = None

    def get_estimator(self):
        return self.reg

    def get_grid_parameters(self):
        return self.grid_parameters

    def set_grid_parameters(self, parameters):
        self.grid_parameters = parameters

    def get_random_parameters(self):
        return self.random_parameters

    def set_random_parameters(self, parameters):
        self.random_parameters = parameters

    def predict(self, model, test):
        """
        Predicts the actual ratings for the given test inputs.
        For example, for binary classification the output will be an array of 0s and 1s
        depending how the input is classified
        :param model: A model that can predict the input
        :param test: The set of examples the model should predict
        :return: The predictions of the model as an array
        """
        return model.predict(test)

    def get_name(self):
        return self.name

    def set_name(self, name: str):
        self.name = name


class KNeighborsHelper(BaseHelper):
    def __init__(self):
        super().__init__("KNeighbors")
        self.reg = neighbors.KNeighborsRegressor(algorithm="brute")
        self.random_parameters = [
            {"n_neighbors": stats.randint(1, 20)}
        ]
        self.grid_parameters = [
            {"n_neighbors": [1, 2, 5, 10, 20],
             "p": [1, 2]}
        ]

class LightGBMHelper(BaseHelper):
    def __init__(self):
        super(LightGBMHelper, self).__init__("LightGBM")
        self.reg = lightgbm.LGBMRegressor()
        self.random_parameters = [
            {"n_estimators": stats.randint(50, 1200), "colsample_bytree": [1, 0.9, 0.8, 0.5, 0.4],
             "eta": stats.expon(scale=.2), "max_depth": stats.randint(1, 12),
             # "lambda": stats.uniform(0.0, 2.0), "alpha": stats.uniform(0.0, 2.0),
             "min_child_weight": stats.randint(1, 3)}
        ]


class CatBoostHelper(BaseHelper):
    def __init__(self):
        super(CatBoostHelper, self).__init__("CatBoost")
        self.reg = catboost.CatBoostRegressor()
        self.random_parameters = [
            {"n_estimators": stats.randint(50, 1200), "eta": stats.expon(scale=.2), "max_depth": stats.randint(1, 12),
             "reg_lambda": stats.uniform(0.0, 2.0)}]


class SGDRegressorHelper(BaseHelper):
    def __init__(self):
        super(SGDRegressorHelper, self).__init__("SGDRegressor")
        # learning rate is constant to be comparable to custom implementation (eta0 is initial learning rate)
        self.reg = linear_model.SGDRegressor(loss="squared_loss", penalty='l2', learning_rate='constant')
        self.random_parameters = [
            {"alpha": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0], "eta0": [0.1, 0.01, 0.001, 0.0001]},
            {"alpha": stats.expon(scale=.1), "eta0": stats.expon(scale=.1)},
        ]
        self.grid_parameters = [
            {"alpha": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0],
             "eta0": [0.1, 0.01, 0.001, 0.0001, 0.00001]}
        ]


class XGBoostHelper(BaseHelper):
    def __init__(self):
        super(XGBoostHelper, self).__init__("XGBoost")
        self.reg = XGBRegressor()
        self.random_parameters = [
            {"n_estimators": stats.randint(50, 1200), "colsample_bytree": [1, 0.9, 0.8, 0.5, 0.4],
             "eta": stats.expon(scale=.2), "max_depth": stats.randint(1, 12), "gamma": [0, 2, 4],
             # "lambda": stats.uniform(0.0, 2.0), "alpha": stats.uniform(0.0, 2.0),
             "min_child_weight": stats.randint(1, 3)}
        ]

class MultiXGBoostHelper(BaseHelper):
    def __init__(self):
        super(MultiXGBoostHelper, self).__init__("MultiXGBoost")
        self.reg = MultiOutputRegressor(XGBRegressor())
        self.random_parameters = [
            {"estimator__n_estimators": stats.randint(50, 1200), "estimator__colsample_bytree": [1, 0.9, 0.8, 0.5, 0.4],
             "estimator__eta": stats.expon(scale=.2), "estimator__max_depth": stats.randint(1, 12), "estimator__gamma": [0, 2, 4],
             # "lambda": stats.uniform(0.0, 2.0), "alpha": stats.uniform(0.0, 2.0),
             "estimator__min_child_weight": stats.randint(1, 3)}
        ]
        self.grid_parameters = [
            {"estimator__n_estimators": np.arange(50, 1200, 1), "estimator__colsample_bytree": [1, 0.9, 0.8, 0.5, 0.4],
             "estimator__eta": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001], "estimator__max_depth": np.arange(1, 12,1), "estimator__gamma": [0, 2, 4],
             # "lambda": stats.uniform(0.0, 2.0), "alpha": stats.uniform(0.0, 2.0),
             "estimator__min_child_weight": np.arange(1, 5, 1)}
        ]

class MultiGradientBoostHelper(BaseHelper):
    def __init__(self):
        super(MultiGradientBoostHelper, self).__init__("MultiGradientBoost")
        self.reg = MultiOutputRegressor(XGBRegressor())
        self.random_parameters = [
            {"n_estimators": stats.randint(50, 1200), "colsample_bytree": [1, 0.9, 0.8, 0.5, 0.4],
             "eta": stats.expon(scale=.2), "max_depth": stats.randint(1, 12), "gamma": [0, 2, 4],
             # "lambda": stats.uniform(0.0, 2.0), "alpha": stats.uniform(0.0, 2.0),
             "min_child_weight": stats.randint(1, 3)}
        ]

class MultiSVRHelper(BaseHelper):
    def __init__(self):
        super(MultiSVRHelper, self).__init__("MultiSVR")
        self.reg = MultiOutputRegressor(SVR())
        self.random_parameters = [
            {'estimator__kernel': ['rbf', 'linear'], 'estimator__C': [10**i for i in np.arange(-2, 5, 0.25)], 'estimator__gamma': [1,0.1,0.01]}
        ]
        self.grid_parameters = [
            {'estimator__kernel': ['rbf'], 'estimator__C': [10**i for i in np.arange(-2, 2, 0.5)], 'estimator__gamma': [1,0.1,0.01,0.001,0.0001]}
        ]

class MultiKNNHelper(BaseHelper):
    def __init__(self):
        super(MultiKNNHelper, self).__init__("MultiKNN")
        self.reg = MultiOutputRegressor(KNeighborsRegressor())
        self.random_parameters = [
            {"estimator__leaf_size": stats.randint(10, 100), "estimator__n_neighbors": stats.randint(1,30),
             "estimator__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'], "estimator__p": [1, 2]}
        ]
        self.grid_parameters = [
            {"estimator__leaf_size": np.arange(10,101,1), "estimator__n_neighbors": np.arange(1,31,1),
             "estimator__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'], "estimator__p": [1, 2]}
        ]


class LassoHelper(BaseHelper):
    def __init__(self):
        super(LassoHelper, self).__init__("Lasso")
        self.reg = linear_model.Lasso()
        self.random_parameters = [
            {"alpha": [1, 0.5, 0.25, 0.1, 0.01, 0.001]},
            {"alpha": stats.expon(scale=.1)},
        ]
        
class DecisionTreeHelper(BaseHelper):
    def __init__(self):
        super(DecisionTreeHelper, self).__init__("DecisionTree")
        self.reg = tree.DecisionTreeRegressor()
        self.random_parameters = [
            {"dec_tree__criterion": ['gini', 'entropy', 'log_loss']},
            {"splitter": ["best", "random"]},
            {"max_depth": np.arange(1,30)},
            {'min_samples_leaf': [1,2,4,5,10,20,30,40,80,100]}
        ]
        
class RandomForestHelper(BaseHelper):
    def __init__(self):
        super(RandomForestHelper, self).__init__("RandomForest")
        self.reg = ensemble.RandomForestRegressor()
        self.random_parameters = [
            {"n_estimators": [10,25,50,100,200,400,600,800,1000,1200,1400,1600,1800]},
            {"criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"]},
            {"max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]},
            {'min_samples_leaf': [1,2,4,5,10,20,30,40,80,100]},
            {'bootstrap': [True, False]},
            {"max_features": ['auto', 'sqrt']},
            {"max_features": ['auto', 'sqrt']},
            {'min_samples_split': [1,2,4,5,10,20,30,40,80,100]}
        ]
        
class AdaBoostHelper(BaseHelper):
    def __init__(self):
        super(AdaBoostHelper, self).__init__("AdaBoost")
        self.reg = ensemble.AdaBoostRegressor()
        self.random_parameters = [
            {"n_estimators": [10,25,50,100,200,400,600,800,1000,1200,1400,1600,1800]},
            {"learning_rate": [0.001,0.005,0.01,0.05,0.1,0.5,1.0,1.5,2.0,1.5,5.0]},
            {"loss": ["linear", "square", "exponential"]},
            {"random_state": [1,2,3,4,5,6,7,8,9]},
            {'base_estimator': ["DecisionTreeRegressor", "LinearRegression", "SVR", "Ridge"]}
        ]
        
       

        
class RidgeHelper(BaseHelper):
    def __init__(self):
        super(RidgeHelper, self).__init__("Ridge")
        self.reg = linear_model.Ridge()
        self.random_parameters = [
            {"alpha": [1, 0.5, 0.25, 0.1, 0.01, 0.001]},
            {"alpha": stats.expon(scale=.1)},
        ]


class MLPHelper(BaseHelper):
    def __init__(self):
        super(MLPHelper, self).__init__("MLP")
        self.reg = MLPRegressor()
        self.random_parameters = [
            {"hidden_layer_sizes": [(10,10), (20,20), (50,50), (50,100), (5,5)]},
            {"activation": ['relu', 'tanh', 'siogmoid']},
            {"batch_size": [2, 4, 8, 16, 32]},
            {"alpha": [0.1, 0.25, 0.5, 0.01, 0.025, 0.05, 0.001, 0.0025, 0.005]},
            {"solver": ['adam', 'sgd']},
            {"learning_rate_init": [0.1, 0.25, 0.5, 0.01, 0.025, 0.05, 0.001, 0.0025, 0.005]},
            {"max_iter": [100, 250, 500]}
        ]


class SVRHelper(BaseHelper):
    def __init__(self):
        super(SVRHelper, self).__init__("SVR")
        self.reg = SVR()
        self.random_parameters = [
            {'kernel': ['rbf', 'linear'], 'C': [10**i for i in np.arange(-2, 5, 0.25)], 'gamma': [1,0.1,0.01]}
        ]
        self.grid_parameters = [
            {'kernel': ['rbf'], 'C': [10**i for i in np.arange(-2, 2, 0.5)], 'gamma': [1,0.1,0.01,0.001,0.0001]}
        ]


class LinearSGD(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.w0 = 0.1
        self.w1 = 0.1
        self.iterations = 1000


    def set_params(self, **parameters):
        if parameters.get("learning_rate", None):
            if parameters['learning_rate'] <= 0.0:
                raise ValueError("learning rate cannot be smaller than 0")

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X_in, y_in) -> None:

        X = copy.deepcopy(np.array(X_in))
        y = copy.deepcopy(np.array(y_in))


        def RSS(x_arr, y_arr, w0 ,w1):
            RSS_val = 0
            for i in range(len(y)):
                RSS_val = RSS_val + (pow((y_arr[i] - (w0 + w1 * x_arr[i])),2))/len(y_arr)
            return RSS_val

        def Dw0(x_arr, y_arr, w0 ,w1):
            Dw0_val = 0
            for i in range(len(y)):
                Dw0_val = Dw0_val + ((-2) * (y_arr[i] - (w0 + w1 * x_arr[i])))/len(y_arr)
            return Dw0_val

        def Dw1(x_arr, y_arr, w0 ,w1):
            Dw1_val = 0
            for i in range(len(y)):
                Dw1_val = Dw1_val + ((-2) * x_arr[i] * (y_arr[i] - (w0 + w1 * x_arr[i])))/len(y_arr)
            return Dw1_val

        for i in range(self.iterations):
            w0_save = self.w0 - self.learning_rate * Dw0(X,y,self.w0, self.w1 )
            w1_save = self.w1 - self.learning_rate * Dw1(X,y,self.w0, self.w1 )
            self.w0 = w0_save
            self.w1 = w1_save

        pass

    def predict(self, X):

        def lin_func(x_in, w0, w1):
            x_arr = copy.deepcopy(np.array(x_in))
            y_arr = list()
            for i in range(len(x_arr)):
                yi = copy.deepcopy(w0 + w1 * x_arr[i])
                y_arr.append(yi[0])
            y_arr = copy.deepcopy(np.array(y_arr))
            return y_arr

        return lin_func(X, self.w0, self.w1)


class MultiLinearSGD(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate: float = 0.1, verbose=False):
        self.learning_rate = learning_rate
        self.w0 = 0.1
        self.eta = 0.1
        self.wi = np.zeros(2)
        self.iterations = 1000
        self.speed_up = False
        self.verbose_print = print if verbose else lambda *a, **k: None

    def set_params(self, **parameters):
        if parameters.get("learning_rate", None):
            if parameters['learning_rate'] <= 0.0:
                raise ValueError("learning rate cannot be smaller than 0")

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X_in, y_in) -> None:
        X = copy.deepcopy(np.array(X_in))
        y = copy.deepcopy(np.array(y_in))
        self.n_dim = len(X[0,:])
        self.wi = np.zeros((self.n_dim+1))
        check_arr = np.zeros(10)

        def highest_number_of_duplicates(array):
            array_int = array*1000
            array_int = np.array(array_int, dtype=int)
            counter_overall = 0
            for i in range(len(array_int)-1):
                counter=0
                for ii in range(len(array_int)):
                    if array_int[i] == array_int[ii]:
                        counter = counter +1
                if counter > counter_overall:
                    counter_overall = counter
            return counter_overall

        def RSSndim(x_arr, y_arr, wi):
            np_sums = np.dot(x_arr, wi[1:])

            NORM = (len(y_arr) * (len(wi) - 1))
            def new_fRSS(sums):
                vals = (np.power((y_arr - (wi[0] + sums)), 2)) / NORM
                return np.sum(vals)

            return new_fRSS(np_sums)

        def Dw0(x_arr, y_arr, wi):
            np_sums = np.dot(x_arr, wi[1:])

            def new_f0(sums):
                vals = ((-2) * (y_arr - (wi[0] + sums))) / (len(y_arr) * (len(wi) - 1))
                return np.sum(vals)

            return new_f0(np_sums)

        def calc_wi_save(x_arr: np.ndarray, y_arr: np.ndarray, wi: np.ndarray, wi_save: np.ndarray, learning_rate: float):
            np_sums = np.dot(x_arr, wi[1:])
            NORM = (len(y_arr) * (len(wi) - 1))

            def new_fw(sums):
                return np.sum((-2 * x_arr * (y_arr - (wi[0] + sums)).reshape((len(y_arr), 1))) / NORM, axis=0)

            new_wi = new_fw(np_sums)
            new_wi_save = wi_save[1:] - learning_rate * new_wi
            wi_save[1:] = new_wi_save

            return wi_save

        self.verbose_print('current RSS:')
        self.verbose_print(RSSndim(X, y, self.wi))
        switch = False
        for i in range(self.iterations):
            wi_save = copy.deepcopy(self.wi)
            wi_save[0] = self.wi[0] - self.learning_rate * Dw0(X, y, self.wi)
            # for ii in range((len(self.wi)-1)):
            #     wi_save[ii+1] = self.wi[ii+1] - self.learning_rate * Dwi(X, y, self.wi, ii)
            wi_save = calc_wi_save(X, y, self.wi, wi_save, self.learning_rate)
            self.wi = copy.deepcopy(wi_save)
            self.verbose_print('current RSS:')
            eta_check = copy.deepcopy(RSSndim(X, y, self.wi))
            #edit checker array
            if self.speed_up:
                check_arr = copy.deepcopy(np.roll(check_arr,1))
                check_arr[0] = copy.deepcopy(eta_check)
                if i > 10:
                    hnod = highest_number_of_duplicates(check_arr)
                    if hnod > 4:
                        switch = True
                if switch:
                    break
                if eta_check <= self.eta:
                    break
        pass

    def predict(self, X):
        return self.wi[0] + np.dot(X, self.wi[1:])


class KNN(BaseEstimator, RegressorMixin):
    def __init__(self, k: int = 5, metric: str = 'minkowski', p: int = 2, n_jobs: int = -1):
        self.k = k
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs
        self.train_X = None
        self.train_y = None

    def set_params(self, **parameters):
        if parameters.get("k", None):
            if parameters['k'] < 1:
                raise ValueError("k cannot be smaller than 1")

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y) -> None:
        self.train_X = X
        self.train_y = y
        pass

    def predict(self, X):
        predictions = list()
        dist = pairwise_distances(X, self.train_X, metric=self.metric, p=self.p)
        for i in range(dist.shape[0]):
            sorted_ind = np.argsort(dist[i, :])
            predictions.append(np.mean(self.train_y.iloc[sorted_ind[0:self.k]]))

        return np.array(predictions)


class KNNHelper(BaseHelper):
    def __init__(self):
        super().__init__("KNN")
        self.reg = KNN()
        self.random_parameters = [
            {"k": stats.randint(1, 20),
             "p": [1, 2]}
        ]
        self.grid_parameters = [
            {"k": [1, 2, 5, 10, 20],
             "p": [1, 2]}
        ]


class LinearSGDHelper(BaseHelper):
    def __init__(self):
        super(LinearSGDHelper, self).__init__("LinearSGD")
        # learning rate is constant to be comparable to custom implementation (eta0 is initial learning rate)
        self.reg = LinearSGD()
        self.random_parameters = [
            {"learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001]}
        ]
        self.grid_parameters = [
            {"learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001]}
        ]


class MultiLinearSGDHelper(BaseHelper):
    def __init__(self):
        super(MultiLinearSGDHelper, self).__init__("MultiLinearSGD")
        # learning rate is constant to be comparable to custom implementation (eta0 is initial learning rate)
        self.reg = MultiLinearSGD()
        self.random_parameters = [
            {"learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001]},
            {"learning_rate": stats.expon(scale=.1)}
        ]
        self.grid_parameters = [
            {"learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001]}
        ]
