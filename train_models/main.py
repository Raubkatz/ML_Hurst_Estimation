import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold

import data_preperation as data_preperation
import util as util
from regressors import *
from persistence_util import save_model, save_predictions, save_cv_results
from scores_util import *
from sklearn.decomposition import PCA


def main():
    args = util.parse_arguments()
    test_name = args.test_name
    rows, verbosity = args.nrows, args.verbosity
    verbose_print = print if verbosity else lambda *a, **k: None

    X_train, X_test, y_train, y_test, pre_processing_pipeline, pca = data_preperation.prepare_data(
        args.dataset, args,
        test_size=0.2)

    if isinstance(X_train, DataFrame):
        verbose_print("Memory usage of dataframe is :", X_train.memory_usage().sum() / 1024 ** 2, " MB")

    verbose_print("Shape ot training data: ", X_train.shape)
    verbose_print("Beginning Training...")
    reg: BaseHelper = get_estimator(args.estimator)

    grid_score = "r2"
    skf = KFold(n_splits=5, shuffle=True)
    if args.search == "grid":
        param_search = GridSearchCV(reg.get_estimator(), reg.get_grid_parameters(),
                                    cv=skf, scoring=grid_score, n_jobs=args.jobs,
                                    verbose=verbosity, return_train_score=True)
    else:
        param_search = RandomizedSearchCV(reg.get_estimator(), reg.get_random_parameters(),
                                          n_iter=args.iterations, cv=skf, scoring=grid_score, n_jobs=args.jobs,
                                          verbose=verbosity, return_train_score=True)
    param_search.fit(X_train, y_train)
    verbose_print("Finished")

    verbose_print("CV Score: ", param_search.best_score_)
    verbose_print("Parameters: ", param_search.best_params_)
    verbose_print("Print: array ", param_search)
    model = param_search.best_estimator_
    best_params = param_search.best_params_

    previous_best = get_cv_score(test_name, reg.get_name())
    if param_search.best_score_ < previous_best:
        print("\nModel has lower cross validation score than best previous")
        print("Best previous: ", previous_best, " Current: ", param_search.best_score_,
              "\nDiff ", previous_best - param_search.best_score_)
        print("Exiting...")

        return

    print("\nModel has higher cross validation score than best previous")
    print("Best previous: ", previous_best, " Current: ", param_search.best_score_, "\nDiff ", previous_best - param_search.best_score_)
    update_cv_scores(test_name, reg.get_name(), param_search.best_score_)
    save_cv_results(test_name, reg.get_name(), param_search.cv_results_)

    # save models in different directories depending on test
    #if args.persist_model:
    save_model(test_name, reg.get_name(), model)

    y_pred = reg.predict(model, X_test)

    # create submission file when applicable
    if args.dataset in {}:
        test_data = data_preperation.read_test_data(args.dataset, args.nrows)
        test_data = pre_processing_pipeline.transform(test_data)
        if pca is not None:
            test_data = pca.transform(test_data)
        test_pred = reg.predict(model, test_data)

    # save scores to json file
    write_scores(test_name, reg.get_name(),
                 y_true=y_test, y_pred=y_pred,
                 params=best_params, time=param_search.refit_time_)

    save_predictions(test_name, reg.get_name(), y_test, y_pred)

    verbose_print("Done")


if __name__ == '__main__':
    main()
