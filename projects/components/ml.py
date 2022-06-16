from xmlrpc.client import Boolean
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV,StratifiedKFold
from sklearn.svm import SVC
from IPython.display import display
import pandas as pd
import os
from time import time
from typing import Tuple
import logging

def experimental_setup_interface(X:np.ndarray,y,groups,cv_result_prefix="") -> np.ndarray:
    return X

def _check_is_done(checkpoint: str) -> Boolean:
    if(os.path.exists(checkpoint) == True):
        return True
    return False


def train_model_segment_first(X,y,groups,cv_result_prefix="") -> np.ndarray:
    """
        X: The sample set with shape of (n_samples, n_features)
        y: The labels with shape of (n_samples,)
        groups: A data indicates which trials the samples belong to (n_samples,)
    """
    # Here we have to perform double cross validation.
    # The idea is we will do 10-CV
    # The option we have to choose is split-first or segment-first
    # If we do "Split-first", then models learn little record of every trial
    # If we doe "Segment-first", then models learn 90% of trials and test on 10%
    # This split-first can be done using "group" argument during split.
    logging.info(f"Running Segment-First")
    logging.info(f"X.shape={X.shape}, y.shape={y.shape}, groups.shape={groups.shape}")
    n_split_outter = 10
    cv_outter = StratifiedGroupKFold(n_splits=n_split_outter, shuffle=False)
    accs = []
    for epoch, (idxs_train, idxs_test) in enumerate(cv_outter.split(X,y,groups)):
        start = time()
        print(f"BEGIN EPOCH: {epoch+1}/{n_split_outter}")
        filename = f"{cv_result_prefix}-{epoch+1}.csv"
        if(_check_is_done(filename)): continue
        X_train, X_test = X[idxs_train], X[idxs_test]
        y_train, y_test = y[idxs_train], y[idxs_test]
        groups_train, groups_test = groups[idxs_train], groups[idxs_test]
        assert set(groups_train).isdisjoint(set(groups_test)),f"Contaminated.\ngroups_train:{groups_train}\ngroups_test:{groups_test}"

        grid = build_model(X_train,y_train,groups_train)
        # Evaluation
        model = grid.best_estimator_
        predict = model.predict(X_test) # type: ignore
        acc = sum(predict == y_test) / len(y_test)
        accs.append(acc)
        # save csv
        logging.info(f"{epoch+1}/{n_split_outter}|grid.best_params_={grid.best_params_}, grid.best_score_={grid.best_score_}, grid.best_index_={grid.best_index_}, acc={acc}, time={time()-start}" )
        pd.DataFrame(grid.cv_results_).to_csv(filename)
    return np.array(accs)

def train_model_split_first(X,y,groups,cv_result_prefix="") -> np.ndarray:
    """
        X: The sample set with shape of (n_samples, n_features)
        y: The labels with shape of (n_samples,)
        groups: will be ignored
    """
    # Here we have to perform double cross validation.
    # The idea is we will do 10-CV
    # The option we have to choose is split-first or segment-first
    # If we do "Split-first", then models learn little record of every trial
    # If we doe "Segment-first", then models learn 90% of trials and test on 10%
    # This split-first can be done using "group" argument during split.
    logging.info(f"Running Split-First")
    logging.info(f"X.shape={X.shape}, y.shape={y.shape}, groups.shape={groups.shape}")
    n_split_outter = 10
    cv_outter = StratifiedKFold(n_splits=n_split_outter, shuffle=False)
    accs = []
    for epoch, (idxs_train, idxs_test) in enumerate(cv_outter.split(X,y)):
        start = time()
        print(f"BEGIN EPOCH: {epoch+1}/{n_split_outter}")
        filename = f"{cv_result_prefix}-{epoch+1}.csv"
        if(_check_is_done(filename)): continue
        X_train, X_test = X[idxs_train], X[idxs_test]
        y_train, y_test = y[idxs_train], y[idxs_test]

        grid = build_model(X_train,y_train)
        # Evaluation
        model = grid.best_estimator_
        predict = model.predict(X_test) # type: ignore
        acc = sum(predict == y_test) / len(y_test)
        accs.append(acc)
        # save csv
        logging.info(f"{epoch+1}/{n_split_outter}|grid.best_params_={grid.best_params_}, grid.best_score_={grid.best_score_}, grid.best_index_={grid.best_index_}, acc={acc}, time={time()-start}" )
        pd.DataFrame(grid.cv_results_).to_csv(filename)
    return np.array(accs)

def build_model(X,y,groups=None) -> GridSearchCV:
    """
        This function will only optimized models
    """
    is_groups_none = type(groups) == type(None) # type: ignore
    logging.info(f"Is groups None: {is_groups_none} ") 

    n_split = 9
    if(is_groups_none == True):
        cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)
    else:
        cv = StratifiedGroupKFold(n_splits=n_split, shuffle=True, random_state=42)
    # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    C_range = np.logspace(-2, 10, 7)
    gamma_range = np.logspace(-9, 3, 7)
    tuned_parameters = [
            {"kernel": ["rbf"],    "C": C_range, "max_iter":[1000],  "gamma": gamma_range},
        ]
    grid = GridSearchCV(SVC(), param_grid=tuned_parameters, cv=cv, n_jobs=os.cpu_count(), refit=True, verbose=4, return_train_score=True)
    if(is_groups_none == True):
        grid.fit(X=X, y=y)
    else:
        grid.fit(X=X, y=y, groups=groups)
    return grid