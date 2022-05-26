import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV
from sklearn.svm import SVC
from IPython.display import display
import pandas as pd
import os
from typing import Tuple

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
    n_split_outter = 10
    cv_outter = StratifiedGroupKFold(n_splits=n_split_outter, shuffle=False)
    accs = []
    for epoch, (idxs_train, idxs_test) in enumerate(cv_outter.split(X,y,groups)):
        print(f"BEGIN EPOCH: {epoch+1}/{n_split_outter}")
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
        pd.DataFrame(grid.cv_results_).to_csv(f"{cv_result_prefix}-{epoch+1}.csv")
    return np.array(accs)

def build_model(X,y,groups) -> GridSearchCV:
    """
        This function will only optimized models
    """
    n_split = 9
    cv = StratifiedGroupKFold(n_splits=n_split, shuffle=True, random_state=42)
    # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    tuned_parameters = [
            {"kernel": ["linear"], "C": C_range},
            {"kernel": ["rbf"],    "C": C_range, "gamma": gamma_range},
        ]
    grid = GridSearchCV(SVC(), param_grid=tuned_parameters, cv=cv, n_jobs=os.cpu_count(), refit=True, verbose=1, return_train_score=True)
    grid.fit(X=X, y=y, groups=groups)
    # df = pd.DataFrame(grid.cv_results_)
    # return grid.best_estimator_, df
    return grid