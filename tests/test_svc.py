'''
A series of unit tests for mvpy.estimators.SVC
'''

import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mvpy as mv

import numpy as np
import torch
import scipy
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification

from typing import Any, Dict, Callable

# setup opts
N_REPEATS = 25
N_SPLITS = 5
F_THR_POS = 0.5

'''
Setup fixtures
'''

def _Xy_1F_2C():
    '''
    Generate data for testing.
    '''
    
    return make_classification(n_classes = 2, n_informative = 4)

def _Xy_1F_3C():
    '''
    Generate data for testing.
    '''
    
    return make_classification(n_classes = 3, n_informative = 6)

def _Xy_2F_nC():
    '''
    Generate data for testing.
    '''
    
    X0, y0 = make_classification(n_classes = 2, n_informative = 12)
    X1, y1 = make_classification(n_classes = 3, n_informative = 18)
    X = np.concatenate((X0, X1), axis = 1)
    y = np.concatenate((y0[:,None], y1[:,None]), axis = 1)
    
    return X, y

def _fit_clf(Xy: np.ndarray, backend: str = 'numpy', arguments: Dict[str, Any] = dict()):
    '''
    Transform data, fit and eval desired estimators.
    '''
    
    # setup kfold
    n_splits = N_SPLITS
    kf = StratifiedKFold(n_splits = n_splits)
    
    # unpack data
    X, y = Xy
    
    # check y
    if len(y.shape) == 1:
        y = y[:, None]
    
    # hash data
    L = np.array([','.join(y[i,:].astype(str)) for i in range(y.shape[0])])
    
    # if required, transform
    if backend == 'torch':
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    
    # setup desired classifiers and outputs
    clf = []
    
    if backend == 'torch':
        y_h = torch.zeros_like(y)
    else:
        y_h = np.zeros_like(y)
    
    # loop over folds
    for f_i, (train, test) in enumerate(kf.split(X, L)):
        # if required, transform
        if backend == 'torch':
            train = torch.from_numpy(train).long()
            test = torch.from_numpy(test).long()
        
        # fit and save
        clf_i = mv.estimators.SVC(**arguments).fit(
            X[train], y[train]
        )
        
        # predict oos
        y_h[test] = clf_i.predict(X[test])
    
    # compute accuracy
    accuracy = mv.math.accuracy(y_h.T, y.T)
    
    return X, y, y_h, clf, accuracy

def _multi_clf(f: Callable, backend: str = 'numpy', arguments: Dict[str, Any] = dict()):
    '''
    Fit a number of classifiers and report accuracy.
    '''
    
    # find number of features
    _, y = f()
    
    if len(y.shape) == 1:
        n_features = 1
    else:
        n_features = y.shape[1]
    
    # setup container
    accuracy = np.zeros((N_REPEATS, n_features))
    
    # if required, transform
    if backend == 'torch':
        accuracy = torch.from_numpy(accuracy).float()
    
    for i in range(N_REPEATS):
        _, _, _, _, accuracy[i] = _fit_clf(
            f(), 
            backend = backend,
            arguments = arguments
        )
    
    return accuracy.mean(0)

'''
Test SVC
'''

@pytest.mark.parametrize(
    "f,backend,arguments",
    [
        # 1F2C OvR
        (_Xy_1F_2C, 'numpy', dict(method = 'OvR', kernel = 'linear')),
        (_Xy_1F_2C, 'torch', dict(method = 'OvR', kernel = 'linear')),
        (_Xy_1F_2C, 'numpy', dict(method = 'OvR', kernel = 'rbf')),
        (_Xy_1F_2C, 'torch', dict(method = 'OvR', kernel = 'rbf')),
        (_Xy_1F_2C, 'numpy', dict(method = 'OvR', kernel = 'poly', degree = 1.0)),
        (_Xy_1F_2C, 'torch', dict(method = 'OvR', kernel = 'poly', degree = 1.0)),
        (_Xy_1F_2C, 'numpy', dict(method = 'OvR', kernel = 'sigmoid')),
        (_Xy_1F_2C, 'torch', dict(method = 'OvR', kernel = 'sigmoid')),
        # 1F2C OvO
        (_Xy_1F_2C, 'numpy', dict(method = 'OvO', kernel = 'linear')),
        (_Xy_1F_2C, 'torch', dict(method = 'OvO', kernel = 'linear')),
        (_Xy_1F_2C, 'numpy', dict(method = 'OvO', kernel = 'rbf')),
        (_Xy_1F_2C, 'torch', dict(method = 'OvO', kernel = 'rbf')),
        (_Xy_1F_2C, 'numpy', dict(method = 'OvO', kernel = 'poly', degree = 1.0)),
        (_Xy_1F_2C, 'torch', dict(method = 'OvO', kernel = 'poly', degree = 1.0)),
        (_Xy_1F_2C, 'numpy', dict(method = 'OvO', kernel = 'sigmoid')),
        (_Xy_1F_2C, 'torch', dict(method = 'OvO', kernel = 'sigmoid')),
        # 1F3C OvR
        (_Xy_1F_3C, 'numpy', dict(method = 'OvR', kernel = 'linear')),
        (_Xy_1F_3C, 'torch', dict(method = 'OvR', kernel = 'linear')),
        (_Xy_1F_3C, 'numpy', dict(method = 'OvR', kernel = 'rbf')),
        (_Xy_1F_3C, 'torch', dict(method = 'OvR', kernel = 'rbf')),
        (_Xy_1F_3C, 'numpy', dict(method = 'OvR', kernel = 'poly', degree = 1.0)),
        (_Xy_1F_3C, 'torch', dict(method = 'OvR', kernel = 'poly', degree = 1.0)),
        (_Xy_1F_3C, 'numpy', dict(method = 'OvR', kernel = 'sigmoid')),
        (_Xy_1F_3C, 'torch', dict(method = 'OvR', kernel = 'sigmoid')),
        # 1F3C OvO
        (_Xy_1F_3C, 'numpy', dict(method = 'OvO', kernel = 'linear')),
        (_Xy_1F_3C, 'torch', dict(method = 'OvO', kernel = 'linear')),
        (_Xy_1F_3C, 'numpy', dict(method = 'OvO', kernel = 'rbf')),
        (_Xy_1F_3C, 'torch', dict(method = 'OvO', kernel = 'rbf')),
        (_Xy_1F_3C, 'numpy', dict(method = 'OvO', kernel = 'poly', degree = 1.0)),
        (_Xy_1F_3C, 'torch', dict(method = 'OvO', kernel = 'poly', degree = 1.0)),
        (_Xy_1F_3C, 'numpy', dict(method = 'OvO', kernel = 'sigmoid')),
        (_Xy_1F_3C, 'torch', dict(method = 'OvO', kernel = 'sigmoid')),
        # 2FnC OvR
        (_Xy_2F_nC, 'numpy', dict(method = 'OvR', kernel = 'linear')),
        (_Xy_2F_nC, 'torch', dict(method = 'OvR', kernel = 'linear')),
        (_Xy_2F_nC, 'numpy', dict(method = 'OvR', kernel = 'rbf')),
        (_Xy_2F_nC, 'torch', dict(method = 'OvR', kernel = 'rbf')),
        (_Xy_2F_nC, 'numpy', dict(method = 'OvR', kernel = 'poly', degree = 1.0)),
        (_Xy_2F_nC, 'torch', dict(method = 'OvR', kernel = 'poly', degree = 1.0)),
        (_Xy_2F_nC, 'numpy', dict(method = 'OvR', kernel = 'sigmoid')),
        (_Xy_2F_nC, 'torch', dict(method = 'OvR', kernel = 'sigmoid')),
        # 2FnC OvO
        (_Xy_2F_nC, 'numpy', dict(method = 'OvO', kernel = 'linear')),
        (_Xy_2F_nC, 'torch', dict(method = 'OvO', kernel = 'linear')),
        (_Xy_2F_nC, 'numpy', dict(method = 'OvO', kernel = 'rbf')),
        (_Xy_2F_nC, 'torch', dict(method = 'OvO', kernel = 'rbf')),
        (_Xy_2F_nC, 'numpy', dict(method = 'OvO', kernel = 'poly', degree = 1.0)),
        (_Xy_2F_nC, 'torch', dict(method = 'OvO', kernel = 'poly', degree = 1.0)),
        (_Xy_2F_nC, 'numpy', dict(method = 'OvO', kernel = 'sigmoid')),
        (_Xy_2F_nC, 'torch', dict(method = 'OvO', kernel = 'sigmoid')),
    ],
    ids = [
        # 1F2C OvR
        '1F2C_numpy_OvR_linear',
        '1F2C_torch_OvR_linear',
        '1F2C_numpy_OvR_rbf',
        '1F2C_torch_OvR_rbf',
        '1F2C_numpy_OvR_poly',
        '1F2C_torch_OvR_poly',
        '1F2C_numpy_OvR_sigmoid',
        '1F2C_torch_OvR_sigmoid',
        # 1F2C OvO
        '1F2C_numpy_OvO_linear',
        '1F2C_torch_OvO_linear',
        '1F2C_numpy_OvO_rbf',
        '1F2C_torch_OvO_rbf',
        '1F2C_numpy_OvO_poly',
        '1F2C_torch_OvO_poly',
        '1F2C_numpy_OvO_sigmoid',
        '1F2C_torch_OvO_sigmoid',
        # 1F3C OvR
        '1F3C_numpy_OvR_linear',
        '1F3C_torch_OvR_linear',
        '1F3C_numpy_OvR_rbf',
        '1F3C_torch_OvR_rbf',
        '1F3C_numpy_OvR_poly',
        '1F3C_torch_OvR_poly',
        '1F3C_numpy_OvR_sigmoid',
        '1F3C_torch_OvR_sigmoid',
        # 1F3C OvO
        '1F3C_numpy_OvO_linear',
        '1F3C_torch_OvO_linear',
        '1F3C_numpy_OvO_rbf',
        '1F3C_torch_OvO_rbf',
        '1F3C_numpy_OvO_poly',
        '1F3C_torch_OvO_poly',
        '1F3C_numpy_OvO_sigmoid',
        '1F3C_torch_OvO_sigmoid',
        # 2FnC OvR
        '2FnC_numpy_OvR_linear',
        '2FnC_torch_OvR_linear',
        '2FnC_numpy_OvR_rbf',
        '2FnC_torch_OvR_rbf',
        '2FnC_numpy_OvR_poly',
        '2FnC_torch_OvR_poly',
        '2FnC_numpy_OvR_sigmoid',
        '2FnC_torch_OvR_sigmoid',
        # 2FnC OvO
        '2FnC_numpy_OvO_linear',
        '2FnC_torch_OvO_linear',
        '2FnC_numpy_OvO_rbf',
        '2FnC_torch_OvO_rbf',
        '2FnC_numpy_OvO_poly',
        '2FnC_torch_OvO_poly',
        '2FnC_numpy_OvO_sigmoid',
        '2FnC_torch_OvO_sigmoid',
    ]
)
def test_SVC(f: Callable, backend: str, arguments: Dict[str, Any]):
    '''
    Run our tests in batch, parameterised by pytest.
    '''
    
    # compute
    accuracy = _multi_clf(
        f, backend = backend, arguments = arguments
    )
    
    assert (accuracy >= F_THR_POS).all(), f"Accuracy: {accuracy} < {F_THR_POS:.2f}"

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])