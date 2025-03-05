'''
A series of unit tests for mvpy.estimators.Decoder
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

from typing import Any

# setup tolerance for np.allclose
_ALLCLOSE_RTOL = 1e-3
_ALLCLOSE_ATOL = 1e-3

'''
Setup fixtures
'''

@pytest.fixture
def Xy():
    '''
    Generate data for testing.
    '''
    
    # setup dims
    N, C, L = 200, 60, 5
    
    # generate data
    X = np.random.randn(N, C)
    y = np.random.randint(low = 0, high = L, size = (N,))
    
    # seperate data
    unq = np.unique(y)
    for i, unq_i in enumerate(unq):
        indc = y == unq_i
        X[indc] += np.random.normal(loc = i, scale = 1.0, size = (1, C))
    
    return (X, y)

'''
Test Classifiers in OvR
'''

def test_Classifier_OvR_numpy(Xy):
    '''
    Make sure numpy classifiers work.
    '''
    
    # unpack data
    X, y = Xy
    n_folds = 20
    
    # setup classifier
    kf = sklearn.model_selection.StratifiedKFold(n_splits = n_folds)
    clf = mv.estimators.Classifier(alphas = np.logspace(-5, 10, 20), method = 'OvR')
    
    # run tests
    oos = np.zeros((n_folds,))
    for f_i, (train, test) in enumerate(kf.split(X, y)):
        clf.fit(X[train], y[train])
        oos[f_i] = mv.math.spearmanr(clf.predict(X[test]), y[test])
    
    assert oos.mean() > 0.95
    
def test_Classifier_OvR_torch(Xy):
    '''
    Make sure torch classifiers work.
    '''
    
    # unpack data
    X, y = Xy
    X, y = torch.from_numpy(X).to(torch.float32), torch.from_numpy(y).to(torch.float32)
    n_folds = 20
    
    # setup classifier
    kf = sklearn.model_selection.StratifiedKFold(n_splits = n_folds)
    clf = mv.estimators.Classifier(alphas = torch.logspace(-5, 10, 20), method = 'OvR')
    
    # run tests
    oos = torch.zeros((n_folds,))
    for f_i, (train, test) in enumerate(kf.split(X, y)):
        clf.fit(X[train], y[train])
        oos[f_i] = mv.math.spearmanr(clf.predict(X[test]), y[test])
    
    assert oos.mean() > 0.95

def test_Classifier_OvR_compare_numpy_torch(Xy):
    '''
    Make sure numpy and torch classifiers are the same.
    '''

    # unpack data
    X_np, y_np = Xy
    X_tr, y_tr = torch.from_numpy(X_np).to(torch.float32), torch.from_numpy(y_np).to(torch.float32)
    n_folds = 20
    
    # setup classifiers
    kf = sklearn.model_selection.StratifiedKFold(n_splits = n_folds)
    clf_np = mv.estimators.Classifier(alphas = np.logspace(-5, 10, 20), method = 'OvR')
    clf_tr = mv.estimators.Classifier(alphas = torch.logspace(-5, 10, 20), method = 'OvR')

    # run tests
    oos = np.zeros((n_folds,))
    for f_i, (train, test) in enumerate(kf.split(X_np, y_np)):
        clf_np.fit(X_np[train], y_np[train])
        clf_tr.fit(X_tr[train], y_tr[train])
        
        oos[f_i] = mv.math.spearmanr(clf_np.predict(X_np[test]), clf_tr.predict(X_tr[test]).cpu().numpy())
    
    assert oos.mean() > 0.95

'''
Test Classifiers in OvO
'''

def test_Classifier_OvO_numpy(Xy):
    '''
    Make sure numpy classifiers work.
    '''
    
    # unpack data
    X, y = Xy
    n_folds = 20
    
    # setup classifier
    kf = sklearn.model_selection.StratifiedKFold(n_splits = n_folds)
    clf = mv.estimators.Classifier(alphas = np.logspace(-5, 10, 20), method = 'OvO')
    
    # run tests
    oos = np.zeros((n_folds,))
    for f_i, (train, test) in enumerate(kf.split(X, y)):
        clf.fit(X[train], y[train])
        oos[f_i] = mv.math.spearmanr(clf.predict(X[test]), y[test])
    
    assert oos.mean() > 0.95
    
def test_Classifier_OvO_torch(Xy):
    '''
    Make sure torch classifiers work.
    '''
    
    # unpack data
    X, y = Xy
    X, y = torch.from_numpy(X).to(torch.float32), torch.from_numpy(y).to(torch.float32)
    n_folds = 20
    
    # setup classifier
    kf = sklearn.model_selection.StratifiedKFold(n_splits = n_folds)
    clf = mv.estimators.Classifier(alphas = torch.logspace(-5, 10, 20), method = 'OvO')
    
    # run tests
    oos = torch.zeros((n_folds,))
    for f_i, (train, test) in enumerate(kf.split(X, y)):
        clf.fit(X[train], y[train])
        oos[f_i] = mv.math.spearmanr(clf.predict(X[test]), y[test])
    
    assert oos.mean() > 0.95

def test_Classifier_OvO_compare_numpy_torch(Xy):
    '''
    Make sure numpy and torch classifiers are the same.
    '''

    # unpack data
    X_np, y_np = Xy
    X_tr, y_tr = torch.from_numpy(X_np).to(torch.float32), torch.from_numpy(y_np).to(torch.float32)
    n_folds = 20
    
    # setup classifiers
    kf = sklearn.model_selection.StratifiedKFold(n_splits = n_folds)
    clf_np = mv.estimators.Classifier(alphas = np.logspace(-5, 10, 20), method = 'OvO')
    clf_tr = mv.estimators.Classifier(alphas = torch.logspace(-5, 10, 20), method = 'OvO')

    # run tests
    oos = np.zeros((n_folds,))
    for f_i, (train, test) in enumerate(kf.split(X_np, y_np)):
        clf_np.fit(X_np[train], y_np[train])
        clf_tr.fit(X_tr[train], y_tr[train])
        
        oos[f_i] = mv.math.spearmanr(clf_np.predict(X_np[test]), clf_tr.predict(X_tr[test]).cpu().numpy())
    
    assert oos.mean() > 0.95

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])