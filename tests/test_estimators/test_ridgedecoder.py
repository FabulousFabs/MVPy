'''
A series of unit tests for mvpy.estimators.RidgeDecoder
'''

import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
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
def Xys():
    '''
    Generate data for testing.
    '''
    
    # setup dims
    N, C, F = 1000, 60, 3
    
    # define latents
    cov_y = np.eye(F) + np.random.rand(F, F)
    cov_y = (cov_y + cov_y.T) / 2.0
    mu_y = np.random.rand(F)
    y = np.random.multivariate_normal(mu_y, cov_y, size = N)
    
    # define a forward model
    F = np.random.uniform(low = -1, high = 1, size = (C, F))
    
    # compute neural patterns
    X = y @ F.T + np.random.normal(loc = 0, scale = 0.05, size = (N, C))

    return X, y, F

'''
Test RidgeDecoder
'''

def test_RidgeDecoder_numpy(Xys):
    '''
    Make sure the ridge decoder works in numpy.
    '''
    
    # unpack data
    X, y, p = Xys
    n_folds = 20
    
    # run tests
    kf = sklearn.model_selection.KFold(n_splits = n_folds)
    decoder = mv.estimators.RidgeDecoder(alphas = np.logspace(-5, 10, 20))
    
    cv_r = np.zeros((n_folds,))
    cv_p = np.zeros((n_folds,))
    
    for f_i, (train, test) in enumerate(kf.split(X, y)):
        # fit decoder
        decoder.fit(X[train], y[train])
        
        # test decoder
        cv_r[f_i] = mv.math.spearmanr(y[test].T, decoder.predict(X[test]).T).mean()
        cv_p[f_i] = mv.math.spearmanr(p.T, decoder.pattern_.T).mean()
    
    assert cv_r.mean() > .90
    assert cv_p.mean() > .80

def test_RidgeDecoder_torch(Xys):
    '''
    Make sure the ridge decoder works in torch.
    '''
    
    # unpack data
    X, y, p = Xys
    n_folds = 20
    
    # transform data
    X, y, p = torch.from_numpy(X).to(torch.float64), torch.from_numpy(y).to(torch.float64), torch.from_numpy(p).to(torch.float64)
    
    # run tests
    kf = sklearn.model_selection.KFold(n_splits = n_folds)
    decoder = mv.estimators.RidgeDecoder(alphas = torch.logspace(-5, 10, 20).to(torch.float64))
    
    cv_r = torch.zeros((n_folds,), dtype = X.dtype)
    cv_p = torch.zeros((n_folds,), dtype = X.dtype)
    
    for f_i, (train, test) in enumerate(kf.split(X, y)):
        # fit decoder
        decoder.fit(X[train], y[train])
        
        # test decoder
        cv_r[f_i] = mv.math.spearmanr(y[test].T, decoder.predict(X[test]).T).mean()
        cv_p[f_i] = mv.math.spearmanr(p.T, decoder.pattern_.T).mean()
    
    assert cv_r.mean() > .90
    assert cv_p.mean() > .80

def test_RidgeDecoder_compare_numpy_torch(Xys):
    '''
    Make sure numpy and torch agree in their ridge decoder results.
    '''
    
    # unpack data
    X_np, y_np, p_np = Xys
    n_folds = 20
    
    # transform data
    X_tr, y_tr, p_tr = torch.from_numpy(X_np).to(torch.float64), torch.from_numpy(y_np).to(torch.float64), torch.from_numpy(p_np).to(torch.float64)
    
    # run tests
    kf = sklearn.model_selection.KFold(n_splits = n_folds)
    dec_np = mv.estimators.RidgeDecoder(alphas = np.logspace(-5, 10, 20))
    dec_tr = mv.estimators.RidgeDecoder(alphas = torch.logspace(-5, 10, 20).to(torch.float64))
    
    for f_i, (train_np, test_np) in enumerate(kf.split(X_np, y_np)):
        # transform data
        train_tr, test_tr = torch.from_numpy(train_np).to(torch.int64), torch.from_numpy(test_np).to(torch.int64)
        
        # fit decoders
        dec_np.fit(X_np[train_np], y_np[train_np])
        dec_tr.fit(X_tr[train_tr], y_tr[train_tr])
        
        assert mv.math.spearmanr(dec_np.estimator.coef_, dec_tr.estimator.coef_.cpu().numpy()).mean() > .95
        assert mv.math.spearmanr(dec_np.pattern_.T, dec_tr.pattern_.cpu().numpy().T).mean() > .80
        assert mv.math.spearmanr(dec_np.predict(X_np[test_np]).T, dec_tr.predict(X_tr[test_tr]).cpu().numpy().T).mean() > .99

def test_RidgeDecoder_shape_mismatch_fit():
    '''
    Make sure ridge decoders throw exceptions for mismatched shapes in fit method.
    '''
    
    # run test
    X_np = np.random.normal(size = (100, 5))
    y_np = np.random.normal(size = (80, 5))
    
    with pytest.raises(ValueError):
        mv.estimators.RidgeDecoder(alphas = np.array([1])).fit(X_np, y_np)
    
    X_tr = torch.normal(0, 1, size = (100, 5))
    y_tr = torch.normal(0, 1, size = (80, 5))
    
    with pytest.raises(ValueError):
        mv.estimators.RidgeDecoder(alphas = torch.tensor([1])).fit(X_tr, y_tr)

def test_RidgeDecoder_shape_mismatch_predict():
    '''
    Make sure decoders throw exceptions for mismatched shapes in predict method.
    '''
    
    # run test
    X_np = np.random.normal(size = (100, 5))
    y_np = np.random.normal(size = (100, 5))
    Z_np = np.random.normal(size = (100, 6))
    dec_np = mv.estimators.RidgeDecoder(alphas = np.array([1])).fit(X_np, y_np)
    
    with pytest.raises(ValueError):
        dec_np.predict(Z_np)
    
    X_tr = torch.normal(0, 1, size = (100, 5))
    y_tr = torch.normal(0, 1, size = (100, 5))
    Z_tr = torch.normal(0, 1, size = (100, 6))
    dec_tr = mv.estimators.RidgeDecoder(alphas = torch.tensor([1])).fit(X_tr, y_tr)
    
    with pytest.raises(ValueError):
        dec_tr.predict(Z_tr)

def test_RidgeDecoder_predict_without_fit():
    '''
    Make sure decoders throw exceptions if not fit.
    '''
    
    # run test
    X_np = np.random.normal(size = (100, 5))
    
    with pytest.raises(ValueError):
        dec_np = mv.estimators.RidgeDecoder(alphas = np.array([1])).predict(X_np)
    
    X_tr = torch.normal(0, 1, size = (100, 5))
    
    with pytest.raises(ValueError):
        dec_tr = mv.estimators.RidgeDecoder(alphas = torch.tensor([1])).predict(X_tr)
    
'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])