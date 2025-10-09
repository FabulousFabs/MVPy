'''
A series of unit tests for mvpy.estimators.Sliding
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
Test Sliding
'''

def test_Sliding_numpy():
    '''
    Make sure sliding works correctly in numpy backend.
    '''
    
    # create data
    X = np.random.normal(size = (240, 60, 100))
    y = np.random.normal(size = (240, 5, 100))
    
    # run tests
    decoder = mv.estimators.RidgeDecoder(alphas = np.logspace(-5, 10, 20))
    model = mv.estimators.Sliding(estimator = decoder, 
                                  dims = np.array([-1]), 
                                  n_jobs = None, 
                                  verbose = False).fit(X, y)
    patterns = model.collect('pattern_')
    coefs = model.collect('coef_')
    y_h = model.predict(X)
    
    for i in range(100):
        decoder.fit(X[...,i], y[...,i])
        
        assert np.allclose(patterns[i], decoder.pattern_, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
        assert np.allclose(coefs[i], decoder.coef_, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
        assert np.allclose(y_h[...,i], decoder.predict(X[...,i]), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_Sliding_torch():
    '''
    Make sure sliding works correctly in torch backend.
    '''
    
    # create data
    X = torch.normal(0, 1, (240, 60, 100))
    y = torch.normal(0, 1, (240, 5, 100))
    
    # run tests
    decoder = mv.estimators.RidgeDecoder(alphas = torch.logspace(-5, 10, 20))
    model = mv.estimators.Sliding(estimator = decoder, 
                                  dims = (-1,), 
                                  n_jobs = None, 
                                  verbose = False).fit(X, y)
    patterns = model.collect('pattern_').cpu().numpy()
    coefs = model.collect('coef_').cpu().numpy()
    y_h = model.predict(X).cpu().numpy()
    
    for i in range(100):
        decoder.fit(X[...,i], y[...,i])
        
        assert np.allclose(patterns[i], decoder.pattern_.cpu().numpy(), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
        assert np.allclose(coefs[i], decoder.coef_.cpu().numpy(), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
        assert np.allclose(y_h[...,i], decoder.predict(X[...,i]).cpu().numpy(), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_Sliding_compare_numpy_torch():
    '''
    Make sure numpy and torch backend produce the same results.
    '''
    
    # create data
    X_np, y_np = np.random.normal(size = (240, 60, 100)), np.random.normal(size = (240, 5, 100))
    X_tr, y_tr = torch.from_numpy(X_np).to(torch.float64), torch.from_numpy(y_np).to(torch.float64)
    
    # run tests
    mod_np = mv.estimators.Sliding(estimator = mv.estimators.RidgeDecoder(alphas = np.array([1])), dims = np.array([-1]), verbose = False).fit(X_np, y_np)
    mod_tr = mv.estimators.Sliding(estimator = mv.estimators.RidgeDecoder(alphas = torch.tensor([1])), dims = torch.tensor([-1]), verbose = False).fit(X_tr, y_tr)
    
    assert mv.math.spearmanr(mod_np.collect('pattern_'), mod_tr.collect('pattern_').cpu().numpy()).mean() > .90
    assert mv.math.spearmanr(mod_np.collect('coef_'), mod_tr.collect('coef_').cpu().numpy()).mean() > .90
    assert mv.math.spearmanr(mod_np.predict(X_np), mod_tr.predict(X_tr).cpu().numpy()).mean() > .90

def test_Sliding_shape_mismatch():
    '''
    Make sure shape mismatches are handled appropriately.
    '''
    
    # create data
    X_np, y_np = np.random.normal(size = (240, 50, 4, 100)), np.random.normal(size = (240, 10, 5, 100))
    X_tr, y_tr = torch.from_numpy(X_np).to(torch.float64), torch.from_numpy(y_np).to(torch.float64)
    
    # run tests
    with pytest.raises(ValueError):
        mv.estimators.Sliding(estimator = mv.estimators.RidgeDecoder(alphas = np.array([1])), dims = np.array([-1]), verbose = False).fit(X_np, y_np)

    with pytest.raises(ValueError):
        mv.estimators.Sliding(estimator = mv.estimators.RidgeDecoder(alphas = torch.tensor([1])), dims = torch.tensor([-1]), verbose = False).fit(X_tr, y_tr)

def test_Sliding_numpy_function():
    '''
    Make sure callables work appropriately in numpy.
    '''
    
    # create data
    X_np, y_np = np.random.normal(size = (240, 60, 100)), np.random.normal(size = (240, 60, 100))
    
    # run tests
    model = mv.estimators.Sliding(estimator = mv.math.euclidean, dims = np.array([-1]), verbose = False)
    
    assert np.allclose(model.transform(X_np, y_np), mv.math.euclidean(X_np.swapaxes(1, 2), y_np.swapaxes(1, 2)), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_Sliding_torch_function():
    '''
    Make sure callables work appropriately in torch.
    '''

    # create data
    X_tr, y_tr = torch.normal(0, 1, (240, 60, 100)), torch.normal(0, 1, (240, 60, 100))
    
    # run tests
    model = mv.estimators.Sliding(estimator = mv.math.euclidean, dims = torch.tensor([-1]), verbose = False)
    
    assert np.allclose(model.transform(X_tr, y_tr).cpu().numpy(), mv.math.euclidean(X_tr.swapaxes(1, 2), y_tr.swapaxes(1, 2)).cpu().numpy(), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_Sliding_compare_numpy_torch_function():
    '''
    Make sure numpy and torch backends produce same results for sliding functions.
    '''
    
    # create data
    X_np, y_np = np.random.normal(size = (240, 60, 100)), np.random.normal(size = (240, 60, 100))
    X_tr, y_tr = torch.from_numpy(X_np).to(torch.float64), torch.from_numpy(y_np).to(torch.float64)
    
    # run tests
    mod_np = mv.estimators.Sliding(estimator = mv.math.euclidean, dims = np.array([-1]), verbose = False)
    mod_tr = mv.estimators.Sliding(estimator = mv.math.euclidean, dims = torch.tensor([-1]), verbose = False)
    
    assert np.allclose(mod_np.transform(X_np, y_np), mod_tr.transform(X_tr, y_tr).cpu().numpy(), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])