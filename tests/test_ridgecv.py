'''
A series of unit tests for mvpy.estimators.RidgeCV
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
_ALLCLOSE_RTOL = 1e-2
_ALLCLOSE_ATOL = 1e-2

'''
Test RidgeCV estimators
'''

def test_RidgeCV_compare_numpy_torch():
    '''
    Make sure that RidgeCV estimators are consistent between numpy and torch.
    '''
    
    # create some data
    ß = np.random.normal(0, 1, size = (5,))
    
    X_np = np.random.normal(0, 1, size = (240, 5))
    y_np = ß @ X_np.T + np.random.normal(0, 1.0, size = (X_np.shape[0],))
    
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    y_tr = torch.from_numpy(y_np).to(torch.float32)
    
    a_np = np.logspace(-5, 10, 20)
    a_tr = torch.from_numpy(a_np).to(torch.float32)
    
    # run tests
    ridge_np = mv.estimators.RidgeCV(alphas = a_np).fit(X_np, y_np)
    ridge_tr = mv.estimators.RidgeCV(alphas = a_tr).fit(X_tr, y_tr)
    
    assert np.allclose(ridge_np.intercept_, ridge_tr.intercept_.cpu().numpy(), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(ridge_np.coef_, ridge_tr.coef_.cpu().numpy(), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_RidgeCV_compare_numpy_torch_multi_target():
    '''
    Make sure that RidgeCV estimators are consistent between numpy and torch.
    '''
    
    # create some data
    ß = np.random.normal(0, 1, size = (5, 10))
    
    X_np = np.random.normal(0, 1, size = (240, 5))
    y_np = np.array([ß[:,i] @ X_np.T for i in range(ß.shape[1])]).T + np.random.normal(0, 1.0, size = (X_np.shape[0], ß.shape[1]))
    
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    y_tr = torch.from_numpy(y_np).to(torch.float32)
    
    a_np = np.logspace(-5, 10, 20)
    a_tr = torch.from_numpy(a_np).to(torch.float32)
    
    # run tests
    ridge_np = mv.estimators.RidgeCV(alphas = a_np, alpha_per_target = True).fit(X_np, y_np)
    ridge_tr = mv.estimators.RidgeCV(alphas = a_tr, alpha_per_target = True).fit(X_tr, y_tr)
    
    assert np.allclose(ridge_np.intercept_, ridge_tr.intercept_.cpu().numpy(), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(ridge_np.coef_, ridge_tr.coef_.cpu().numpy(), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_RidgeCV_shape_mismatch():
    '''
    Make sure RidgeCV fails when shapes mismatch.
    '''
    
    X = torch.normal(0, 1, (100, 5))
    y = torch.normal(0, 1, (80, 5))
    
    with pytest.raises(ValueError):
        mv.estimators.RidgeCV().fit(X, y)

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])