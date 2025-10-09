'''
A series of unit tests for mvpy.preprocessing.Scaler
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
# here, we're going a bit higher because
# of DDoF corrections
_ALLCLOSE_RTOL = 1e-2
_ALLCLOSE_ATOL = 1e-2

'''
Test scaler estimators
'''

def test_scaler_numpy():
    '''
    Make sure that the numpy scaler works as expected.
    '''
    
    # setup data
    X0 = np.random.normal(loc = 5, scale = 10, size = (100, 60, 50))
    X1 = np.random.normal(loc = 5, scale = 10, size = (60, 50))
    
    # run test
    Z0 = mv.preprocessing.Scaler().fit_transform(X0)
    Z1 = mv.preprocessing.Scaler().fit_transform(X1)
    
    assert np.allclose(Z0.mean(axis = (0, 2)), np.zeros((60,)), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(Z0.std(axis = (0, 2)), np.ones((60,)), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(Z1.mean(axis = 0), np.zeros((50,)), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(Z1.std(axis = 0), np.ones((50,)), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_scaler_torch():
    '''
    Make sure that the torch scaler works as expected.
    '''
    
    # setup data
    X0 = torch.normal(5, 10, (100, 60, 50))
    X1 = torch.normal(5, 10, (60, 50))
    
    # run test
    Z0 = mv.preprocessing.Scaler().fit_transform(X0).cpu().numpy()
    Z1 = mv.preprocessing.Scaler().fit_transform(X1).cpu().numpy()
    
    assert np.allclose(Z0.mean(axis = (0, 2)), np.zeros((60,)), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(Z0.std(axis = (0, 2)), np.ones((60,)), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(Z1.mean(axis = 0), np.zeros((50,)), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(Z1.std(axis = 0), np.ones((50,)), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_scaler_compare_numpy_torch():
    '''
    Make sure that numpy and torch agree.
    '''
    
    # setup data
    X_np = np.random.normal(loc = 5, scale = 10, size = (100, 60, 50))
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    
    # run test
    Z_np = mv.preprocessing.Scaler().fit_transform(X_np)
    Z_tr = mv.preprocessing.Scaler().fit_transform(X_tr).cpu().numpy()
    
    assert np.allclose(Z_np, Z_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])