'''
A series of unit tests for mvpy.preprocessing.Clamp
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

_ALLCLOSE_RTOL = 1e-3
_ALLCLOSE_ATOL = 1e-3

'''
Test clamp estimators
'''

def test_clamp_numpy():
    '''
    Make sure that the numpy clamp works as expected.
    '''
    
    # setup data
    X0 = np.random.normal(loc = 0, scale = 1, size = (100, 60, 50))
    X0[33,5,10] = 1e3
    X1 = np.random.normal(loc = 5, scale = 1, size = (60, 50))
    X1[33,5] = -1e3
    
    # run test
    Z0 = mv.preprocessing.Clamp(upper = 10.0).fit_transform(X0)
    Z1 = mv.preprocessing.Clamp(lower = -10.0).fit_transform(X1)
    
    assert np.isclose(Z0.max(), 10.0).all()
    assert np.isclose(Z1.min(), -10.0).all()

def test_clamp_torch():
    '''
    Make sure that the torch clmap works as expected.
    '''
    
    # setup data
    X0 = torch.normal(5, 10, (100, 60, 50))
    X0[33,5,10] = 1e3
    X1 = torch.normal(5, 10, (60, 50))
    X1[33,5] = -1e3
    
    # run test
    Z0 = mv.preprocessing.Clamp(upper = 10.0).fit_transform(X0).cpu().numpy()
    Z1 = mv.preprocessing.Clamp(lower = -10.0).fit_transform(X1).cpu().numpy()
    
    assert np.isclose(Z0.max(), 10.0).all()
    assert np.isclose(Z1.min(), -10.0).all()

def test_clamp_iqr_compare_numpy_torch():
    '''
    Make sure that numpy and torch agree in 'iqr'.
    '''
    
    # setup data
    X_np = np.random.normal(loc = 5, scale = 10, size = (100, 60, 50))
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    
    # run test
    Z_np = mv.preprocessing.Clamp(method = 'iqr').fit_transform(X_np)
    Z_tr = mv.preprocessing.Clamp(method = 'iqr').fit_transform(X_tr).cpu().numpy()
    
    assert np.allclose(Z_np, Z_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_clamp_quantile_compare_numpy_torch():
    '''
    Make sure that numpy and torch agree in 'quantile'.
    '''
    
    # setup data
    X_np = np.random.normal(loc = 5, scale = 10, size = (100, 60, 50))
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    
    # run test
    Z_np = mv.preprocessing.Clamp(method = 'quantile').fit_transform(X_np)
    Z_tr = mv.preprocessing.Clamp(method = 'quantile').fit_transform(X_tr).cpu().numpy()
    
    assert np.allclose(Z_np, Z_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_clamp_mad_compare_numpy_torch():
    '''
    Make sure that numpy and torch agree in 'mad'.
    '''
    
    # setup data
    X_np = np.random.normal(loc = 5, scale = 10, size = (100, 60, 50))
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    
    # run test
    Z_np = mv.preprocessing.Clamp(method = 'mad').fit_transform(X_np)
    Z_tr = mv.preprocessing.Clamp(method = 'mad').fit_transform(X_tr).cpu().numpy()
    
    assert np.allclose(Z_np, Z_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])