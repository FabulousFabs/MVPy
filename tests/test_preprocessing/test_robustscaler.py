'''
A series of unit tests for mvpy.preprocessing.RobustScaler
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
Test robust scaler estimators
'''

def test_robustscaler_numpy():
    '''
    Make sure that the numpy robust scaler works as expected.
    '''
    
    # setup data
    X = np.random.normal(size = (120, 50))
    
    # run test
    Z0 = mv.preprocessing.RobustScaler().fit_transform(X)
    Z1 = sklearn.preprocessing.RobustScaler().fit_transform(X)
    
    assert np.allclose(Z0.mean(axis = 0), Z1.mean(axis = 0), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(Z0.std(axis = 0), Z1.std(axis = 0), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_robustscaler_torch():
    '''
    Make sure that the torch robust scaler works as expected.
    '''
    
    # setup data
    X = torch.normal(5, 10, (120, 50))
    
    # run test
    Z0 = mv.preprocessing.RobustScaler().fit_transform(X).cpu().numpy()
    Z1 = sklearn.preprocessing.RobustScaler().fit_transform(X.cpu().numpy())
    
    assert np.allclose(Z0.mean(axis = 0), Z1.mean(axis = 0), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(Z0.std(axis = 0), Z1.std(axis = 0), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_robustscaler_compare_numpy_torch_3d():
    '''
    Make sure that numpy and torch agree.
    '''
    
    # setup data
    X_np = np.random.normal(loc = 5, scale = 10, size = (100, 60, 50))
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    
    # run test
    Z_np = mv.preprocessing.RobustScaler().fit_transform(X_np)
    Z_tr = mv.preprocessing.RobustScaler().fit_transform(X_tr).cpu().numpy()
    
    assert np.allclose(Z_np, Z_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_robustscaler_compare_numpy_torch_4d():
    '''
    Make sure that numpy and torch agree.
    '''
    
    # setup data
    X_np = np.random.normal(loc = 5, scale = 10, size = (100, 60, 50, 15))
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    
    # run test
    Z_np = mv.preprocessing.RobustScaler(dims = (1, 2)).fit_transform(X_np)
    Z_tr = mv.preprocessing.RobustScaler(dims = (1, 2)).fit_transform(X_tr).cpu().numpy()
    
    assert np.allclose(Z_np, Z_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_robustscaler_shapes_numpy():
    '''
    Make sure numpy backend correctly handles shapes.
    '''
    
    X2 = np.random.normal(size = (100, 50))
    X3 = np.random.normal(size = (100, 50, 25))
    X4 = np.random.normal(size = (100, 50, 25, 12))
    
    # test shapes for 2D
    assert mv.preprocessing.RobustScaler(dims = (0,)).fit(X2).centre_.shape == (1, 50)
    assert mv.preprocessing.RobustScaler(dims = (1,)).fit(X2).centre_.shape == (100, 1)
    assert mv.preprocessing.RobustScaler(dims = (0, 1)).fit(X2).centre_.shape == (1, 1)
    
    # test some shapes for 3D
    assert mv.preprocessing.RobustScaler(dims = (0,)).fit(X3).centre_.shape == (1, 50, 25)
    assert mv.preprocessing.RobustScaler(dims = (1,)).fit(X3).centre_.shape == (100, 1, 25)
    assert mv.preprocessing.RobustScaler(dims = (2,)).fit(X3).centre_.shape == (100, 50, 1)
    assert mv.preprocessing.RobustScaler(dims = (0, 1)).fit(X3).centre_.shape == (1, 1, 25)
    assert mv.preprocessing.RobustScaler(dims = (1, 2)).fit(X3).centre_.shape == (100, 1, 1)
    assert mv.preprocessing.RobustScaler(dims = (0, 2)).fit(X3).centre_.shape == (1, 50, 1)
    
    # test some shapes for 4D
    assert mv.preprocessing.RobustScaler(dims = (0,)).fit(X4).centre_.shape == (1, 50, 25, 12)
    assert mv.preprocessing.RobustScaler(dims = (1,)).fit(X4).centre_.shape == (100, 1, 25, 12)
    assert mv.preprocessing.RobustScaler(dims = (2,)).fit(X4).centre_.shape == (100, 50, 1, 12)
    assert mv.preprocessing.RobustScaler(dims = (0, 1)).fit(X4).centre_.shape == (1, 1, 25, 12)
    assert mv.preprocessing.RobustScaler(dims = (1, 2)).fit(X4).centre_.shape == (100, 1, 1, 12)
    assert mv.preprocessing.RobustScaler(dims = (0, 2)).fit(X4).centre_.shape == (1, 50, 1, 12)
    assert mv.preprocessing.RobustScaler(dims = (0, 1, 2)).fit(X4).centre_.shape == (1, 1, 1, 12)
    assert mv.preprocessing.RobustScaler(dims = (1, 2, 3)).fit(X4).centre_.shape == (100, 1, 1, 1)
    assert mv.preprocessing.RobustScaler(dims = (0, 2, 3)).fit(X4).centre_.shape == (1, 50, 1, 1)

def test_robustscaler_shapes_torch():
    '''
    Make sure numpy backend correctly handles shapes.
    '''
    
    X2 = torch.normal(0, 1, size = (100, 50))
    X3 = torch.normal(0, 1, size = (100, 50, 25))
    X4 = torch.normal(0, 1, size = (100, 50, 25, 12))
    
    # test shapes for 2D
    assert mv.preprocessing.RobustScaler(dims = (0,)).fit(X2).centre_.shape == (1, 50)
    assert mv.preprocessing.RobustScaler(dims = (1,)).fit(X2).centre_.shape == (100, 1)
    assert mv.preprocessing.RobustScaler(dims = (0, 1)).fit(X2).centre_.shape == (1, 1)
    
    # test some shapes for 3D
    assert mv.preprocessing.RobustScaler(dims = (0,)).fit(X3).centre_.shape == (1, 50, 25)
    assert mv.preprocessing.RobustScaler(dims = (1,)).fit(X3).centre_.shape == (100, 1, 25)
    assert mv.preprocessing.RobustScaler(dims = (2,)).fit(X3).centre_.shape == (100, 50, 1)
    assert mv.preprocessing.RobustScaler(dims = (0, 1)).fit(X3).centre_.shape == (1, 1, 25)
    assert mv.preprocessing.RobustScaler(dims = (1, 2)).fit(X3).centre_.shape == (100, 1, 1)
    assert mv.preprocessing.RobustScaler(dims = (0, 2)).fit(X3).centre_.shape == (1, 50, 1)
    
    # test some shapes for 4D
    assert mv.preprocessing.RobustScaler(dims = (0,)).fit(X4).centre_.shape == (1, 50, 25, 12)
    assert mv.preprocessing.RobustScaler(dims = (1,)).fit(X4).centre_.shape == (100, 1, 25, 12)
    assert mv.preprocessing.RobustScaler(dims = (2,)).fit(X4).centre_.shape == (100, 50, 1, 12)
    assert mv.preprocessing.RobustScaler(dims = (0, 1)).fit(X4).centre_.shape == (1, 1, 25, 12)
    assert mv.preprocessing.RobustScaler(dims = (1, 2)).fit(X4).centre_.shape == (100, 1, 1, 12)
    assert mv.preprocessing.RobustScaler(dims = (0, 2)).fit(X4).centre_.shape == (1, 50, 1, 12)
    assert mv.preprocessing.RobustScaler(dims = (0, 1, 2)).fit(X4).centre_.shape == (1, 1, 1, 12)
    assert mv.preprocessing.RobustScaler(dims = (1, 2, 3)).fit(X4).centre_.shape == (100, 1, 1, 1)
    assert mv.preprocessing.RobustScaler(dims = (0, 2, 3)).fit(X4).centre_.shape == (1, 50, 1, 1)

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])