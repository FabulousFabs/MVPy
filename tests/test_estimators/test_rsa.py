'''
A series of unit tests for mvpy.estimators.RSA
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
def Xy_cos() -> tuple[np.ndarray]:
    '''
    Fixture to provide data for tests
    '''
    
    X = np.array([[1.0, 0.0], [-1.0, 0.0],
                  [0.0, 1.0], [0.0, -1.0]])
    y = np.array([[-1], [0], [0], [0], [0], [-1]])
    
    return (X, y)

@pytest.fixture
def Xy_cveuc() -> tuple[np.ndarray]:
    '''
    Fixture to provide data for tests
    '''
    
    X = np.random.normal(size = (50, 3, 60, 5))
    X[:,[0, 2]] += 1.0
    y = np.array([1, 0, 1])
    
    return (X, y)

'''
Test trial-wise estimators
'''

def test_RSA_numpy(Xy_cos: tuple[np.ndarray]):
    '''
    Make sure we can correctly compute RSA using numpy.
    '''
    
    # unpack
    X, y = Xy_cos
    
    # run test
    y_h = mv.estimators.RSA(estimator = mv.math.cosine).transform(X)
    
    assert np.allclose(y, y_h, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_RSA_torch(Xy_cos: tuple[np.ndarray]):
    '''
    Make sure we can correctly compute RSA using torch.
    '''

    # unpack
    X, y = Xy_cos

    # transform to torch
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)

    # run test
    y_h = mv.estimators.RSA(estimator = mv.math.cosine).transform(X)

    assert np.allclose(y, y_h, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_RSA_compare_numpy_torch():
    '''
    Make sure numpy and torch agree.
    '''
    
    # setup data
    X_np = np.random.normal(size = (100, 60, 50))
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    
    # run tests
    r_np = mv.estimators.RSA().transform(X_np)
    r_tr = mv.estimators.RSA().transform(X_tr).cpu().numpy()
    
    assert np.allclose(r_np, r_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_RSA_type_mismatch():
    '''
    Make sure we fail if type is unrecognised.
    '''
    
    # setup data
    X = np.random.normal(size = (100, 60, 50))
    X = list(X)
    
    # run test
    with pytest.raises(ValueError):
        mv.estimators.RSA().transform(X)

def test_RSA_shape_mismatch():
    '''
    Make sure we fail if the shape is unworkable.
    '''
    
    # setup data
    X_np = np.random.normal(size = (100,))
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    
    # run tests
    with pytest.raises(ValueError):
        mv.estimators.RSA().transform(X_np)
        
    with pytest.raises(ValueError):
        mv.estimators.RSA().transform(X_tr)

'''
Run grouped estimator tests
'''

def test_GroupedRSA_numpy(Xy_cveuc: tuple[np.ndarray]):
    '''
    Make sure we can correctly compute grouped RSA using numpy.
    '''
    
    # unpack
    X, y = Xy_cveuc
    
    # run test
    y_h = mv.estimators.RSA(grouped = True, estimator = mv.math.cv_euclidean).transform(X)
    y_h = y_h.mean(axis = 1)
    
    assert mv.math.spearmanr(y, y_h) > 0.5

def test_GroupedRSA_torch(Xy_cveuc: tuple[np.ndarray]):
    '''
    Make sure we can correctly compute grouped RSA using torch.
    '''
    
    # unpack
    X, y = Xy_cveuc
    
    # transform to torch
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    
    # run test
    y_h = mv.estimators.RSA(grouped = True, estimator = mv.math.cv_euclidean).transform(X)
    y_h = y_h.mean(axis = 1)
    
    assert mv.math.spearmanr(y, y_h) > 0.5

def test_GroupedRSA_compare_numpy_torch():
    '''
    Make sure numpy and torch agree.
    '''
    
    # setup data
    X_np = np.random.normal(size = (40, 10, 60, 50))
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    
    # run tests
    r_np = mv.estimators.RSA(grouped = True, estimator = mv.math.cv_euclidean).transform(X_np)
    r_tr = mv.estimators.RSA(grouped = True, estimator = mv.math.cv_euclidean).transform(X_tr).cpu().numpy()
    
    assert np.allclose(r_np, r_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_GroupedRSA_type_mismatch():
    '''
    Make sure we fail if type is unrecognised.
    '''
    
    # setup data
    X = np.random.normal(size = (100, 60, 50))
    X = list(X)
    
    # run test
    with pytest.raises(ValueError):
        mv.estimators.RSA(grouped = True, estimator = mv.math.cv_euclidean).transform(X)

def test_GroupedRSA_shape_mismatch():
    '''
    Make sure we fail if the shape is unworkable.
    '''
    
    # setup data
    X_np = np.random.normal(size = (100,))
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    
    # run tests
    with pytest.raises(ValueError):
        mv.estimators.RSA(grouped = True, estimator = mv.math.cv_euclidean).transform(X_np)
        
    with pytest.raises(ValueError):
        mv.estimators.RSA(grouped = True, estimator = mv.math.cv_euclidean).transform(X_tr)

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])