'''
A series of unit tests for mvpy.estimators.Covariance
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
def X() -> np.ndarray:
    '''
    Fixture to provide data for tests
    '''
    
    # setup dims
    N, C, T = 200, 60, 50
    
    # setup covariance
    cov = np.fromfunction(lambda i, j: 0.9 ** np.abs(i - j), (C, C))
    
    # draw data
    X = np.empty((N, T, C))
    
    for i in range(N):
        X[i] = np.random.multivariate_normal(np.zeros(C), cov, size = (T,))

    # reshape
    X = X.transpose(0, 2, 1)
    
    return X

@pytest.fixture
def LW(X: np.ndarray) -> Any:
    '''
    Fixture to provide the LedoitWolf solution from sklearn
    '''
    
    from sklearn.covariance import LedoitWolf

    # reshape data
    X_h = X.swapaxes(1, 2).reshape((-1, X.shape[1]))
    
    # estimate covariance
    lw = LedoitWolf().fit(X_h)
    
    return lw

@pytest.fixture
def EM(X: np.ndarray) -> Any:
    '''
    Fixture to provide the Empirical solution
    '''

    from sklearn.covariance import EmpiricalCovariance
    
    # reshape data
    X_h = X.swapaxes(1, 2).reshape((-1, X.shape[1]))
    
    # estimate covariance
    em = EmpiricalCovariance().fit(X_h)
    
    return em

'''
Test Covariance
'''

def test_Covariance_unknown_method():
    '''
    Make sure we fail if we try to use an unknown method
    '''
    
    with pytest.raises(ValueError):
        mv.estimators.Covariance(method = 'idontexist')

'''
Test Empirical
'''

def test_Empirical_numpy(X: np.ndarray, EM: Any):
    '''
    Ensure numpy results for Empirical are correct.
    '''

    # fit estimator
    mv_em = mv.estimators.Covariance(method = 'Empirical').fit(X)

    # assert equality
    assert np.allclose(mv_em.covariance_, EM.covariance_, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_Empirical_torch(X: np.ndarray, EM: Any):
    '''
    Ensure torch results for Empirical are correct.
    '''
    
    # transform to torch
    X = torch.from_numpy(X).to(torch.float32)
    
    # fit estimator
    mv_em = mv.estimators.Covariance(method = 'Empirical').fit(X)
    
    # assert equality
    assert np.allclose(mv_em.covariance_.cpu().numpy(), EM.covariance_, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_Empirical_compare_numpy_torch(X: np.ndarray, EM: Any):
    '''
    Ensure numpy and torch have same solutions for Empirical.
    '''
    
    # transform to torch
    X_tr = torch.from_numpy(X).to(torch.float32)
    
    # fit estimators
    cov_np = mv.estimators.Covariance(method = 'Empirical').fit(X).covariance_
    cov_tr = mv.estimators.Covariance(method = 'Empirical').fit(X_tr).covariance_.cpu().numpy()
    
    # assert equality
    assert np.allclose(cov_np, cov_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_Empirical_shape_mismatch():
    '''
    Ensure we fail if shape cannot produce covariance.
    '''
    
    x_np = np.random.normal(size = (100,))
    x_tr = torch.normal(0, 1, size = (100,))
    
    with pytest.raises(ValueError):
        mv.estimators.Covariance(method = 'Empirical').fit(x_np)
    
    with pytest.raises(ValueError):
        mv.estimators.Covariance(method = 'Empirical').fit(x_tr)

'''
Test LedoitWolf
'''

def test_LedoitWolf_numpy(X: np.ndarray, LW: Any):
    '''
    Ensure numpy results for LedoitWolf are correct.
    '''
    
    # fit estimator
    mv_lw = mv.estimators.Covariance(method = 'LedoitWolf').fit(X)
    
    # assert equality
    assert np.allclose(mv_lw.covariance_, LW.covariance_, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(mv_lw.shrinkage_, LW.shrinkage_, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_LedoitWolf_torch(X: np.ndarray, LW: Any):
    '''
    Ensure torch results for LedoitWolf are correct.
    '''
    
    # transform to torch
    X = torch.from_numpy(X).to(torch.float32)
    
    # fit estimator
    mv_lw = mv.estimators.Covariance(method = 'LedoitWolf').fit(X)
    
    # assert equality
    assert np.allclose(mv_lw.covariance_.cpu().numpy(), LW.covariance_, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(mv_lw.shrinkage_.cpu().numpy(), LW.shrinkage_, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_LedoitWolf_compare_numpy_torch(X: np.ndarray):
    '''
    Make sure numpy and torch agree in their solutions.
    '''
    
    # transform to torch
    X_tr = torch.from_numpy(X).to(torch.float32)
    
    # fit estimators
    cov_np = mv.estimators.Covariance(method = 'LedoitWolf').fit(X)
    cov_tr = mv.estimators.Covariance(method = 'LedoitWolf').fit(X_tr)
    
    # assert equality
    assert np.allclose(cov_np.covariance_, cov_tr.covariance_.cpu().numpy(), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(cov_np.shrinkage_, cov_tr.shrinkage_.cpu().numpy(), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_LedoitWolf_shape_mismatch():
    '''
    Ensure we fail if shape cannot produce covariance.
    '''
    
    x_np = np.random.normal(size = (100,))
    x_tr = torch.normal(0, 1, size = (100,))
    
    with pytest.raises(ValueError):
        mv.estimators.Covariance(method = 'LedoitWolf').fit(x_np)
    
    with pytest.raises(ValueError):
        mv.estimators.Covariance(method = 'LedoitWolf').fit(x_tr)

'''
Test whiteners
'''

def test_Whitener_numpy(X: np.ndarray):
    '''
    Ensure that the whitening operation works as expected.
    '''
    
    # fit estimator
    X_w = mv.estimators.Covariance(method = 'LedoitWolf').fit_transform(X)
    Σ = X_w.std(axis = (0, 2))

    assert np.allclose(Σ, np.ones(X.shape[1]), rtol = 0.1, atol = 0.1)

def test_Whitener_torch(X: np.ndarray):
    '''
    Ensure that the whitening operation works as expected.
    '''
    
    # transform to torch
    X = torch.from_numpy(X).to(torch.float32)
    
    # fit estimator
    X_w = mv.estimators.Covariance(method = 'LedoitWolf').fit_transform(X)
    Σ = X_w.std((0, 2)).cpu().numpy()
    
    assert np.allclose(Σ, np.ones(X.shape[1]), rtol = 0.1, atol = 0.1)

def test_Whitener_type_mismatch(X: np.ndarray):
    '''
    Ensure that the whitener fails if no covariance is provided.
    '''
    
    # convert to torch
    X_tr = torch.from_numpy(X).to(torch.float32)

    with pytest.raises(ValueError):
        cov = mv.estimators.Covariance(method = 'LedoitWolf').fit(X)
        X_w = cov.transform(X_tr)

    with pytest.raises(ValueError):
        cov = mv.estimators.Covariance(method = 'LedoitWolf').fit(X_tr)
        X_w = cov.transform(X)

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])