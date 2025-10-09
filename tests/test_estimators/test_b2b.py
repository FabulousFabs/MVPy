'''
A series of unit tests for mvpy.estimators.B2B
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
    X = y @ F.T + np.random.normal(loc = 0, scale = 1.0, size = (N, C))
    
    # add a correlated but non-causal predictor
    y = np.concatenate((y, y.mean(axis = 1, keepdims = True) + np.random.normal(loc = 0, scale = 1.0, size = (N, 1))), axis = 1)
    
    return X, y, F

'''
Test B2B
'''

def test_B2B_numpy(Xys):
    '''
    Make sure the B2B estimator works in numpy.
    '''
    
    # unpack data
    X, y, p = Xys
    n_permutations = 50
    
    # run tests
    b2b = mv.estimators.B2B(alphas = np.logspace(-5, 10, 20))
    
    cv_r = np.zeros((n_permutations, y.shape[-1]))
    
    for p_i in range(n_permutations):
        # permute data
        indc = np.random.choice(np.arange(X.shape[0]), replace = False, size = (X.shape[0],))
        X_p, y_p = X[indc], y[indc]
        
        # fit model
        b2b.fit(X_p, y_p)
        
        # obtain results
        cv_r[p_i] = b2b.causal_.copy()
    
    out = np.ones((y.shape[-1],))
    out[-1] = 0.0
    
    assert mv.math.pearsonr(out, cv_r.mean(axis = 0)) > .80

def test_B2B_torch(Xys):
    '''
    Make sure the B2B estimator works in torch.
    '''

    # unpack data
    X, y, p = Xys
    n_permutations = 50
    
    # convert data
    X, y = torch.from_numpy(X).to(torch.float32), torch.from_numpy(y).to(torch.float32)
    
    # run tests
    b2b = mv.estimators.B2B(alphas = torch.logspace(-5, 10, 20))
    
    cv_r = torch.zeros((n_permutations, y.shape[-1]), dtype = X.dtype)
    
    for p_i in range(n_permutations):
        # permute data
        indc = torch.randperm(X.shape[0])
        X_p, y_p = X[indc], y[indc]

        # fit model
        b2b.fit(X_p, y_p)
        
        # obtain results
        cv_r[p_i] = b2b.causal_.clone()
    
    out = torch.ones((y.shape[-1],), dtype = X.dtype)
    out[-1] = 0.0
    
    assert mv.math.pearsonr(out, cv_r.mean(axis = 0)) > .80

def test_B2B_compare_numpy_torch(Xys):
    '''
    Make sure the B2B estimator works in numpy and torch.
    '''
    
    # unpack data
    X_np, y_np, p = Xys
    n_permutations = 50
    
    # convert data
    X_tr, y_tr = torch.from_numpy(X_np).to(torch.float32), torch.from_numpy(y_np).to(torch.float32)
    
    # run tests
    b2b_np = mv.estimators.B2B(alphas = np.array([1]))
    b2b_tr = mv.estimators.B2B(alphas = torch.tensor([1]))
    
    cv_r_np = np.zeros((n_permutations, y_np.shape[-1]))
    cv_r_tr = torch.zeros((n_permutations, y_tr.shape[-1]))
    
    for p_i in range(n_permutations):
        # permute data
        indc = np.random.choice(np.arange(X_np.shape[0]), replace = False, size = (X_np.shape[0],))
        X_np, y_np = X_np[indc], y_np[indc]
        X_tr, y_tr = X_tr[torch.from_numpy(indc).to(torch.int32)], y_tr[torch.from_numpy(indc).to(torch.int32)]

        # fit model
        b2b_np.fit(X_np, y_np)
        b2b_tr.fit(X_tr, y_tr)
        
        # obtain results
        cv_r_np[p_i] = b2b_np.causal_.copy()
        cv_r_tr[p_i] = b2b_tr.causal_.clone()
    
    assert mv.math.pearsonr(cv_r_np, cv_r_tr.cpu().numpy()).mean() > .90

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])