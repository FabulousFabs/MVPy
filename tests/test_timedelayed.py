'''
A series of unit tests for mvpy.estimators.TimeDelayed
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
import warnings

from typing import Any

# setup tolerance for np.allclose
_ALLCLOSE_RTOL = 1e-3
_ALLCLOSE_ATOL = 1e-3

'''
Setup fixtures
'''

@pytest.fixture
def Xyß():
    '''
    Generate data for testing mTRF estimation.
    '''
    
    # setup dims
    N, C, T, F = 240, 60, 100, 20
    
    # define real TRFs
    t = torch.linspace(0, 1, F)
    ß = torch.stack([torch.stack([torch.sin(2 * torch.pi * t * torch.randint(low = 1, high = 5, size = (1,))),
                                  torch.cos(4 * torch.pi * t * torch.randint(low = 1, high = 5, size = (1,)))], 0) for i in range(C)], 0)
    
    # here, torch may throw a warning in the convolution that we don't want to see
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # define X and y
        X = torch.normal(0, 1, size = (N, 2, T))
        y = torch.stack([torch.stack([torch.nn.functional.conv1d(X[:,None,i,:], ß[None,None,j,i,:], padding = 'same') for i in range(2)], -1).sum(-1) for j in range(C)], 1)
        y = y + torch.normal(0, 1, size = y.shape)
        y = y.squeeze()
    
    return X, y, ß

'''
Test TimeDelayed (TRF)
'''

def test_TimeDelayed_trf_numpy(Xyß):
    '''
    Make sure the TRF estimator works in numpy.
    '''
    
    # unpack data
    X, y, ß = Xyß
    X, y, ß = X.cpu().numpy(), y.cpu().numpy(), ß.cpu().numpy()
    
    # run test
    trf = mv.estimators.TimeDelayed(-10, 10, 1, alphas = np.logspace(-5, 10, 20))
    trf.fit(X, y)
    
    assert mv.math.pearsonr(ß, trf.coef_[...,::-1][...,1:]).mean() > .95

def test_TimeDelayed_trf_torch(Xyß):
    '''
    Make sure the TRF estimator works in torch.
    '''
    
    # unpack data
    X, y, ß = Xyß
    
    # run test
    trf = mv.estimators.TimeDelayed(-10, 10, 1, alphas = torch.logspace(-5, 10, 20))
    trf.fit(X, y)
    
    assert mv.math.pearsonr(ß, trf.coef_.flip(-1)[...,1:]).mean() > .95

def test_TimeDelayed_trf_compare_numpy_torch(Xyß):
    '''
    Make sure numpy and torch agree on the TRF estimator.
    '''
    
    # unpack data
    X_tr, y_tr, ß_tr = Xyß
    X_np, y_np, ß_np = X_tr.cpu().numpy(), y_tr.cpu().numpy(), ß_tr.cpu().numpy()
    
    # run test
    trf_np = mv.estimators.TimeDelayed(-10, 10, 1, alphas = np.logspace(-5, 10, 20)).fit(X_np, y_np)
    trf_tr = mv.estimators.TimeDelayed(-10, 10, 1, alphas = torch.logspace(-5, 10, 20)).fit(X_tr, y_tr)
    
    assert mv.math.pearsonr(trf_np.coef_, trf_tr.coef_.cpu().numpy()).mean() > .95

'''
Test TimeDelayed (SR)
'''

def test_TimeDelayed_sr_numpy(Xyß):
    '''
    Make sure the SR estimator works in numpy.
    '''
    
    # unpack data
    X, y, ß = Xyß
    X, y, ß = X.cpu().numpy(), y.cpu().numpy(), ß.cpu().numpy()
    
    # run test
    sr = mv.estimators.TimeDelayed(-10, 10, 1, alphas = np.logspace(-5, 10, 20)).fit(y, X)
    y_h = sr.predict(y)
    
    assert mv.math.pearsonr(X, y_h).mean() > .80 # here, the threshold is quite low because X is just noise

def test_TimeDelayed_sr_torch(Xyß):
    '''
    Make sure the SR estimator works in torch.
    '''
    
    # unpack data
    X, y, ß = Xyß
    
    # run test
    sr = mv.estimators.TimeDelayed(-10, 10, 1, alphas = torch.logspace(-5, 10, 20)).fit(y, X)
    y_h = sr.predict(y)
    
    assert mv.math.pearsonr(X, y_h).mean() > .80 # here, the threshold is quite low because X is just noise

def test_TimeDelayed_sr_compare_numpy_torch(Xyß):
    '''
    Make sure numpy and torch agree on the SR estimator.
    '''
    
    # unpack data
    X_tr, y_tr, ß_tr = Xyß
    X_np, y_np, ß_np = X_tr.cpu().numpy(), y_tr.cpu().numpy(), ß_tr.cpu().numpy()
    
    # run test
    sr_np = mv.estimators.TimeDelayed(-10, 10, 1, alphas = np.logspace(-5, 10, 20)).fit(y_np, X_np)
    sr_tr = mv.estimators.TimeDelayed(-10, 10, 1, alphas = torch.logspace(-5, 10, 20)).fit(y_tr, X_tr)
    
    assert mv.math.pearsonr(sr_np.predict(y_np), sr_tr.predict(y_tr).cpu().numpy()).mean() > .80 # here, the threshold is quite low because X is just noise

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])