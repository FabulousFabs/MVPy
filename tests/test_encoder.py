'''
A series of unit tests for mvpy.estimators.Encoder
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
def Xyß_2d():
    '''
    Generate data for testing simple encoder estimation.
    '''
    
    # setup dims
    N, F = 240, 5
    
    # create data
    X = torch.normal(0, 1, (N, F))
    ß = torch.normal(0, 1, (F,))
    y = X @ ß
    y = y[:,None] + torch.normal(0, 1, (N, 1))
    
    return X, y, ß

@pytest.fixture
def Xyß_3d():
    '''
    Generate data for testing simple encoder estimation.
    '''
    
    # setup dims
    N, F, T, C = 240, 5, 100, 60
    
    # create data
    X = torch.normal(0, 1, (N, F, T))
    ß = torch.normal(0, 1, (C, F, T))
    y = torch.stack([torch.stack([X[:,:,i] @ ß[j,:,i] for i in range(X.shape[2])], 0) for j in range(ß.shape[0])], 0).swapaxes(0, 2).swapaxes(1, 2)
    y = y + torch.normal(0, 1, y.shape)
    
    return X, y, ß

'''
Test encoders
'''

def test_Encoder_numpy(Xyß_2d):
    '''
    Test the 2D encoder in numpy.
    '''
    
    # unpack and transform
    X, y, ß = Xyß_2d
    X, y, ß = X.cpu().numpy(), y.cpu().numpy(), ß.cpu().numpy()
    
    # run test
    encoder = mv.estimators.Encoder(alphas = np.logspace(-5, 10, 20)).fit(X, y)
    
    assert mv.math.pearsonr(ß, encoder.coef_.squeeze()) > .9

def test_Encoder_torch(Xyß_2d):
    '''
    Test the 2D encoder in torch.
    '''
    
    # unpack
    X, y, ß = Xyß_2d
    
    # run test
    encoder = mv.estimators.Encoder(alphas = torch.logspace(-5, 10, 20)).fit(X, y)
    
    assert mv.math.pearsonr(ß, encoder.coef_.squeeze()) > .9

def test_Encoder_compare_numpy_torch(Xyß_2d):
    '''
    Make sure numpy and torch agree in the simple encoder estimation.
    '''
    
    # unpack
    X_tr, y_tr, ß_tr = Xyß_2d
    X_np, y_np, ß_np = X_tr.cpu().numpy(), y_tr.cpu().numpy(), ß_tr.cpu().numpy()
    
    # run tests
    encoder_np = mv.estimators.Encoder(alphas = np.logspace(-5, 10, 20)).fit(X_np, y_np)
    encoder_tr = mv.estimators.Encoder(alphas = torch.logspace(-5, 10, 20)).fit(X_tr, y_tr)
    
    assert mv.math.pearsonr(encoder_np.coef_, encoder_tr.coef_.cpu().numpy()).mean() > .9

def test_Encoder_expanded_numpy(Xyß_3d):
    '''
    Test the 3D encoder in numpy.
    '''
    
    # unpack and transform
    X, y, ß = Xyß_3d
    X, y, ß = X.cpu().numpy(), y.cpu().numpy(), ß.cpu().numpy()
    
    # run test
    encoder = mv.estimators.Encoder(alphas = np.logspace(-5, 10, 20)).fit(X, y)
    
    assert mv.math.pearsonr(ß, encoder.coef_.squeeze()).mean() > .9

def test_Encoder_expanded_torch(Xyß_3d):
    '''
    Test the 3D encoder in torch.
    '''
    
    # unpack
    X, y, ß = Xyß_3d
    
    # run test
    encoder = mv.estimators.Encoder(alphas = torch.logspace(-5, 10, 20)).fit(X, y)
    
    assert mv.math.pearsonr(ß, encoder.coef_.squeeze()).mean() > .9

def test_Encoder_expanded_compare_numpy_torch(Xyß_3d):
    '''
    Make sure numpy and torch agree in the simple encoder estimation.
    '''

    # unpack
    X_tr, y_tr, ß_tr = Xyß_3d
    X_np, y_np, ß_np = X_tr.cpu().numpy(), y_tr.cpu().numpy(), ß_tr.cpu().numpy()

    # run tests
    encoder_np = mv.estimators.Encoder(alphas = np.logspace(-5, 10, 20)).fit(X_np, y_np)
    encoder_tr = mv.estimators.Encoder(alphas = torch.logspace(-5, 10, 20)).fit(X_tr, y_tr)
    
    assert mv.math.pearsonr(encoder_np.coef_, encoder_tr.coef_.cpu().numpy()).mean() > .9

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])