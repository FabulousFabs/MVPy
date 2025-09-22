'''
A series of unit tests for mvpy.crossvalidation.KFold
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

'''
Setup fixtures
'''

@pytest.fixture
def XEE():
    '''
    '''
    
    # create data
    X = np.random.normal(size = (10, 50))
    
    # setup expectation
    E_train = np.array([[2, 3, 4, 5, 6, 7, 8, 9],
                        [0, 1, 4, 5, 6, 7, 8, 9],
                        [0, 1, 2, 3, 6, 7, 8, 9],
                        [0, 1, 2, 3, 4, 5, 8, 9],
                        [0, 1, 2, 3, 4, 5, 6, 7]])
    
    E_test = np.array([[0, 1],
                       [2, 3],
                       [4, 5],
                       [6, 7],
                       [8, 9]])
    
    return X, E_train, E_test

'''
Test kfold estimators
'''

def test_KFold_numpy(XEE):
    '''
    Test KFold in numpy.
    '''
    
    # unpack
    X, E_train, E_test = XEE
    
    # setup kfold
    kf = mv.crossvalidation.KFold(n_splits = 5)
    
    # test
    for f_i, (train, test) in enumerate(kf.split(X)):
        assert (E_train[f_i] == train).all()
        assert (E_test[f_i] == test).all()

def test_KFold_torch(XEE):
    '''
    Test KFold in torch.
    '''
    
    # unpack
    X, E_train, E_test = XEE
    
    # transform
    X = torch.from_numpy(X).float()
    E_train = torch.from_numpy(E_train).long()
    E_test = torch.from_numpy(E_test).long()
    
    # setup kfold
    kf = mv.crossvalidation.KFold(n_splits = 5)
    
    # test
    for f_i, (train, test) in enumerate(kf.split(X)):
        assert (E_train[f_i] == train).all()
        assert (E_test[f_i] == test).all()
    
'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])