'''
A series of unit tests for mvpy.crossvalidation.RepeatedKFold
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
Test repeated kfold estimators
'''

def test_RepeatedKFold_numpy():
    '''
    Test RepeatedKFold in numpy.
    '''
    
    # setup data
    X = np.random.normal(size = (101, 50))
    indc_e = np.arange(X.shape[0])
    
    # setup repeated k-folds
    n_splits = 10
    n_repeats = 2
    kf = mv.crossvalidation.RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats)
    
    # perform kfold
    for f_i, (train, test) in enumerate(kf.split(X)):
        # check validity
        indc_o = np.concatenate((train, test))
        indc_o = indc_o[np.argsort(indc_o)]
        assert (indc_e == indc_o).all()

def test_RepeatedKFold_torch():
    '''
    Test RepeatedKFold in torch.
    '''
    
    # setup data
    X = torch.randn(101, 50)
    indc_e = torch.arange(X.shape[0])
    
    # setup repeated k-folds
    n_splits = 10
    n_repeats = 2
    kf = mv.crossvalidation.RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats)
    
    # perform kfold
    for f_i, (train, test) in enumerate(kf.split(X)):
        # check validity
        indc_o = torch.cat((train, test))
        indc_o = indc_o[torch.argsort(indc_o)]
        assert (indc_e == indc_o).all()

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])