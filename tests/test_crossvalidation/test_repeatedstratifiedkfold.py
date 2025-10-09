'''
A series of unit tests for mvpy.crossvalidation.RepeatedStratifiedKFold
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

'''
Test repeated stratified kfold estimators
'''

def test_RepeatedStratifiedKFold_numpy():
    '''
    Test RepeatedStratifiedKFold in numpy.
    '''
    
    # generate data
    X = np.random.random(size = (125, 50))
    y = np.array([0] * 25 + [1] * 90 + [2] * 10)
    
    # setup kfold
    kf = mv.crossvalidation.RepeatedStratifiedKFold(n_splits = 10, n_repeats = 2)
    
    # setup counts
    n_counts_train = np.zeros((20, 3))
    n_counts_test = np.zeros((20, 3))
    
    # run folds
    indc_e = np.arange(X.shape[0])
    for f_i, (train, test) in enumerate(kf.split(X, y)):
        # count
        _, n_counts_train[f_i] = np.unique(y[train], return_counts = True)
        _, n_counts_test[f_i] = np.unique(y[test], return_counts = True)
        
        # check validity
        indc_o = np.concatenate((train, test))
        indc_o = indc_o[np.argsort(indc_o)]
        assert (indc_e == indc_o).all()
    
    # check approximate count validity
    for i in range(3):
        assert ((n_counts_train[0,i] - n_counts_train[:,i]) < 2).all()
        assert ((n_counts_test[0,i] - n_counts_test[:,i]) < 2).all()

def test_RepeatedStratifiedKFold_torch():
    '''
    Test RepeatedStratifiedKFold in torch.
    '''
    
    # generate data
    X = np.random.random(size = (125, 50))
    y = np.array([0] * 25 + [1] * 90 + [2] * 10)
    
    # move data
    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).long()
    
    # setup kfold
    kf = mv.crossvalidation.RepeatedStratifiedKFold(n_splits = 10, n_repeats = 2)
    
    # setup counts
    n_counts_train = torch.zeros((20, 3))
    n_counts_test = torch.zeros((20, 3))
    
    # run folds
    indc_e = torch.arange(X.shape[0])
    for f_i, (train, test) in enumerate(kf.split(X, y)):
        # count
        _, n_counts_train[f_i] = torch.unique(y[train], return_counts = True)
        _, n_counts_test[f_i] = torch.unique(y[test], return_counts = True)
        
        # check validity
        indc_o = torch.cat((train, test))
        indc_o = indc_o[torch.argsort(indc_o)]
        assert (indc_e == indc_o).all()
    
    # check approximate count validity
    for i in range(3):
        assert ((n_counts_train[0,i] - n_counts_train[:,i]) < 2).all()
        assert ((n_counts_test[0,i] - n_counts_test[:,i]) < 2).all()

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])