import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mvpy as mv

import numpy as np
import scipy

def test_pearsonr_1d():
    '''
    '''
    
    x = np.random.normal(size = (100,))
    y = np.random.normal(size = (100,))
    
    r = mv.math.pearsonr(x, y)
    s = scipy.stats.pearsonr(x, y).statistic
    
    assert(np.isclose(r, s).all())

def test_pearsonr_2d():
    '''
    '''
    
    x = np.random.normal(size = (100, 10))
    y = np.random.normal(size = (100, 10))
    
    r = mv.math.pearsonr(x, y)
    s = np.array([scipy.stats.pearsonr(x[i,:], y[i,:]).statistic for i in range(x.shape[0])])
        
    assert(np.isclose(r, s).all())

def test_pearsonr_3d():
    '''
    '''
    
    x = np.random.normal(size = (100, 10, 5))
    y = np.random.normal(size = (100, 10, 5))
    
    r = mv.math.pearsonr(x, y)
    s = np.array([[scipy.stats.pearsonr(x[i,j,:], y[i,j,:]).statistic for j in range(x.shape[1])] for i in range(x.shape[0])])
    
    assert(np.isclose(r, s).all())

def test_pearson_higher_dim():
    '''
    '''
    
    x = np.random.normal(size = (100, 10, 5, 2))
    y = np.random.normal(size = (100, 10, 5, 2))
    
    with pytest.raises(NotImplementedError):
        mv.math.pearsonr(x, y)