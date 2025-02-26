import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mvpy as mv

import numpy as np
import scipy

def test_cosine_numpy():
    '''
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # test 1D
    x, y = np.random.normal(size = (sx,)), np.random.normal(size = (sx,))
    r_mv = mv.math.cosine_d(x, y)
    r_sp = scipy.spatial.distance.cosine(x, y)
    
    assert np.allclose(r_mv, r_sp)
    
    # test 2D
    x, y = np.random.normal(size = (sx, sy)), np.random.normal(size = (sx, sy))
    r_mv = mv.math.cosine_d(x, y)
    r_sp = np.array([scipy.spatial.distance.cosine(x[i,:], y[i,:]) for i in range(sx)])
    
    assert np.allclose(r_mv, r_sp)
    
    # test 3D
    x, y = np.random.normal(size = (sx, sy, sz)), np.random.normal(size = (sx, sy, sz))
    r_mv = mv.math.cosine_d(x, y)
    r_sp = np.array([[scipy.spatial.distance.cosine(x[i,j,:], y[i,j,:]) for j in range(sy)] for i in range(sx)])
    
    assert np.allclose(r_mv, r_sp)

def test_cosine_shape_mismatch():
    '''
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # run test
    x, y = np.random.normal(size = (sx, sy)), np.random.normal(size = (sz, sy))
    
    with pytest.raises(ValueError):
        mv.math.cosine_d(x, y)