'''
A series of unit tests for mvpy.preprocessing.LabelBinariser
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

from typing import List, Any

'''
Setup fixtures
'''

def make_data(N: int, F: List[List[Any]]):
    '''
    Generate the labels
    '''
    
    return np.array(
        [np.random.choice(F[i], replace = True, size = (N,)) for i in range(len(F))]
    ).T

@pytest.fixture
def y_F1_C2_int():
    '''
    Generate data for testing
    '''
    
    # setup dims
    N, F = 240, [[0, 1]]
    
    return make_data(N, F)

@pytest.fixture
def y_F1_C5_int():
    '''
    Generate data for testing
    '''
    
    # setup dims
    N, F = 240, [[0, 1, 2, 3, 4]]
    
    return make_data(N, F)

@pytest.fixture
def y_F5_C2_int():
    '''
    Generate data for testing
    '''
    
    # setup dims
    N, F = 240, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    
    return make_data(N, F)

@pytest.fixture
def y_F5_C5_int():
    '''
    Generate data for testing
    '''
    
    # setup dims
    N, F = 240, [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    
    return make_data(N, F)

@pytest.fixture
def y_F5_CX_int():
    '''
    Generate data for testing
    '''
    
    # setup dims
    N, F = 240, [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]
    
    return make_data(N, F)

@pytest.fixture
def y_F5_CX_str():
    '''
    Generate data for testing
    '''
    
    # setup dims
    N, F = 240, [['a', 'b'], ['c', 'd', 'e'], ['f', 'g', 'h', 'i'], ['j', 'k', 'l', 'm', 'n'], ['o', 'p', 'q', 'r', 's', 't']]
    
    return make_data(N, F)

'''
Test LabelBinariser estimators (with defaults)
'''

def test_LabelBinariser_identity_numpy():
    '''
    Make sure numpy binariser works for identity.
    '''
    
    # make data
    N = 10
    y = np.arange(10)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_numpy()
    L = label.fit_transform(y)
    
    assert (L == np.eye(N)).all()

def test_LabelBinariser_identity_torch():
    '''
    Make sure torch binariser works for identity.
    '''
    
    # make data
    N = 10
    y = torch.arange(10)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_torch()
    L = label.fit_transform(y)
    
    assert (L == torch.eye(N)).all()

def test_LabelBinariser_F1_C2_int_numpy(y_F1_C2_int):
    '''
    Make sure numpy binariser works for single-feature binary problem.
    '''
    
    # rename
    y = y_F1_C2_int
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F1_C2_int_torch(y_F1_C2_int):
    '''
    Make sure torch binariser works for single-feature binary problem.
    '''
    
    # rename and convert
    y = torch.from_numpy(y_F1_C2_int)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_torch()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F1_C5_int_numpy(y_F1_C5_int):
    '''
    Make sure numpy binariser works for single-feature multiclass problem.
    '''
    
    # rename
    y = y_F1_C5_int
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F1_C5_int_torch(y_F1_C5_int):
    '''
    Make sure torch binariser works for single-feature multiclass problem.
    '''
    
    # rename and convert
    y = torch.from_numpy(y_F1_C5_int)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_torch()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_C2_int_numpy(y_F5_C2_int):
    '''
    Make sure numpy binariser works for multi-feature binary problem.
    '''
    
    # rename
    y = y_F5_C2_int
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_C2_int_torch(y_F5_C2_int):
    '''
    Make sure torch binariser works for multi-feature binary problem.
    '''
    
    # rename and convert
    y = torch.from_numpy(y_F5_C2_int)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_torch()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_C5_int_numpy(y_F5_C5_int):
    '''
    Make sure numpy binariser works for multi-feature multi-class problem.
    '''
    
    # rename
    y = y_F5_C5_int
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_C5_int_torch(y_F5_C5_int):
    '''
    Make sure torch binariser works for multi-feature multi-class problem.
    '''
    
    # rename and convert
    y = torch.from_numpy(y_F5_C5_int)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_torch()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_CX_int_numpy(y_F5_CX_int):
    '''
    Make sure numpy binariser works for multi-feature multi-class problem with varying number of classes.
    '''
    
    # rename
    y = y_F5_CX_int
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_CX_int_torch(y_F5_CX_int):
    '''
    Make sure torch binariser works for multi-feature multi-class problem with varying number of classes.
    '''
    
    # rename and convert
    y = torch.from_numpy(y_F5_CX_int)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_torch()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_CX_int_compare_numpy_torch(y_F5_CX_int):
    '''
    Make sure binariser are equal in numpy/torch for multi-feature multi-class problem with varying number of classes.
    '''
    
    # rename and convert
    y_np = y_F5_CX_int.copy()
    y_tr = torch.from_numpy(y_F5_CX_int)
    
    # setup binarisers
    label_np = mv.preprocessing.LabelBinariser().to_numpy()
    label_tr = mv.preprocessing.LabelBinariser().to_torch()
    
    L_np = label_np.fit_transform(y_np)
    H_np = label_np.inverse_transform(L_np)
    L_tr = label_tr.fit_transform(y_tr)
    H_tr = label_tr.inverse_transform(L_tr)
    
    assert (L_np == L_tr.numpy()).all()
    assert (H_np == H_tr.numpy()).all()

def test_LabelBinariser_F5_CX_str_numpy(y_F5_CX_str):
    '''
    Make sure numpy binariser works for multi-feature multi-class problem with varying number of classes that are strings.
    '''
    
    # rename
    y = y_F5_CX_str
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser().to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

'''
Test LabelBinariser estimators (with custom values)
'''

def test_LabelBinariser_identity_numpy_custom():
    '''
    Make sure numpy binariser works for identity.
    '''
    
    # make data
    N = 10
    y = np.arange(10)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_numpy()
    L = label.fit_transform(y)
    
    # setup eye
    eye = np.eye(N)
    eye = 2 * ((2 * eye) - 1)
    
    assert (L == eye).all()

def test_LabelBinariser_identity_torch_custom():
    '''
    Make sure torch binariser works for identity.
    '''
    
    # make data
    N = 10
    y = torch.arange(10)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_torch()
    L = label.fit_transform(y)
    
    # setup eye
    eye = torch.eye(N)
    eye = 2 * ((2 * eye) - 1)
    
    assert (L == eye).all()

def test_LabelBinariser_F1_C2_int_numpy_custom(y_F1_C2_int):
    '''
    Make sure numpy binariser works for single-feature binary problem.
    '''
    
    # rename
    y = y_F1_C2_int
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F1_C2_int_torch_custom(y_F1_C2_int):
    '''
    Make sure torch binariser works for single-feature binary problem.
    '''
    
    # rename and convert
    y = torch.from_numpy(y_F1_C2_int)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_torch()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F1_C5_int_numpy_custom(y_F1_C5_int):
    '''
    Make sure numpy binariser works for single-feature multiclass problem.
    '''
    
    # rename
    y = y_F1_C5_int
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F1_C5_int_torch_custom(y_F1_C5_int):
    '''
    Make sure torch binariser works for single-feature multiclass problem.
    '''
    
    # rename and convert
    y = torch.from_numpy(y_F1_C5_int)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_torch()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_C2_int_numpy_custom(y_F5_C2_int):
    '''
    Make sure numpy binariser works for multi-feature binary problem.
    '''
    
    # rename
    y = y_F5_C2_int
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_C2_int_torch_custom(y_F5_C2_int):
    '''
    Make sure torch binariser works for multi-feature binary problem.
    '''
    
    # rename and convert
    y = torch.from_numpy(y_F5_C2_int)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_torch()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_C5_int_numpy_custom(y_F5_C5_int):
    '''
    Make sure numpy binariser works for multi-feature multi-class problem.
    '''
    
    # rename
    y = y_F5_C5_int
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_C5_int_torch_custom(y_F5_C5_int):
    '''
    Make sure torch binariser works for multi-feature multi-class problem.
    '''
    
    # rename and convert
    y = torch.from_numpy(y_F5_C5_int)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_torch()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_CX_int_numpy_custom(y_F5_CX_int):
    '''
    Make sure numpy binariser works for multi-feature multi-class problem with varying number of classes.
    '''
    
    # rename
    y = y_F5_CX_int
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_CX_int_torch_custom(y_F5_CX_int):
    '''
    Make sure torch binariser works for multi-feature multi-class problem with varying number of classes.
    '''
    
    # rename and convert
    y = torch.from_numpy(y_F5_CX_int)
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_torch()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

def test_LabelBinariser_F5_CX_int_compare_numpy_torch_custom(y_F5_CX_int):
    '''
    Make sure binariser are equal in numpy/torch for multi-feature multi-class problem with varying number of classes.
    '''
    
    # rename and convert
    y_np = y_F5_CX_int.copy()
    y_tr = torch.from_numpy(y_F5_CX_int)
    
    # setup binarisers
    label_np = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_numpy()
    label_tr = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_torch()
    
    L_np = label_np.fit_transform(y_np)
    H_np = label_np.inverse_transform(L_np)
    L_tr = label_tr.fit_transform(y_tr)
    H_tr = label_tr.inverse_transform(L_tr)
    
    assert (L_np == L_tr.numpy()).all()
    assert (H_np == H_tr.numpy()).all()

def test_LabelBinariser_F5_CX_str_numpy_custom(y_F5_CX_str):
    '''
    Make sure numpy binariser works for multi-feature multi-class problem with varying number of classes that are strings.
    '''
    
    # rename
    y = y_F5_CX_str
    
    # setup binariser
    label = mv.preprocessing.LabelBinariser(
        neg_label = -2, 
        pos_label = 2
    ).to_numpy()
    L = label.fit_transform(y)
    H = label.inverse_transform(L)
    
    assert (y == H).all()

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])