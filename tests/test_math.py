'''
A series of unit tests for mvpy.math.*
'''

import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mvpy as mv

import numpy as np
import torch
import scipy

from typing import Callable

# setup tolerance for np.allclose
_ALLCLOSE_RTOL = 1e-3
_ALLCLOSE_ATOL = 1e-3

def _run_test_numpy(f_mv: Callable, f_sp: Callable) -> None:
    '''
    Compute a batch of tests for 1D, 2D and 3D arrays
    using the numpy backend and compare results to
    those from scipy.
    
    INPUTS:
        f_mv    -   Function from our package to test
        f_sp    -   Corresponding function from scipy. Note that, if scipy returns more than just the metric, please use a lambda to wrap the function and return only the metric.
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # test 1D
    x, y = np.random.normal(size = (sx,)), np.random.normal(size = (sx,))
    r_mv = f_mv(x, y)
    r_sp = f_sp(x, y)
    
    assert np.allclose(r_mv, r_sp, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    
    # test 2D
    x, y = np.random.normal(size = (sx, sy)), np.random.normal(size = (sx, sy))
    r_mv = f_mv(x, y)
    r_sp = np.array([f_sp(x[i,:], y[i,:]) for i in range(sx)])
    
    assert np.allclose(r_mv, r_sp, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    
    # test 3D
    x, y = np.random.normal(size = (sx, sy, sz)), np.random.normal(size = (sx, sy, sz))
    r_mv = f_mv(x, y)
    r_sp = np.array([[f_sp(x[i,j,:], y[i,j,:]) for j in range(sy)] for i in range(sx)])
    
    assert np.allclose(r_mv, r_sp, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def _run_test_torch(f_mv: Callable, f_sp: Callable) -> None:
    '''
    Compute a batch of tests for 1D, 2D and 3D arrays
    using the torch backend and compare results to
    those from scipy.
    
    INPUTS:
        f_mv    -   Function from our package to test
        f_sp    -   Corresponding function from scipy. Note that, if scipy returns more than just the metric, please use a lambda to wrap the function and return only the metric.
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # test 1D
    x, y = torch.normal(0, 1, size = (sx,)), torch.normal(0, 1, size = (sx,))
    r_mv = f_mv(x, y).cpu().numpy()
    r_sp = f_sp(x.cpu().numpy(), y.cpu().numpy())
    
    assert np.allclose(r_mv, r_sp, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    
    # test 2D
    x, y = torch.normal(0, 1, size = (sx, sy)), torch.normal(0, 1, size = (sx, sy))
    r_mv = f_mv(x, y).cpu().numpy()
    r_sp = np.array([f_sp(x[i,:].cpu().numpy(), y[i,:].cpu().numpy()) for i in range(sx)])
    
    assert np.allclose(r_mv, r_sp, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    
    # test 3D
    x, y = torch.normal(0, 1, size = (sx, sy, sz)), torch.normal(0, 1, size = (sx, sy, sz))
    r_mv = f_mv(x, y).cpu().numpy()
    r_sp = np.array([[f_sp(x[i,j,:].cpu().numpy(), y[i,j,:].cpu().numpy()) for j in range(sy)] for i in range(sx)])
    
    assert np.allclose(r_mv, r_sp, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def _run_comparison_numpy_torch(f_numpy: Callable, f_torch: Callable) -> None:
    '''
    Compare the metrics between implementations in numpy and torch.
    
    INPUTS:
        f_numpy     -   Which numpy function to use?
        f_torch     -   Which torch function to use?
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # solve numpy
    x_np, y_np = np.random.normal(size = (sx, sy, sz)), np.random.normal(size = (sx, sy, sz))
    r_numpy = f_numpy(x_np, y_np)
    
    # solve torch
    x_tr, y_tr = torch.from_numpy(x_np).to(torch.float64), torch.from_numpy(y_np).to(torch.float64)
    r_torch = f_torch(x_tr, y_tr).cpu().numpy()
    
    # run test
    assert np.allclose(r_numpy, r_torch, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

'''
Euclidean tests
'''

def test_euclidean_numpy():
    '''
    Check correctness of results between our implementation and scipy on numpy backend.
    '''
    
    _run_test_numpy(mv.math.euclidean, scipy.spatial.distance.euclidean)

def test_euclidean_torch():
    '''
    Check correctness of results between our implementation and scipy on torch backend.
    '''
    
    _run_test_torch(mv.math.euclidean, scipy.spatial.distance.euclidean)

def test_euclidean_compare_numpy_torch():
    '''
    Check that both numpy and torch arrive at the same values.
    '''
    
    _run_comparison_numpy_torch(mv.math.euclidean, mv.math.euclidean)

def test_euclidean_shape_mismatch():
    '''
    Ensure that shape mismatches throw a ValueError.
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # run test
    x, y = np.random.normal(size = (sx, sy)), np.random.normal(size = (sz, sy))
    
    with pytest.raises(ValueError):
        mv.math.euclidean(x, y)
    
    with pytest.raises(ValueError):
        mv.math.euclidean(torch.from_numpy(x).to(torch.float64), torch.from_numpy(y).to(torch.float64))
    
def test_euclidean_type_mismatch():
    '''
    Ensure that type mismatches throw a ValueError.
    '''
    
    # setup dims
    sx, sy = 100, 10
    
    # run test
    x = np.random.normal(size = (sx, sy))
    y = torch.normal(0, 1, size = (sx, sy))
    
    with pytest.raises(ValueError):
        mv.math.euclidean(x, y)

'''
Mahalanobis tests
'''

def test_mahalanobis_numpy():
    '''
    Check correctness of our results from numpy against scipy.
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # test 1D
    x, y = np.random.normal(size = (sx,)), np.random.normal(size = (sx,))
    p = np.cov(np.stack((x, y), axis = 0).T)
    p = np.linalg.inv(p)
    
    r_mv = mv.math.mahalanobis(x, y, p)
    r_sp = scipy.spatial.distance.mahalanobis(x, y, p)

    if np.isnan(np.array([r_mv, r_sp])).any():
        assert np.isnan(np.array([r_mv, r_sp])).all()
    else:
        assert np.allclose(r_mv, r_sp, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    
    # test 2D
    x, y = np.random.normal(size = (sx, sy)), np.random.normal(size = (sx, sy))
    p = np.cov(np.concatenate((x, y), axis = 0).T)
    p = np.linalg.inv(p)
    
    r_mv = mv.math.mahalanobis(x, y, p)
    r_sp = np.array([scipy.spatial.distance.mahalanobis(x[i,:], y[i,:], p) for i in range(sx)])
    
    mask = ~(np.isnan(r_mv) | np.isnan(r_sp))
    assert np.allclose(r_mv[mask], r_sp[mask], rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    
    # test 3D
    x, y = np.random.normal(size = (sx, sy, sz)), np.random.normal(size = (sx, sy, sz))
    p = np.array([np.cov(np.concatenate((x[i], y[i]), axis = 0).T) for i in range(sx)]).mean(axis = 0)
    p = np.linalg.inv(p)
    
    r_mv = mv.math.mahalanobis(x, y, p)
    r_sp = np.array([[scipy.spatial.distance.mahalanobis(x[i,j,:], y[i,j,:], p) for j in range(sy)] for i in range(sx)])
    
    mask = ~(np.isnan(r_mv) | np.isnan(r_sp))
    assert np.allclose(r_mv[mask], r_sp[mask], rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_mahalanobis_torch():
    '''
    Check correctness of our results from torch against scipy.
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # test 1D
    x, y = torch.normal(0, 1, size = (sx,)), torch.normal(0, 1, size = (sx,))
    p = torch.cov(torch.stack((x, y), 0).T)
    p = torch.linalg.inv(p)
    
    with pytest.raises(NotImplementedError):
        mv.math.mahalanobis(x, y, p)
    
    # test 2D
    x, y = torch.normal(0, 1, size = (sx, sy)), torch.normal(0, 1, size = (sx, sy))
    p = torch.cov(torch.cat((x, y), 0).T)
    p = torch.linalg.inv(p)
    
    r_mv = mv.math.mahalanobis(x, y, p).cpu().numpy()
    r_sp = np.array([scipy.spatial.distance.mahalanobis(x[i,:].cpu().numpy(), y[i,:].cpu().numpy(), p.cpu().numpy()) for i in range(sx)])
    
    mask = ~(np.isnan(r_mv) | np.isnan(r_sp))
    assert np.allclose(r_mv[mask], r_sp[mask], rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    
    # test 3D
    x, y = torch.normal(0, 1, size = (sx, sy, sz)), torch.normal(0, 1, size = (sx, sy, sz))
    p = np.array([np.cov(np.concatenate((x[i].cpu().numpy(), y[i].cpu().numpy()), axis = 0).T) for i in range(sx)]).mean(axis = 0)
    p = np.linalg.inv(p)
    p = torch.from_numpy(p).to(x.dtype)
    
    r_mv = mv.math.mahalanobis(x, y, p).cpu().numpy()
    r_sp = np.array([[scipy.spatial.distance.mahalanobis(x[i,j,:].cpu().numpy(), y[i,j,:].cpu().numpy(), p.cpu().numpy()) for j in range(sy)] for i in range(sx)])
    
    mask = ~(np.isnan(r_mv) | np.isnan(r_sp))
    assert np.allclose(r_mv[mask], r_sp[mask], rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_compare_mahalanobis_numpy_torch():
    '''
    Make sure torch and numpy agree in mahalanobis distance.
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # solve numpy
    x_np, y_np = np.random.normal(size = (sx, sy, sz)), np.random.normal(size = (sx, sy, sz))
    p = np.array([np.cov(np.concatenate((x_np[i], y_np[i]), axis = 0).T) for i in range(sx)]).mean(axis = 0)
    p = np.linalg.inv(p)
    r_numpy = mv.math.mahalanobis(x_np, y_np, p)
    
    # solve torch
    x_tr, y_tr = torch.from_numpy(x_np).to(torch.float64), torch.from_numpy(y_np).to(torch.float64)
    p_tr = torch.from_numpy(p).to(torch.float64)
    r_torch = mv.math.mahalanobis(x_tr, y_tr, p_tr).cpu().numpy()
    
    # run test
    assert np.allclose(r_numpy, r_torch, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

'''
Cosine tests
'''

def test_cosine_numpy():
    '''
    Check correctness of results between our implementation and scipy on numpy backend.
    '''
    
    _run_test_numpy(mv.math.cosine_d, scipy.spatial.distance.cosine)

def test_cosine_torch():
    '''
    Check correctness of results between our implementation and scipy on torch backend.
    '''
    
    _run_test_torch(mv.math.cosine_d, scipy.spatial.distance.cosine)

def test_cosine_compare_numpy_torch():
    '''
    Check that both numpy and torch arrive at the same values.
    '''
    
    _run_comparison_numpy_torch(mv.math.cosine_d, mv.math.cosine_d)

def test_cosine_shape_mismatch():
    '''
    Ensure that shape mismatches throw a ValueError.
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # run test
    x, y = np.random.normal(size = (sx, sy)), np.random.normal(size = (sz, sy))
    
    with pytest.raises(ValueError):
        mv.math.cosine_d(x, y)
    
    with pytest.raises(ValueError):
        mv.math.cosine_d(torch.from_numpy(x).to(torch.float64), torch.from_numpy(y).to(torch.float64))

def test_cosine_type_mismatch():
    '''
    Ensure that type mismatches throw a ValueError.
    '''
    
    # setup dims
    sx, sy = 100, 10
    
    # run test
    x = np.random.normal(size = (sx, sy))
    y = torch.normal(0, 1, size = (sx, sy))
    
    with pytest.raises(ValueError):
        mv.math.cosine_d(x, y)

'''
Pearson r tests
'''

def test_pearsonr_numpy():
    '''
    Check correctness of results between our implementation and scipy on numpy backend.
    '''
    
    _run_test_numpy(mv.math.pearsonr_d, lambda x, y: 1 - scipy.stats.pearsonr(x, y).statistic)

def test_pearsonr_torch():
    '''
    Check correctness of results between our implementation and scipy on torch backend.
    '''
    
    _run_test_torch(mv.math.pearsonr_d, lambda x, y: 1 - scipy.stats.pearsonr(x, y).statistic)

def test_pearsonr_compare_numpy_torch():
    '''
    Check that both numpy and torch arrive at the same values.
    '''
    
    _run_comparison_numpy_torch(mv.math.pearsonr_d, mv.math.pearsonr_d)

def test_pearsonr_shape_mismatch():
    '''
    Ensure that shape mismatches throw a ValueError.
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # run test
    x, y = np.random.normal(size = (sx, sy)), np.random.normal(size = (sz, sy))
    
    with pytest.raises(ValueError):
        mv.math.pearsonr_d(x, y)
    
    with pytest.raises(ValueError):
        mv.math.pearsonr_d(torch.from_numpy(x).to(torch.float64), torch.from_numpy(y).to(torch.float64))

def test_pearsonr_type_mismatch():
    '''
    Ensure that type mismatches throw a ValueError.
    '''
    
    # setup dims
    sx, sy = 100, 10
    
    # run test
    x = np.random.normal(size = (sx, sy))
    y = torch.normal(0, 1, size = (sx, sy))
    
    with pytest.raises(ValueError):
        mv.math.pearsonr_d(x, y)

'''
Rank tests
'''

def test_rank_numpy():
    '''
    Ensure ranking works in numpy.
    '''
    
    # setup dims
    sx, sy = 1000, 100
    
    # run test
    x = np.random.choice(np.arange(sy), replace = True, size = (sx, sy))
    r_mv = mv.math.rank(x)
    r_sp = scipy.stats.rankdata(x, axis = -1)

    assert np.allclose(r_mv, r_sp, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_rank_torch():
    '''
    Ensure rankign works in torch.
    '''
    
    # setup dims
    sx, sy = 1000, 10
    
    # run test
    x = torch.randint(low = 0, high = sy, size = (sx, sy))
    r_mv = mv.math.rank(x).cpu().numpy()
    r_sp = scipy.stats.rankdata(x.cpu().numpy(), axis = -1)
    
    assert np.allclose(r_mv, r_sp, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_compare_rank_numpy_torch():
    '''
    Ensure numpy and torch have same solutions.
    '''
    
    # setup dims
    sx, sy = 1000, 10
    
    # run test
    x = np.random.choice(np.arange(sy), replace = True, size = (sx, sy))
    r_np = mv.math.rank(x)
    r_tr = mv.math.rank(torch.from_numpy(x))
    
    assert np.allclose(r_np, r_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_rank_type_mismatch():
    '''
    Ensure type errors are thrown while ranking data.
    '''
    
    # setup dims
    sx, sy = 1000, 10
    
    # run test
    x = np.random.choice(np.arange(sy), replace = True, size = (sx, sy))
    x = list(x)
    
    with pytest.raises(ValueError):
        r_mv = mv.math.rank(x)

'''
Spearman rho tests
'''

def test_spearmanr_numpy():
    '''
    Check correctness of results between our implementation and scipy on numpy backend.
    '''
    
    _run_test_numpy(mv.math.spearmanr_d, lambda x, y: 1 - scipy.stats.spearmanr(x, y).statistic)

def test_spearmanr_torch():
    '''
    Check correctness of results between our implementation and scipy on torch backend.
    '''
    
    _run_test_torch(mv.math.spearmanr_d, lambda x, y: 1 - scipy.stats.spearmanr(x, y).statistic)

def test_spearmanr_compare_numpy_torch():
    '''
    Check that both numpy and torch arrive at the same values.
    '''
    
    _run_comparison_numpy_torch(mv.math.spearmanr_d, mv.math.spearmanr_d)

def test_spearmanr_shape_mismatch():
    '''
    Ensure that shape mismatches throw a ValueError.
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # run test
    x, y = np.random.normal(size = (sx, sy)), np.random.normal(size = (sz, sy))
    
    with pytest.raises(ValueError):
        mv.math.spearmanr_d(x, y)
    
    with pytest.raises(ValueError):
        mv.math.spearmanr_d(torch.from_numpy(x).to(torch.float64), torch.from_numpy(y).to(torch.float64))

def test_spearmanr_type_mismatch():
    '''
    Ensure that type mismatches throw a ValueError.
    '''
    
    # setup dims
    sx, sy = 100, 10
    
    # run test
    x = np.random.normal(size = (sx, sy))
    y = torch.normal(0, 1, size = (sx, sy))
    
    with pytest.raises(ValueError):
        mv.math.spearmanr_d(x, y)

'''
Cross-validated euclidean tests
'''

def test_cv_euclidean_numpy():
    '''
    Test that our estimator is truly unbiased.
    '''
    
    # setup dims
    sx, sy, sz = 100, 5, 50
    
    # run test
    x, y = np.random.normal(size = (sx, sy, sz)), np.random.normal(size = (sx, sy, sz))
    
    assert np.isclose(mv.math.cv_euclidean(x, y).mean(), 0.0, rtol = 1.0, atol = 1.0)

def test_cv_euclidean_torch():
    '''
    Test that our estimator is truly unbiased.
    '''
    
    # setup dims
    sx, sy, sz = 100, 5, 50
    
    # run test
    x, y = torch.normal(0, 1, size = (sx, sy, sz)), torch.normal(0, 1, size = (sx, sy, sz))
    
    assert np.isclose(mv.math.cv_euclidean(x, y).mean(), 0.0, rtol = 1.0, atol = 1.0)

def test_cv_euclidean_compare_numpy_torch():
    '''
    Ensure that numpy and torch agree.
    '''
    
    # setup dims
    sx, sy, sz = 100, 5, 50
    
    # run numpy
    x, y = np.random.normal(size = (sx, sy, sz)), np.random.normal(size = (sx, sy, sz))
    r_np = mv.math.cv_euclidean(x, y)
    
    # run torch
    x_tr, y_tr = torch.from_numpy(x).to(torch.float64), torch.from_numpy(y).to(torch.float64)
    r_tr = mv.math.cv_euclidean(x, y)
    
    # run test
    assert np.allclose(r_np, r_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_cv_euclidean_mismatch_type():
    '''
    Make sure we correctly report type mismatches.
    '''
    
    # setup dims
    sx, sy, sz = 100, 5, 50
    
    # run test
    x, y = np.random.normal(size = (sx, sy, sz)), torch.normal(0, 1, size = (sx, sy, sz))
    
    with pytest.raises(ValueError):
        mv.math.cv_euclidean(x, y)

'''
Cross-validated mahalanobis tests
'''

def test_cv_mahalanobis_numpy():
    '''
    Test that our estimator is truly unbiased.
    '''
    
    # setup dims
    sx, sy, sz = 100, 5, 50
    
    # run test
    x, y = np.random.normal(size = (sx, sy, sz)), np.random.normal(size = (sx, sy, sz))
    p = np.array([np.cov(np.concatenate((x[i], y[i]), axis = 0).T) for i in range(sx)]).mean(axis = 0)
    p = np.linalg.inv(p)
    
    assert np.isclose(mv.math.cv_mahalanobis(x, y, p).mean(), 0.0, rtol = 1.0, atol = 1.0)

def test_cv_mahalanobis_torch():
    '''
    Test that our estimator is truly unbiased.
    '''

    # setup dims
    sx, sy, sz = 100, 5, 50
    
    # run test
    x, y = torch.normal(0, 1, size = (sx, sy, sz)), torch.normal(0, 1, size = (sx, sy, sz))
    p = np.array([np.cov(np.concatenate((x[i].cpu().numpy(), y[i].cpu().numpy()), axis = 0).T) for i in range(sx)]).mean(axis = 0)
    p = np.linalg.inv(p)
    p = torch.from_numpy(p).to(x.dtype)
    
    assert np.isclose(mv.math.cv_mahalanobis(x, y, p).cpu().numpy().mean(), 0.0, rtol = 1.0, atol = 1.0)

def test_cv_mahalanobis_compare_numpy_torch():
    '''
    Make sure numpy and torch backends converge in crossnobis.
    '''
    
    # setup dims
    sx, sy, sz = 100, 10, 5
    
    # solve numpy
    x_np, y_np = np.random.normal(size = (sx, sy, sz)), np.random.normal(size = (sx, sy, sz))
    p = np.array([np.cov(np.concatenate((x_np[i], y_np[i]), axis = 0).T) for i in range(sx)]).mean(axis = 0)
    p = np.linalg.inv(p)
    r_numpy = mv.math.cv_mahalanobis(x_np, y_np, p)
    
    # solve torch
    x_tr, y_tr = torch.from_numpy(x_np).to(torch.float64), torch.from_numpy(y_np).to(torch.float64)
    p_tr = torch.from_numpy(p).to(torch.float64)
    r_torch = mv.math.cv_mahalanobis(x_tr, y_tr, p_tr).cpu().numpy()
    
    # run test
    assert np.allclose(r_numpy, r_torch, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_cv_mahalanobis_type_mismatch():
    '''
    Make sure errors are thrown for type mismatches.
    '''
    
    # setup dims
    sx, sy, sz = 100, 5, 50
    
    # run test
    x, y = torch.normal(0, 1, size = (sx, sy, sz)), np.random.normal(size = (sx, sy, sz))
    p = np.array([np.cov(np.concatenate((x[i].cpu().numpy(), y[i]), axis = 0).T) for i in range(sx)]).mean(axis = 0)
    p = np.linalg.inv(p)
    p = torch.from_numpy(p).to(x.dtype)

    with pytest.raises(ValueError):
        mv.math.cv_mahalanobis(x, y, p)

'''
Accuracy tests
'''

def test_accuracy_torch():
    '''
    Make sure torch computes accuracy correctly.
    '''
    
    from sklearn.metrics import accuracy_score
    
    # setup data
    y_true = np.random.choice([0, 1], size = (100, 50))
    y_score = np.random.choice([0, 1], size = (100, 50))
    
    # run test
    r_sk = np.array([accuracy_score(y_true[i], y_score[i]) for i in range(y_true.shape[0])])
    r_mv = mv.math.accuracy(torch.from_numpy(y_true).to(torch.float32), torch.from_numpy(y_score).to(torch.float32)).cpu().numpy()

    assert np.allclose(r_sk, r_mv, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_accuracy_numpy():
    '''
    Make sure numpy computes accuracy correctly.
    '''

    from sklearn.metrics import accuracy_score

    # setup data
    y_true = np.random.choice([0, 1], size = (100, 50))
    y_score = np.random.choice([0, 1], size = (100, 50))

    # run test
    r_sk = np.array([accuracy_score(y_true[i], y_score[i]) for i in range(y_true.shape[0])])
    r_mv = mv.math.accuracy(y_true, y_score)
    
    assert np.allclose(r_sk, r_mv, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_accuracy_compare_numpy_torch():
    '''
    Make sure numpy and torch backends converge in accuracy.
    '''

    # setup data
    y_true = np.random.choice([0, 1], size = (100, 50))
    y_score = np.random.choice([0, 1], size = (100, 50))

    # run test
    r_np = mv.math.accuracy(y_true, y_score)
    r_tr = mv.math.accuracy(torch.from_numpy(y_true).to(torch.float32), torch.from_numpy(y_score).to(torch.float32)).cpu().numpy()
    
    assert np.allclose(r_np, r_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_accuracy_shape_mismatch():
    '''
    Make sure functions fail when shapes are mismatched.
    '''
    
    # setup data
    y_true = np.random.choice([0, 1], size = (100, 50))
    y_score = np.random.choice([0, 1], size = (100, 40))
    
    # run tests
    with pytest.raises(ValueError):
        mv.math.accuracy(y_true, y_score)
    
    with pytest.raises(ValueError):
        mv.math.accuracy(torch.from_numpy(y_true).to(torch.float32), torch.from_numpy(y_score).to(torch.float32))

def test_accuracy_type_mismatch():
    '''
    Make sure functions fail when types are mismatched.
    '''
    
    # setup data
    y_true = np.random.choice([0, 1], size = (100, 50))
    y_score = torch.from_numpy(np.random.choice([0, 1], size = (100, 50))).to(torch.float32)
    
    # run tests
    with pytest.raises(ValueError):
        mv.math.accuracy(y_true, y_score)

'''
ROC-AUC tests
'''

def test_roc_auc_numpy():
    '''
    Make sure numpy computes roc-auc correctly.
    '''
    
    from sklearn.metrics import roc_auc_score
    
    # setup data
    y_true = np.random.choice([0, 1], size = (100, 50))
    y_score = np.random.normal(size = (100, 50))
    
    # run test
    r_sk = np.array([roc_auc_score(y_true[i], y_score[i]) for i in range(y_true.shape[0])])
    r_mv = mv.math.roc_auc(y_true, y_score)
    
    assert np.allclose(r_sk, r_mv, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_roc_auc_torch():
    '''
    Make sure torch computes roc-auc correctly.
    '''
    
    from sklearn.metrics import roc_auc_score
    
    # setup data
    y_true = np.random.choice([0, 1], size = (100, 50))
    y_score = np.random.normal(size = (100, 50))
    
    # run test
    r_sk = np.array([roc_auc_score(y_true[i], y_score[i]) for i in range(y_true.shape[0])])
    r_mv = mv.math.roc_auc(torch.from_numpy(y_true).to(torch.float32), torch.from_numpy(y_score).to(torch.float32)).cpu().numpy()
    
    assert np.allclose(r_sk, r_mv, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_roc_auc_compare_numpy_torch():
    '''
    Make sure numpy and torch backends converge in roc-auc.
    '''

    # setup
    y_true = np.random.choice([0, 1], size = (100, 50))
    y_score = np.random.normal(size = (100, 50))
    
    # run test
    r_np = mv.math.roc_auc(y_true, y_score)
    r_tr = mv.math.roc_auc(torch.from_numpy(y_true).to(torch.float32), torch.from_numpy(y_score).to(torch.float32)).cpu().numpy()
    
    assert np.allclose(r_np, r_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_roc_auc_multiclass_numpy():
    '''
    Make sure numpy computes roc-auc correctly in multiclass case.
    '''
    
    from sklearn.metrics import roc_auc_score
    
    # setup data
    y_true = np.random.choice([0, 1, 2, 3], size = (100, 50))
    y_score = np.abs(np.random.normal(size = (100, 4, 50)))
    y_score = y_score / y_score.sum(axis = 1, keepdims = True)

    # run test
    r_sk = np.array([roc_auc_score(y_true[i].T, y_score[i].T, multi_class = 'ovr') for i in range(y_true.shape[0])])
    r_mv = mv.math.roc_auc(y_true, y_score)
    
    assert np.allclose(r_sk, r_mv, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_roc_auc_multiclass_torch():
    '''
    Make sure torch computes roc-auc correctly in multiclass case.
    '''
    
    from sklearn.metrics import roc_auc_score
    
    # setup data
    y_true = np.random.choice([0, 1, 2, 3], size = (100, 50))
    y_score = np.abs(np.random.normal(size = (100, 4, 50)))
    y_score = y_score / y_score.sum(axis = 1, keepdims = True)
    
    # run test
    r_sk = np.array([roc_auc_score(y_true[i].T, y_score[i].T, multi_class = 'ovr') for i in range(y_true.shape[0])])
    r_mv = mv.math.roc_auc(torch.from_numpy(y_true).to(torch.float32), torch.from_numpy(y_score).to(torch.float32)).cpu().numpy()
    
    assert np.allclose(r_sk, r_mv, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_roc_auc_multiclass_compare_numpy_torch():
    '''
    Make sure numpy and torch agree in their multiclass scoring.
    '''
    
    # setup data
    y_true = np.random.choice([0, 1, 2, 3], size = (100, 50))
    y_score = np.abs(np.random.normal(size = (100, 4, 50)))
    y_score = y_score / y_score.sum(axis = 1, keepdims = True)
    
    # run test
    r_np = mv.math.roc_auc(y_true, y_score)
    r_tr = mv.math.roc_auc(torch.from_numpy(y_true).to(torch.float32), torch.from_numpy(y_score).to(torch.float32)).cpu().numpy()
    
    assert np.allclose(r_np, r_tr, rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_roc_auc_shape_mismatch():
    '''
    Make sure functions fail when shapes are mismatched.
    '''
    
    # setup data
    y_true = np.random.choice([0, 1], size = (100, 50))
    y_score = np.random.normal(size = (80, 50))
    
    # run tests
    with pytest.raises(ValueError):
        mv.math.roc_auc(y_true, y_score)
    
    with pytest.raises(ValueError):
        mv.math.roc_auc(torch.from_numpy(y_true).to(torch.float32), torch.from_numpy(y_score).to(torch.float32))

def test_roc_auc_multiclass_shape_mismatch():
    '''
    Make sure functions fail when shapes are mismatched in the multiclass case, i.e. we have fewer logits than classes.
    '''
    
    # setup data
    y_true = np.random.choice([0, 1, 2, 3], size = (100, 50))
    y_score = np.abs(np.random.normal(size = (100, 3, 50)))
    
    # run tests
    with pytest.raises(ValueError):
        mv.math.roc_auc(y_true, y_score)
        
    with pytest.raises(ValueError):
        mv.math.roc_auc(torch.from_numpy(y_true).to(torch.float32), torch.from_numpy(y_score).to(torch.float32))

def test_roc_auc_type_mismatch():
    '''
    Make sure functions fail when types are mismatched.
    '''

    # setup data
    y_true = np.random.choice([0, 1], size = (100, 50))
    y_score = np.random.normal(size = (100, 50))
    
    # run tests
    with pytest.raises(ValueError):
        mv.math.roc_auc(y_true, torch.from_numpy(y_score).to(torch.float32))

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])