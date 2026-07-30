import numpy as np
import torch
import pytest

from mvpy.preprocessing import Scaler

from tests._common.assertions import assert_allclose, assert_equalshape, assert_equaldevice
from tests._common.backends import Backend, ALL_BACKENDS, TORCH_BACKENDS
from tests._common.dtypes import to_numpy

from typing import Callable

@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
    indirect = True,
)
@pytest.mark.parametrize(
    "shape,dims",
    [
        pytest.param((12,), None, id = '1d-default'),
        pytest.param((12, 4), None, id = '2d-default'),
        pytest.param((12, 4), 0, id = '2d-0'),
        pytest.param((12, 4), 1, id = '2d-1'),
        pytest.param((12, 4), (0, 1), id = '2d-0-1'),
        pytest.param((12, 8, 4), None, id = '3d-default'),
        pytest.param((12, 8, 4), 0, id = '3d-0'),
        pytest.param((12, 8, 4), 1, id = '3d-1'),
        pytest.param((12, 8, 4), 2, id = '3d-2'),
        pytest.param((12, 8, 4), (0, 1), id = '3d-0-1'),
        pytest.param((12, 8, 4), (0, 2), id = '3d-0-2'),
        pytest.param((12, 8, 4), (1, 2), id = '3d-1-2'),
        pytest.param((12, 8, 4), (0, 1, 2), id = '3d-0-1-2')
    ],
)
@pytest.mark.parametrize(
    "with_mean",
    [
        pytest.param(True, id = "with-mean"),
        pytest.param(False, id = "without-mean"),
    ],
)
@pytest.mark.parametrize(
    "with_std",
    [
        pytest.param(True, id = "with-std"),
        pytest.param(False, id = "without-std"),
    ],
)
def test_scaler_matches_statistics(backend: Backend, dist_normal: Callable, shape: tuple[int], dims: tuple[int], with_mean: bool, with_std: bool) -> None:
    """Test whether mvpy.preprocessing.Scaler matches our reference computations here.
    """
    
    # make data
    X_np = dist_normal(shape)
    
    # convert data
    X = backend.asarray(X_np)
    
    # create expected data
    dims_ = dims if dims is not None else 0
    if isinstance(dims, int): dims = (dims,) 
    
    if backend.name == 'numpy':
        expected = X.copy()
        
        mu = X.mean(axis = dims_, keepdims = True)
        std = X.std(axis = dims_, keepdims = True, ddof = 1)
        std[np.isnan(std)] = 1.0
    else:
        expected = X.clone()
        
        mu = X.mean(dim = dims_, keepdim = True)
        std = X.std(dim = dims_, keepdim = True)
        std[torch.isnan(std)] = 1.0
    
    std[(std == 0.0)] = 1.0
    
    if with_mean:
        expected -= mu
    
    if with_std:
        expected /= std
    
    # create observed data
    opts = dict(dims = dims, with_mean = with_mean, with_std = with_std)
    scaler = Scaler(**opts).to_numpy() if backend.name == 'numpy' else Scaler(**opts).to_torch()
    observed = scaler.fit_transform(X)
    
    # assert outcome
    assert_allclose(observed, expected)
    
    # create inverted data
    inverted = scaler.inverse_transform(observed)
    
    # assert backtransform
    assert_allclose(inverted, X)

@pytest.mark.parametrize(
    "with_mean,with_std",
    [
        pytest.param(True, True, id = "with-mean-with-std"),
        pytest.param(True, False, id = "with-mean-without-std"),
        pytest.param(False, True, id = "without-mean-with-std")
    ],
)
@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
    indirect = True,
)
def test_scaler_requires_fit_before_transform(backend: Backend, dist_normal: Callable, with_mean: bool, with_std: bool):
    """Test whether mvpy.preprocessing.Scaler requires fitting before transform calls.
    """
    
    # make data
    X_np = dist_normal((10, 4))
    
    # convert data
    X = backend.asarray(X_np)
    
    # create scaler
    opts = dict(with_mean = with_mean, with_std = with_std)
    scaler = Scaler(**opts).to_numpy() if backend.name == 'numpy' else Scaler(**opts).to_torch()
    
    with pytest.raises(ValueError):
        scaler.transform(X)
    
    with pytest.raises(ValueError):
        scaler.inverse_transform(X)

@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
    indirect = True,
)
def test_scaler_does_not_mutate_input(backend: Backend, dist_normal: Callable):
    """Test whether mvpy.preprocessing.Scaler mutates inputs.
    """
    
    # make data
    X_np = dist_normal((10, 4))
    
    # convert data
    X = backend.asarray(X_np)
    
    # save and fit
    X_bk = X_np.copy()
    scaler = Scaler().to_numpy() if backend.name == 'numpy' else Scaler().to_torch()
    X_tr = scaler.fit_transform(X)
    X_rt = scaler.inverse_transform(X_tr)
    
    # assert data remains unchanged
    assert_allclose(X, X_bk)

@pytest.mark.parametrize(
    "backend",
    TORCH_BACKENDS,
    indirect = True,
)
def test_scaler_preserves_device(backend: Backend, dist_normal: Callable):
    """Test whether mvpy.preprocessing.Scaler preserves devices during fitting and transform.
    """
    
    # make data
    X_np = dist_normal((10, 4))
    
    # convert data
    X = backend.asarray(X_np)
    
    # make observed data
    scaler = Scaler().to_torch().fit(X)
    X_h = scaler.transform(X)
    X_b = scaler.inverse_transform(X_h)
    
    # test devices
    assert_equaldevice(X_h, X)
    assert_equaldevice(scaler.mean_, X)
    assert_equaldevice(scaler.scale_, X)
    assert_equaldevice(X_b, X)

@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
    indirect = True,
)
def test_scaler_uses_training_statistics(backend: Backend, dist_normal: Callable):
    """Test whether mvpy.preprocessing.Scaler uses training statistics appropriately on test data.
    """
    
    # make data
    X_np = dist_normal((10, 4))
    n_np = np.arange(5).astype(int)
    z_np = np.arange(5, 10).astype(int)
    
    # convert data
    X = backend.asarray(X_np)
    n = backend.asarray(n_np)
    z = backend.asarray(z_np)
    
    # compute reference
    if backend.name == 'numpy':
        n = n.astype(int)
        z = z.astype(int)
        
        mu = X[n].mean(0, keepdims = True)
        std = X[n].std(0, keepdims = True, ddof = 1)
        std[np.isnan(std)] = 1.0
    else:
        n = n.to(torch.int)
        z = z.to(torch.int)
        
        mu = X[n].mean(0, keepdim = True)
        std = X[n].std(0, keepdim = True)
        std[torch.isnan(std)] = 1.0
    
    std[std == 0.0] = 1.0
    expected = (X[z] - mu) / std
    
    # fit and get observed
    scaler = Scaler().to_numpy() if backend.name == 'numpy' else Scaler().to_torch()
    scaler.fit(X[n])
    observed = scaler.transform(X[z])
    
    # test correct usage
    assert_allclose(observed, expected)