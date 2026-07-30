from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import pytest

from tests._common.backends import Backend

@pytest.fixture
def backend(request: pytest.FixtureRequest) -> Backend:
    """
    """
    
    # get requested backend
    name = request.param
    
    # check numpy
    if name == 'numpy':
        return Backend('numpy')
    
    # check torch (cpu)
    if name == 'torch-cpu':
        return Backend(
            'torch-cpu',
            device = torch.device('cpu')
        )
    
    # check torch (cuda)
    if name == 'torch-cuda':
        # check availability
        if not torch.cuda.is_available():
            pytest.skip('CUDA is not available.')
        
        return Backend(
            'torch-cuda',
            device = torch.device('cuda')
        )
    
    # check torch (mps)
    if name == 'torch-mps':
        # check availability
        if not torch.backends.mps.is_available('mps'):
            pytest.skip('MPS is not available.')
        
        return Backend(
            'torch-mps',
            device = torch.device('mps')
        )
    
    # otherwise fail
    raise ValueError(
        f'Unknown test backend: {name!r}.'
    )

@pytest.fixture
def regression_problem() -> Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    """
    
    def make(n_samples: int, n_features: int, n_targets: int = 1, snr: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        """
        
        # set rng
        rng = np.random.default_rng(42)
        
        # make data
        X = rng.normal(size = (n_samples, n_features))
        ß = rng.normal(size = (n_features, n_targets))
        ε = rng.normal(size = (n_samples, n_targets))
        
        y = X @ ß
        
        snr_current = ((y ** 2).mean() / (ε ** 2).mean())
        snr_factor = snr_current / snr
        y = y + ε * snr_factor

        # check output
        if n_targets == 1:
            y = y[:,0]
        
        return X, y, ß

    return make

@pytest.fixture
def dist_normal() -> Callable[..., np.ndarray]:
    """
    """
    
    def make(n_dims: tuple[int]) -> np.ndarray:
        """
        """
        
        # set rng
        rng = np.random.default_rng(42)
        
        # make data
        X = rng.normal(size = n_dims)
        
        return X
    
    return make    