'''
Functions to compute 2D linear kernels (as used in, for example, SVC).
'''

import numpy as np
import torch

from typing import Union, Any

def _kernel_linear_numpy(x: np.ndarray, y: np.ndarray, *args: Any) -> np.ndarray:
    """Compute the linear kernel function.
    
    Parameters
    ----------
    x : np.ndarray
        Matrix x of shape (n_samples, n_features).
    y : np.ndarray
        Matrix y of shape (n_samples, n_features).
    
    Returns
    -------
    k : np.ndarray
        Kernel matrix of shape (n_samples, n_samples).
    
    Notes
    -----
    The linear kernel is computed as:
    
    .. math::

        \kappa(x, y) = x y^T
    
    Note that, unlike other math functions, this is specifically for 2D inputs and outputs.
    """
    
    return x.dot(y.T)

def _kernel_linear_torch(x: torch.Tensor, y: torch.Tensor, *args: Any) -> torch.Tensor:
    """Compute the linear kernel function.
    
    Parameters
    ----------
    x : torch.Tensor
        Matrix x of shape (n_samples, n_features).
    y : torch.Tensor
        Matrix y of shape (n_samples, n_features).
    
    Returns
    -------
    k : torch.Tensor
        Kernel matrix of shape (n_samples, n_samples).
    
    Notes
    -----
    The linear kernel is computed as:
    
    .. math::

        \kappa(x, y) = x y^T
    
    Note that, unlike other math functions, this is specifically for 2D inputs and outputs.
    """
    
    return x @ y.T

def kernel_linear(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
    """Compute the linear kernel function.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Matrix x of shape (n_samples, n_features).
    y : Union[np.ndarray, torch.Tensor]
        Matrix y of shape (n_samples, n_features).
    
    Returns
    -------
    k : Union[np.ndarray, torch.Tensor]
        Kernel matrix of shape (n_samples, n_samples).
    
    Notes
    -----
    The linear kernel is computed as:
    
    .. math::

        \kappa(x, y) = x y^T
    
    Note that, unlike other math functions, this is specifically for 2D inputs and outputs.
    """
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _kernel_linear_torch(x, y)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _kernel_linear_numpy(x, y)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')