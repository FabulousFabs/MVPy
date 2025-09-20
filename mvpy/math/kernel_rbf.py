'''
Functions to compute 2D rbf kernels (as used in, for example, SVC).
'''

import numpy as np
import torch
import torch.nn.functional as F

from typing import Union, Any

def _kernel_rbf_numpy(x: np.ndarray, y: np.ndarray, γ: float, *args: Any) -> np.ndarray:
    """Compute the radial basis kernel function.
    
    Parameters
    ----------
    x : np.ndarray
        Matrix x of shape (n_samples, n_features).
    y : np.ndarray
        Matrix y of shape (n_samples, n_features).
    γ : float
        Gamma parameter as float.
    
    Returns
    -------
    k : np.ndarray
        Kernel matrix of shape (n_samples, n_samples).
    
    Notes
    -----
    The radial basis kernel is computed as:
    
    .. math::

        \kappa(x, y) = -\gamma \lvert\lvert x - y\rvert\rvert^2
    
    Note that, unlike other math functions, this is specifically for 2D inputs and outputs.
    """
    
    return np.exp(
        -γ * np.linalg.norm(x[:,None] - y[None,:], axis = 2) ** 2
    )

def _kernel_rbf_torch(x: torch.Tensor, y: torch.Tensor, γ: float, *args: Any) -> torch.Tensor:
    """Compute the radial basis kernel function.
    
    Parameters
    ----------
    x : torch.Tensor
        Matrix x of shape (n_samples, n_features).
    y : torch.Tensor
        Matrix y of shape (n_samples, n_features).
    γ : float
        Gamma parameter as float.
    
    Returns
    -------
    k : torch.Tensor
        Kernel matrix of shape (n_samples, n_samples).
    
    Notes
    -----
    The radial basis kernel is computed as:
    
    .. math::

        \kappa(x, y) = -\gamma \lvert\lvert x - y\rvert\rvert^2
    
    Note that, unlike other math functions, this is specifically for 2D inputs and outputs.
    """
    
    return torch.exp(
        -γ * torch.linalg.norm(x[:,None,:] - y[None,:,:], dim = 2) ** 2
    )

def kernel_rbf(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], γ: float, *args: Any) -> Union[np.ndarray, torch.Tensor]:
    """Compute the radial basis kernel function.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Matrix x of shape (n_samples, n_features).
    y : Union[np.ndarray, torch.Tensor]
        Matrix y of shape (n_samples, n_features).
    γ : float
        Gamma parameter as float.
    
    Returns
    -------
    k : Union[np.ndarray, torch.Tensor]
        Kernel matrix of shape (n_samples, n_samples).
    
    Notes
    -----
    The radial basis kernel is computed as:
    
    .. math::

        \kappa(x, y) = -\gamma \lvert\lvert x - y\rvert\rvert^2
    
    Note that, unlike other math functions, this is specifically for 2D inputs and outputs.
    """
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _kernel_rbf_torch(x, y, γ)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _kernel_rbf_numpy(x, y, γ)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')