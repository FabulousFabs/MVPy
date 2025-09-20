'''
Functions to compute 2D sigmoid kernels (as used in, for example, SVC).
'''

import numpy as np
import torch
import torch.nn.functional as F

from typing import Union, Any

def _kernel_sigmoid_numpy(x: np.ndarray, y: np.ndarray, γ: float, coef0: float, *args: Any) -> np.ndarray:
    """Compute the sigmoid kernel function.
    
    Parameters
    ----------
    x : np.ndarray
        Matrix x of shape (n_samples, n_features).
    y : np.ndarray
        Matrix y of shape (n_samples, n_features).
    γ : float
        Gamma parameter as float.
    coef0 : float
        Coefficient parameter as float.
    
    Returns
    -------
    k : np.ndarray
        Kernel matrix of shape (n_samples, n_samples).
    
    Notes
    -----
    The sigmoid kernel is computed as:
    
    .. math::

        \kappa(x, y) = \tanh(c_0 + \gamma x y^T)
    
    Note that, unlike other math functions, this is specifically for 2D inputs and outputs.
    """
    
    return np.tanh(coef0 + γ * x.dot(y.T))

def _kernel_sigmoid_torch(x: torch.Tensor, y: torch.Tensor, γ: float, coef0: float, *args: Any) -> torch.Tensor:
    """Compute the sigmoid kernel function.
    
    Parameters
    ----------
    x : torch.Tensor
        Matrix x of shape (n_samples, n_features).
    y : torch.Tensor
        Matrix y of shape (n_samples, n_features).
    γ : float
        Gamma parameter as float.
    coef0 : float
        Coefficient parameter as float.
    
    Returns
    -------
    k : torch.Tensor
        Kernel matrix of shape (n_samples, n_samples).
    
    Notes
    -----
    The sigmoid kernel is computed as:
    
    .. math::

        \kappa(x, y) = \tanh(c_0 + \gamma x y^T)
    
    Note that, unlike other math functions, this is specifically for 2D inputs and outputs.
    """
    
    return F.tanh(coef0 + γ * (x @ y.T))

def kernel_sigmoid(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], γ: float, coef0: float, *args: Any) -> Union[np.ndarray, torch.Tensor]:
    """Compute the sigmoid kernel function.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Matrix x of shape (n_samples, n_features).
    y : Union[np.ndarray, torch.Tensor]
        Matrix y of shape (n_samples, n_features).
    γ : float
        Gamma parameter as float.
    coef0 : float
        Coefficient parameter as float.
    
    Returns
    -------
    k : Union[np.ndarray, torch.Tensor]
        Kernel matrix of shape (n_samples, n_samples).
    
    Notes
    -----
    The sigmoid kernel is computed as:
    
    .. math::

        \kappa(x, y) = \tanh(c_0 + \gamma x y^T)
    
    Note that, unlike other math functions, this is specifically for 2D inputs and outputs.
    """
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _kernel_sigmoid_torch(x, y, γ, coef0)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _kernel_sigmoid_numpy(x, y, γ, coef0)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')