'''
Functions to rank data in a nice and vectorised manner using 
either numpy or torch.
'''

import numpy as np
import torch

from typing import Union

def _r2_numpy(y: np.ndarray, y_h: np.ndarray) -> np.ndarray:
    """Rank torch tensor along final dimension. Ties are computed as averages.

    Parameters
    ----------
    y : np.ndarray
        True outcomes of shape ([...,] n_features).
    y_h : np.ndarray
        Predicted outcomes of shape ([...,] n_features).
        
    Returns
    -------
    r2 : np.ndarray
        R2 scores of shape ([...,]).
    """
    
    # check shape
    if y.shape != y_h.shape:
        raise ValueError(f'`y` and `y_h` must have the same shape, but got {y.shape} and {y_h.shape}.')
    
    # compute numerator and denominator
    num = ((y - y_h) ** 2).sum(-1)
    den = ((y - y.mean(-1, keepdims = True)) ** 2).sum(-1)
    
    # make NaN-safe
    mask = (den > 0.0)
    r2 = np.zeros(y.shape[:-1])
    r2[mask] = 1 - num[mask] / den[mask]
    
    return r2

def _r2_torch(y: torch.Tensor, y_h: torch.Tensor) -> torch.Tensor:
    """Rank torch tensor along final dimension. Ties are computed as averages.

    Parameters
    ----------
    y : torch.Tensor
        True outcomes of shape ([...,] n_features).
    y_h : torch.Tensor
        Predicted outcomes of shape ([...,] n_features).
        
    Returns
    -------
    r2 : torch.Tensor
        R2 scores of shape ([...,]).
    """
    
    # check shape
    if y.shape != y_h.shape:
        raise ValueError(f'`y` and `y_h` must have the same shape, but got {y.shape} and {y_h.shape}.')
    
    # compute numerator and denominator
    num = ((y - y_h) ** 2).sum(-1)
    den = ((y - y.mean(-1, keepdim = True)) ** 2).sum(-1)
    
    # make NaN-safe
    mask = (den > 0.0)
    r2 = torch.zeros(*y.shape[:-1], dtype = y.dtype, device = y.device)
    r2[mask] = 1 - num[mask] / den[mask]
    
    return r2

def r2(y: Union[np.ndarray, torch.Tensor], y_h: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Rank data in x along its final feature dimension. Ties are computed as averages.
    
    Parameters
    ----------
    y : Union[np.ndarray, torch.Tensor]
        True outcomes of shape ``([n_samples, ...,] n_features)``.
    y_h : Union[np.ndarray, torch.Tensor]
        Predicted outcomes of shape ``([n_samples, ...,] n_features)``.
    
    Returns
    -------
    r : Union[np.ndarray, torch.Tensor]
        R2 scores of shape ``([n_samples, ...])``.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import rank
    >>> y = torch.tensor([1.0, 2.0, 3.0])
    >>> y_h = torch.tensor([1.0, 2.0, 3.0])
    >>> r2(x)
    tensor([1.0])
    """
    
    if isinstance(y, torch.Tensor) & isinstance(y_h, torch.Tensor):
        return _r2_torch(y, y_h)
    elif isinstance(y, np.ndarray) & isinstance(y_h, np.ndarray):
        return _r2_numpy(y, y_h)
    
    raise ValueError(f'`y` and `y_h` must be of type np.ndarray or torch.Tensor, but received {type(y)} and {type(y_h)} instead.')