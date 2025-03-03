'''
Functions to compute euclidean distances in a 
nice and vectorised manner using either numpy
or torch.
'''

import numpy as np
import torch

from typing import Union

def _euclidean_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes euclidean distances using NumPy. This function is not exported and should not be called directly. See :func:`euclidean` instead.

    Parameters
    ----------
    x : np.ndarray
        Matrix ([samples ...] x features)
    y : np.ndarray
        Matrix ([samples ...] x features)
    
    Returns
    -------
    np.ndarray
        Vector or matrix of euclidean distances
    
    Notes
    -----
    Euclidean distances are defined as:
    
    .. math::

        d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}

    where :math:`x_i` and :math:`y_i` are the :math:`i`-th elements of :math:`x` and :math:`y`, respectively.
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return np.sqrt(np.sum((x - y) ** 2, axis = -1))

def _euclidean_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes euclidean distances using Torch. This function is not exported and should not be called directly. See :func:`euclidean` instead.

    Parameters
    ----------
    x : torch.Tensor
        Matrix ([samples ...] x features)
    y : torch.Tensor
        Matrix ([samples ...] x features)
    
    Returns
    -------
    torch.Tensor
        Vector or matrix of euclidean distances
    
    Notes
    -----
    
    Euclidean distances are defined as:
    
    .. math::

        d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}

    where :math:`x_i` and :math:`y_i` are the :math:`i`-th elements of :math:`x` and :math:`y`, respectively.
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return torch.sqrt(torch.sum((x - y) ** 2, -1))

def euclidean(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Computes euclidean distances between x and y.

    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Matrix ([samples ...] x features)
    y : Union[np.ndarray, torch.Tensor]
        Matrix ([samples ...] x features)
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Vector or matrix of euclidean distances
    
    Notes
    -----
    
    Euclidean distances are defined as:
    
    .. math::

        d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}

    where :math:`x_i` and :math:`y_i` are the :math:`i`-th elements of :math:`x` and :math:`y`, respectively.
    
    Examples
    --------
    >>> import torch
    >>> import mvpy as mv
    >>> x, y = torch.normal(0, 1, (10, 50)), torch.normal(0, 1, (10, 50))
    >>> d = mv.math.euclidean(x, y)
    >>> d.shape
    torch.Size([10])
    """
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _euclidean_torch(x, y)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _euclidean_numpy(x, y)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')