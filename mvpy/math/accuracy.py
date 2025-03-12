'''
Functions to compute accuracy in a 
nice and vectorised manner using either numpy or 
torch.
'''

import numpy as np
import torch

from typing import Union

def _accuracy_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute accuracy between x and y. Note that this function is not exported and should not be called directly.
    
    Parameters
    ----------
    x : np.ndarray
        Vector/Matrix/Tensor
    y : np.ndarray
        Vector/Matrix/Tensor
    
    Returns
    -------
    np.ndarray
        Accuracy
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return (x == y).mean(axis = -1)

def _accuracy_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute accuracy between x and y. Note that this function is not exported and should not be called directly.
    
    Parameters
    ----------
    x : torch.Tensor
        Vector/Matrix/Tensor
    y : torch.Tensor
        Vector/Matrix/Tensor
    
    Returns
    -------
    torch.Tensor
        Accuracy
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return (x == y).to(x.dtype).to(x.device).mean(-1)

def accuracy(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Compute accuracy between x and y. Note that accuracy is always computed over the final dimension.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Vector/Matrix/Tensor
    y : Union[np.ndarray, torch.Tensor]
        Vector/Matrix/Tensor
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Accuracy
    
    Notes
    -----
    Accuracy is defined as:
    
    .. math::

        \\text{accuracy}(x, y) = \\frac{1}{N}\\sum_i^N{1(x_i = y_i)}
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import accuracy
    >>> x = torch.tensor([1, 0])
    >>> y = torch.tensor([-1, 0])
    >>> accuracy(x, y)
    tensor([0.5])
    """
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _accuracy_torch(x, y)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _accuracy_numpy(x, y)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')