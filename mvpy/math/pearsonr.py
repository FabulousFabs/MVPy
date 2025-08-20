'''
Functions to compute Pearson correlation or distance in a 
nice and vectorised manner using either numpy or 
torch.
'''

import numpy as np
import torch

from typing import Union, Any

def _pearsonr_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes pearson correlations between final dimensions of x and y. Note that this function is not exported and should not be called directly.
    
    Parameters
    ----------
    x : np.ndarray
        Matrix ([samples ...] x features)
    y : np.ndarray
        Matrix ([samples ...] x features)
    
    Returns
    -------
    np.ndarray
        Vector or matrix of pearson correlations
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    Δx, Δy = (x - x.mean(-1, keepdims = True)), (y - y.mean(-1, keepdims = True))
    
    return np.sum(Δx * Δy, axis = -1) / np.sqrt(np.sum(Δx ** 2, axis = -1) * np.sum(Δy ** 2, axis = -1))

def _pearsonr_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes pearson correlations between final dimensions of x and y. Note that this function is not exported and should not be called directly.
    
    Parameters
    ----------
    x : torch.Tensor
        Matrix ([samples ...] x features)
    y : torch.Tensor
        Matrix ([samples ...] x features)
    
    Returns
    -------
    torch.Tensor
        Pearson correlations.
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    Δx, Δy = (x - x.mean(-1, keepdim = True)), (y - y.mean(-1, keepdim = True))
    
    return torch.sum(Δx * Δy, -1) / torch.sqrt(torch.sum(Δx ** 2, -1) * torch.sum(Δy ** 2, -1))

def pearsonr(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
    """Computes pearson correlations between x and y. Note that correlations are always computed over the final dimension.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Matrix ([samples ...] x features)
    y : Union[np.ndarray, torch.Tensor]
        Matrix ([samples ...] x features)
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Vector or matrix of pearson correlations
    
    Notes
    -----
    Pearson correlations are defined as:

    .. math:: r = \\frac{\\sum{(x_i - \\bar{x})(y_i - \\bar{y})}}{\\sqrt{\sum{(x_i - \\bar{x})^2} \\sum{(y_i - \\bar{y})^2}}}
    
    where :math:`x_i` and :math:`y_i` are the :math:`i`-th elements of :math:`x` and :math:`y`, respectively.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import pearsonr
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([4, 5, 6])
    >>> pearsonr(x, y)
    tensor(1., dtype=torch.float64)
    """
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _pearsonr_torch(x, y)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _pearsonr_numpy(x, y)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')

def pearsonr_d(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
    """Computes Pearson distance between x and y. Note that distances are always computed over the final dimension in your inputs.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Vector/Matrix/Tensor
    y : Union[np.ndarray, torch.Tensor]
        Vector/Matrix/Tensor
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Distances
    
    Notes
    -----
    Pearson distance is defined as :math:`1 - \\text{pearsonr}(x, y)`.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import pearsonr_d
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([-1, -2, -3])
    >>> pearsonr_d(x, y)
    tensor(2.0, dtype=torch.float64)
    
    See also
    --------
    :func:`mvpy.math.pearsonr`
    """
    
    return 1 - pearsonr(x, y)