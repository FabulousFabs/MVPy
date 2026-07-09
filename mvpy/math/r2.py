'''
Functions to rank data in a nice and vectorised manner using 
either numpy or torch.
'''

import numpy as np
import torch

from typing import Union, Optional

def _r2_numpy(y: np.ndarray, y_h: np.ndarray, y_b: Optional[np.ndarray] = None) -> np.ndarray:
    """Rank torch tensor along final dimension. Ties are computed as averages.

    Parameters
    ----------
    y : np.ndarray
        True outcomes of shape ([...,] n_features).
    y_h : np.ndarray
        Predicted outcomes of shape ([...,] n_features).
    y_b : Optional[np.ndarray], default=None
        Mean of the training data, if available. Must match shape of y with first dimension of size one.
    
    Returns
    -------
    r2 : np.ndarray
        R2 scores of shape ([...,]).
    """
    
    # check shape
    if y.shape != y_h.shape:
        raise ValueError(f'`y` and `y_h` must have the same shape, but got {y.shape} and {y_h.shape}.')
    
    # check training data mu
    if y_b is None:
        y_b = y.mean(-1, keepdims = True)
    else:
        n_dim_y = len(y.shape)
        n_dim_b = len(y_b.shape)
        
        if n_dim_y != n_dim_b:
            raise ValueError(
                f'When supplying `y_b`, it must have the same number of ' + 
                f'dimensions as `y`, but got {n_dim_y} and {n_dim_b}.'
            )
    
    # compute numerator and denominator
    num = ((y - y_h) ** 2).sum(-1)
    den = ((y - y_b) ** 2).sum(-1)
    
    # make NaN-safe
    mask = (den > 0.0)
    r2 = np.zeros(y.shape[:-1])
    r2[mask] = 1 - num[mask] / den[mask]
    
    return r2

def _r2_torch(y: torch.Tensor, y_h: torch.Tensor, y_b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Rank torch tensor along final dimension. Ties are computed as averages.

    Parameters
    ----------
    y : torch.Tensor
        True outcomes of shape ([...,] n_features).
    y_h : torch.Tensor
        Predicted outcomes of shape ([...,] n_features).
    y_b : Optional[torch.Tensor], default=None
        Mean of the training data, if available. Must match shape of y with first dimension of size one.
    
    Returns
    -------
    r2 : torch.Tensor
        R2 scores of shape ([...,]).
    """
    
    # check shape
    if y.shape != y_h.shape:
        raise ValueError(f'`y` and `y_h` must have the same shape, but got {y.shape} and {y_h.shape}.')
    
    # check training data mu
    if y_b is None:
        y_b = y.mean(-1, keepdim = True)
    else:
        # verify shape
        if (len(y.shape) != len(y_b.shape)) or (y.shape[1:] != y_b.shape):
            raise ValueError(
                f'When supplying `y_b`, it must be of shape (1, **y.shape[1:]), ' + 
                f'but got y={y.shape} and y_b={y_b.shape}.'
            )
    
    # compute numerator and denominator
    num = ((y - y_h) ** 2).sum(-1)
    den = ((y - y_b) ** 2).sum(-1)
    
    # make NaN-safe
    mask = (den > 0.0)
    r2 = torch.zeros(*y.shape[:-1], dtype = y.dtype, device = y.device)
    r2[mask] = 1 - num[mask] / den[mask]
    
    return r2

def r2(y: Union[np.ndarray, torch.Tensor], y_h: Union[np.ndarray, torch.Tensor], y_b: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Union[np.ndarray, torch.Tensor]:
    """Compute :math:`R^2` between the final dimension of :math:`y` and :math:`\\hat{y}`.
    
    :math:`R^2`, also referred to as the coefficient of determination, is computed as:
    
    .. math::
        R^2 = 1 - \\frac{\\sum_i{(y_i - \hat{y}_i)^2}}{\\sum_i{(y_i - \\bar{y})^2}}
    
    where :math:`i` are samples and :math:`\\bar{y}` is the mean over observed samples.
    
    .. warning::
        In cross-validated procedures, :math:`\\bar{y}` naturally represents the mean of 
        the test distribution. Principally, this constitutes a form of data leakage, as 
        the trained model should not have access to those data. In practice, this leads 
        to miscalibrated :math:`R^2` computations and should be avoided. To remedy this, 
        please supply :py:attr:`y_b` in these cases which will then be substituted 
        for :math:`\\bar{y}` in computations.
    
    Parameters
    ----------
    y : np.ndarray | torch.Tensor
        True outcomes of shape ``([n_samples, ...,] n_features)``.
    y_h : np.ndarray | torch.Tensor
        Predicted outcomes of shape ``([n_samples, ...,] n_features)``.
    y_b : np.ndarray | torch.Tensor | None, default=None
        Mean of the training data, if available. Must match shape of y with first dimension of size one.
    
    Returns
    -------
    r : np.ndarray | torch.Tensor
        :math:`R^2` scores of shape ``([n_samples, ...])``.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import rank
    >>> y = torch.tensor([1.0, 2.0, 3.0])
    >>> y_h = torch.tensor([1.0, 2.0, 3.0])
    >>> r2(x)
    tensor([1.0])
    """
    
    if isinstance(y, torch.Tensor) & isinstance(y_h, torch.Tensor) & ((y_b is None) or (isinstance(y_b, torch.Tensor))):
        return _r2_torch(y, y_h, y_b = y_b)
    elif isinstance(y, np.ndarray) & isinstance(y_h, np.ndarray) & ((y_b is None) or (isinstance(y_b, np.ndarray))):
        return _r2_numpy(y, y_h, y_b = y_b)
    
    raise ValueError(f'`y`, `y_h` and, if supplied, `y_b` must be of type np.ndarray or torch.Tensor, but received {type(y)}, {type(y_h)} and {type(y_b)} instead.')