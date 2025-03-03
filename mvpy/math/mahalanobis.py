'''
Functions to compute mahalanobis distances in a 
nice and vectorised manner using either numpy
or torch.
'''

import numpy as np
import torch

from typing import Union

def _mahalanobis_numpy(x: np.ndarray, y: np.ndarray, Σ: np.ndarray) -> np.ndarray:
    """Computes mahalanobis distance between x and y using inverse covariance matrix Σ. Note that this function is not exported and should not be called directly. See :func:`mahalanobis` instead.

    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Matrix ([samples ...] x features)
    y : Union[np.ndarray, torch.Tensor]
        Matrix ([samples ...] x features)
    Σ : Union[np.ndarray, torch.Tensor]
        Precision matrix (features x features)
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Vector or matrix of distances.
    
    Notes
    -----
    Mahalanobis distance is defined as:
    
    .. math::
        d(x, y) = \sqrt{(x - y)^T Σ^{-1} (x - y)}
    
    where :math:`x` and :math:`y` are the matrices to compute the distance between, and :math:`Σ` is the covariance matrix.
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    if Σ.shape != (x.shape[-1], x.shape[-1]):
        raise ValueError(f'`Σ` must be of size `(x[-1], x[-1])`, but received x={x.shape} and Σ={Σ.shape}.')
    
    Δ = x - y
    L = np.matmul(Δ[...,np.newaxis,:], Σ)
    S = np.matmul(L, Δ[...,np.newaxis]).squeeze(-1).squeeze(-1)
    
    return np.sqrt(S)

def _mahalanobis_torch(x: torch.Tensor, y: torch.Tensor, Σ: torch.Tensor) -> torch.Tensor:
    """Computes mahalanobis distance between x and y using inverse covariance matrix Σ. Note that this function is not exported and should not be called directly. See :func:`mahalanobis` instead.

    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Matrix ([samples ...] x features)
    y : Union[np.ndarray, torch.Tensor]
        Matrix ([samples ...] x features)
    Σ : Union[np.ndarray, torch.Tensor]
        Precision matrix (features x features)
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Vector or matrix of distances.
    
    Notes
    -----
    Mahalanobis distance is defined as:
    
    .. math::
        d(x, y) = \sqrt{(x - y)^T Σ^{-1} (x - y)}
    
    where :math:`x` and :math:`y` are the matrices to compute the distance between, and :math:`Σ` is the covariance matrix.
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')

    if Σ.shape != (x.shape[-1], x.shape[-1]):
        raise ValueError(f'`Σ` must be of size `(x[-1], x[-1])`, but received x={x.shape} and Σ={Σ.shape}.')
    
    '''
    @TODO: Look into why on earth the 1D test case kept failing for torch; there seems to be something wrong.
    '''
    if x.dim() == 1:
        raise NotImplementedError(f'Mahalanobis distance currently unavailable for 1d vectors on torch.')
    
    Δ = x - y
    L = torch.matmul(Δ[...,None,:], Σ)
    S = torch.matmul(L, Δ[...,None]).squeeze(-1).squeeze(-1)
    
    return torch.sqrt(S)

def mahalanobis(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], Σ: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Computes mahalanobis distance between x and y using inverse covariance matrix Σ.

    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Matrix ([samples ...] x features)
    y : Union[np.ndarray, torch.Tensor]
        Matrix ([samples ...] x features)
    Σ : Union[np.ndarray, torch.Tensor]
        Precision matrix (features x features)
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Vector or matrix of distances.
    
    Notes
    -----
    Mahalanobis distance is defined as:
    
    .. math::
        d(x, y) = \sqrt{(x - y)^T Σ^{-1} (x - y)}
    
    where :math:`x` and :math:`y` are the matrices to compute the distance between, and :math:`Σ` is the covariance matrix.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import mahalanobis
    >>> from mvpy.estimators import Covariance
    >>> x, y = torch.normal(0, 1, (10, 50)), torch.normal(0, 1, (10, 50))
    >>> Σ = Covariance().fit(torch.cat((x, y), 0)).covariance_
    >>> Σ = torch.linalg.inv(Σ)
    >>> d = mahalanobis(x, y, Σ)
    >>> d.shape
    torch.Size([10])
    """
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor) & isinstance(Σ, torch.Tensor):
        return _mahalanobis_torch(x, y, Σ)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray) & isinstance(Σ, np.ndarray):
        return _mahalanobis_numpy(x, y, Σ)
    
    raise ValueError(f'`x`, `y` and `Σ` must be of the same type, but got `{type(x)}`, `{type(y)}` and `{type(Σ)}` instead.')