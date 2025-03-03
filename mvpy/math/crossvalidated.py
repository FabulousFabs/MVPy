'''
Wrapper functions to compute cross-validated metrics from
the metrics already in this package.
'''

import numpy as np
import torch

from sklearn.model_selection import KFold

from typing import Union, Any, Callable

def _cv_numpy(x: np.ndarray, y: np.ndarray, estimator: Callable, *args: Any, n_splits: Union[int, None] = None) -> np.ndarray:
    """Computes cross-validation steps of `estimator` over `x`, `y` and `args` using torch as backend.
    
    Parameters
    ----------
    x : np.ndarray
        Matrix ([samples ...] x features)
    y : np.ndarray
        Matrix ([samples ...] x features)
    estimator : Callable
        Which estimator should be used?
    *args : Any
        Additional arguments to pass to `estimator`
    n_splits : Union[int, None], optional
        Number of splits to use for cross-validation, by default None
    
    Returns
    -------
    np.ndarray
        Matrix of cross-validated distances
    """
    
    if n_splits is None:
        n_splits = x.shape[0]
    
    cv_outer = KFold(n_splits = n_splits)
    cv_inner = KFold(n_splits = n_splits)
    
    r = np.zeros((n_splits, n_splits, x.shape[0] // n_splits, *x.shape[1:-1]))
    
    for f_i, (train_i, test_i) in enumerate(cv_outer.split(x)):
        for f_j, (train_j, test_j) in enumerate(cv_inner.split(x)):
            if f_i <= f_j: continue
            
            r[f_i,f_j] = r[f_j,f_i] = estimator(x, y, test_i, test_j, *args)

    return r.mean(axis = (0, 1, 2))

def _cv_torch(x: torch.Tensor, y: torch.Tensor, estimator: Callable, *args: Any, n_splits: Union[int, None] = None) -> torch.Tensor:
    """Computes cross-validation steps of `estimator` over `x`, `y` and `args` using torch as backend.
    
    Parameters
    ----------
    x : torch.Tensor
        Matrix ([samples ...] x features)
    y : torch.Tensor
        Matrix ([samples ...] x features)
    estimator : Callable
        Which estimator function to use?
    n_splits : Union[int, None], optional
        How many splits should be used? (default = None for leave-one-out)
    
    Returns
    -------
    torch.Tensor
        Cross-validated distances.
    """
    
    if n_splits is None:
        n_splits = x.shape[0]
    
    cv_outer = KFold(n_splits = n_splits)
    cv_inner = KFold(n_splits = n_splits)
        
    r = torch.zeros((n_splits, n_splits, *x.shape[0:-1]), dtype = x.dtype, device = x.device)
    
    for f_i, (train_i, test_i) in enumerate(cv_outer.split(x)):
        train_i, test_i = torch.from_numpy(train_i).to(torch.int32).to(x.device), torch.from_numpy(test_i).to(torch.int32).to(x.device)
        
        for f_j, (train_j, test_j) in enumerate(cv_inner.split(x)):
            if f_i <= f_j: continue
            
            train_j, test_j = torch.from_numpy(train_j).to(torch.int32).to(x.device), torch.from_numpy(test_j).to(torch.int32).to(x.device)
            r[f_i,f_j] = r[f_j,f_i] = estimator(x, y, test_i, test_j, *args)

    return r.mean((0, 1, 2))

def _euclidean_numpy(x: np.ndarray, y: np.ndarray, f_i: np.ndarray, f_j: np.ndarray) -> np.ndarray:
    """Computes cross-validated euclidean distances between vectors in x and y. Note that this function is not exported and should not be called directly.
    
    Parameters
    ----------
    x : np.ndarray
        Tensor ([[samples x ]samples x ]features)
    y : np.ndarray
        Tensor ([[samples x ]samples x ]features
    
    Returns
    -------
    np.ndarray
        Cross-validated distance
    """
    
    return np.sum((x[f_i] - y[f_i]) * (x[f_j] - y[f_j]), axis = -1)

def _euclidean_torch(x: torch.Tensor, y: torch.Tensor, f_i: torch.Tensor, f_j: torch.Tensor) -> torch.Tensor:
    """Computes cross-validated euclidean distances between vectors in x and y. Note that this function is not exported and should not be called directly.
    
    Parameters
    ----------
    x : torch.Tensor
        Tensor ([[samples x ]samples x ]features)
    y : torch.Tensor
        Tensor ([[samples x ]samples x ]features)
    
    Returns
    -------
    torch.Tensor
        Cross-validated distance
    """
    
    return torch.sum((x[f_i] - y[f_i]) * (x[f_j] - y[f_j]), -1)

def cv_euclidean(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Computes cross-validated euclidean distances between vectors in x and y.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Tensor ([[samples x ]samples x ]features)
    y : Union[np.ndarray, torch.Tensor]
        Tensor ([[samples x ]samples x ]features)
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Cross-validated distances
    
    Notes
    -----
    Cross-validated euclidean distances are defined as:
    
    .. math::

        d(x, y)^2 = \sum (x_{i} - y_{i})(x_{j} - y_{j})

    where :math:`i` and :math:`j` refer to the indices in the cross-validation folds. Note that this is, therefore, technically a squared measure. For more information, see [1]_.
    
    References
    ----------
    .. [1] Walther, A., Nili, H., Ejaz, N., Alink, A., Kriegeskorte, N., & Diedrichsen, J. (2016). Reliability of dissimilarity measures for multi-voxel pattern analysis. NeuroImage, 137, 188-200. 10.1016/j.neuroimage.2015.12.012
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import cv_euclidean
    >>> x = torch.randn(100, 10)
    >>> y = torch.randn(100, 10)
    >>> d = cv_euclidean(x, y)
    >>> d.shape
    torch.Size([100])
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _cv_torch(x, y, _euclidean_torch)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _cv_numpy(x, y, _euclidean_numpy)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')

def _mahalanobis_numpy(x: np.ndarray, y: np.ndarray, f_i: np.ndarray, f_j: np.ndarray, Σ: np.ndarray) -> np.ndarray:
    """Computes crossnobis distances over x and y using numpy as a backend.
    
    Parameters
    ----------
    x : np.ndarray
        Matrix ([samples ...] x features)
    y : np.ndarray
        Matrix ([samples ...] x features)
    f_i : np.ndarray
        Indices of the first samples
    f_j : np.ndarray
        Indices of the second samples
    Σ : np.ndarray
        Precision matrix
    
    Returns
    -------
    np.ndarray
        Crossnobis distances
    """
    
    return np.matmul(np.matmul((x[f_i] - y[f_i])[...,None,:], Σ), (x[f_j] - y[f_j])[...,None]).squeeze(-1).squeeze(-1)

def _mahalanobis_torch(x: torch.Tensor, y: torch.Tensor, f_i: torch.Tensor, f_j: torch.Tensor, Σ: torch.Tensor) -> torch.Tensor:
    """Computes crossnobis distances over x and y using torch as a backend.
    
    Parameters
    ----------
    x : torch.Tensor
        Matrix ([samples ...] x features)
    y : torch.Tensor
        Matrix ([samples ...] x features)
    f_i : torch.Tensor
        Indices of the first samples
    f_j : torch.Tensor
        Indices of the second samples
    Σ : torch.Tensor
        Precision matrix
    
    Returns
    -------
    torch.Tensor
        Cross-validated distances
    """
    
    return torch.matmul(torch.matmul((x[f_i] - y[f_i])[...,None,:], Σ), (x[f_j] - y[f_j])[...,None]).squeeze(-1).squeeze(-1)

def cv_mahalanobis(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], Σ: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Computes cross-validated mahalanobis distances between x and y. This is sometimes also referred to as the crossnobis distance.

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
        Vector or matrix of distances. Note that this will eliminate the trial dimension, too, during cross-validation.
    
    Notes
    -----
    Crossnobis distance is defined as:
    
    .. math::
        d(x, y)^2 = (x_i - y_i)^T Σ^{-1} (x_j - y_j)
    
    where :math:`x` and :math:`y` are the matrices to compute the distance between, and :math:`Σ` is the covariance matrix. Note that here :math:`i` and :math:`j` refer to folds of a cross-validation. Note also that, principally, this metric is a squared measure. For more information, see [2]_.
    
    References
    ----------
    .. [2] Diedrichsen, J., Provost, S., & Zareamoghaddam, H. (2016). On the distribution of cross-validated Mahalanobis distances. arXiv. 10.48550/arXiv.1607.01371
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import Covariance
    >>> from mvpy.math import cv_mahalanobis
    >>> x, y = torch.normal(0, 1, (100, 50, 60)), torch.normal(0, 1, (100, 50, 60))
    >>> Σ = Covariance().fit(torch.cat((x, y), 0).swapaxes(1, 2)).covariance_
    >>> Σ = torch.linalg.inv(Σ)
    >>> d = cv_mahalanobis(x, y, Σ)
    >>> d.shape
    torch.Size([50])
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    if Σ.shape != (x.shape[-1], x.shape[-1]):
        raise ValueError(f'`Σ` must be of size `(x[-1], x[-1])`, but received x={x.shape} and Σ={Σ.shape}.')
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor) & isinstance(Σ, torch.Tensor):
        return _cv_torch(x, y, _mahalanobis_torch, Σ)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray) & isinstance(Σ, np.ndarray):
        return _cv_numpy(x, y, _mahalanobis_numpy, Σ)
    
    raise ValueError(f'`x`, `y` and `Σ` must be of the same type, but got `{type(x)}`, `{type(y)}` and `{type(Σ)}` instead.')