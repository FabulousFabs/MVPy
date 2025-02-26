'''
Wrapper functions to compute cross-validated metrics from
the metrics already in this package.

For more information, see:

    Diedrichsen, J., Provost, S., & Zareamoghaddam, H. (2016). On the distribution of cross-validated Mahalanobis distances. arXiv. 10.48550/arXiv.1607.01371
    Schütt, H.H., Kipnis, A.D., Diedrichsen, J., & Kriegeskorte, N. (2023). Statistical inference on representational geometries. eLife, 12, e82566. 10.7554/eLife.82566
'''

import numpy as np
import torch

from sklearn.model_selection import KFold

from typing import Union, Any, Callable

def _cv_numpy(x: np.ndarray, y: np.ndarray, estimator: Callable, *args: Any, n_splits: Union[int, None] = None) -> np.ndarray:
    '''
    Computes cross-validation steps of `estimator` over `x`, `y` and `args`
    using numpy backend.
    
    INPUTS:
        x           -   Vector/matrix/tensor
        y           -   Vector/matrix/tensor
        estimator   -   Estimator function
        *args       -   Additional arguments for estimator
        n_splits    -   How many splits should be used? (default = None for leave-one-out)
    
    OUTPUTS:
        s           -   Cross-validated distances
    '''
    
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
    '''
    Computes cross-validation steps of `estimator` over `x`, `y` and `args`
    using torch backend.
    
    INPUTS:
        x           -   Vector/matrix/tensor
        y           -   Vector/matrix/tensor
        estimator   -   Estimator function
        *args       -   Additional arguments for estimator
        n_splits    -   How many splits should be used? (default = None for leave-one-out)
    
    OUTPUTS:
        s           -   Cross-validated distances
    '''
    
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
    '''
    Computes cross-validated euclidean distance between vectors in x and y
    using numpy backend.
    
    INPUTS:
        x   -   Tensor ([[samples x ]samples x ]features)
        y   -   Tensor ([[samples x ]samples x ]features)

    OUTPUTS:
        s   -   Cross-validated distance
    '''
    
    return np.sum((x[f_i] - y[f_i]) * (x[f_j] - y[f_j]), axis = -1)

def _euclidean_torch(x: torch.Tensor, y: torch.Tensor, f_i: torch.Tensor, f_j: torch.Tensor) -> torch.Tensor:
    '''
    Computes cross-validated euclidean distance between vectors in x and y
    using torch backend.
    
    INPUTS:
        x   -   Tensor ([[samples x ]samples x ]features)
        y   -   Tensor ([[samples x ]samples x ]features)

    OUTPUTS:
        s   -   Cross-validated distance
    '''
    
    return torch.sum((x[f_i] - y[f_i]) * (x[f_j] - y[f_j]), -1)

def cv_euclidean(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    '''
    Computes cross-validated euclidean distance between vectors in x and y.
    
    INPUTS:
        x   -   Tensor ([[samples x ]samples x ]features)
        y   -   Tensor ([[samples x ]samples x ]features)

    OUTPUTS:
        s   -   Cross-validated distance
    '''
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _cv_torch(x, y, _euclidean_torch)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _cv_numpy(x, y, _euclidean_numpy)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')

def _mahalanobis_numpy(x: np.ndarray, y: np.ndarray, f_i: np.ndarray, f_j: np.ndarray, Σ: np.ndarray) -> np.ndarray:
    '''
    Computes crossnobis distance between vectors in x and y using
    numpy as a backend.
    
    INPUTS:
        x   -   Tensor ([[samples x ]samples x ]features)
        y   -   Tensor ([[samples x ]samples x ]features)
        Σ   -   Precision matrix (features x features)
    
    OUTPUTS:
        s   -   Cross-validated distance
    '''
    
    return np.matmul(np.matmul((x[f_i] - y[f_i])[...,None,:], Σ), (x[f_j] - y[f_j])[...,None]).squeeze(-1).squeeze(-1)

def _mahalanobis_torch(x: torch.Tensor, y: torch.Tensor, f_i: torch.Tensor, f_j: torch.Tensor, Σ: torch.Tensor) -> torch.Tensor:
    '''
    Computes crossnobis distance between vectors in x and y using
    torch as a backend.
    
    INPUTS:
        x   -   Tensor ([[samples x ]samples x ]features)
        y   -   Tensor ([[samples x ]samples x ]features)
        Σ   -   Precision matrix (features x features)
    
    OUTPUTS:
        s   -   Cross-validated distance
    '''
    
    return torch.matmul(torch.matmul((x[f_i] - y[f_i])[...,None,:], Σ), (x[f_j] - y[f_j])[...,None]).squeeze(-1).squeeze(-1)

def cv_mahalanobis(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], Σ: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    '''
    Computes cross-validated mahalanobis distance (also called 
    crossnobis distance) between `x` and `y` among final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
        Σ   -   Inverse covariance matrix
    
    OUTPUTS:
        s   -   Cross-validated distances
    '''
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    if Σ.shape != (x.shape[-1], x.shape[-1]):
        raise ValueError(f'`Σ` must be of size `(x[-1], x[-1])`, but received x={x.shape} and Σ={Σ.shape}.')
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor) & isinstance(Σ, torch.Tensor):
        return _cv_torch(x, y, _mahalanobis_torch, Σ)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray) & isinstance(Σ, np.ndarray):
        return _cv_numpy(x, y, _mahalanobis_numpy, Σ)
    
    raise ValueError(f'`x`, `y` and `Σ` must be of the same type, but got `{type(x)}`, `{type(y)}` and `{type(Σ)}` instead.')