'''
A collection of estimators for covariance estimation and
pre-whitening of data.
'''

import sklearn
import numpy as np
import torch

from typing import Union, Any

'''
Track registered estimators 
'''
_ESTIMATORS = []

'''
Whitening estimators
'''

class _Whitener_numpy(sklearn.base.BaseEstimator):
    '''
    Class to compute whitening estimator from covariance matrix.
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        
        super().__init__()
        
        self.whitener_ = None
    
    def fit(self, X: np.ndarray):
        '''
        Fit the whitening estimator given covariance estimate.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        '''
        
        # make sure covariance has been estimated
        if (hasattr(self, 'covariance_') == False) | (self.covariance_ is None):
            raise AttributeError('Covariance has not been estimated yet.')
        
        # compute eigen-decomposition
        val, vec = np.linalg.eigh(self.covariance_)
        
        # for stability, treat as only non-negative eigenvalues
        nonzero = val > 0
        val[~nonzero] = 0.0
        
        # take inverse square root
        inv_val = np.zeros(val.shape)
        inv_val[nonzero] = 1 / np.sqrt(val[nonzero])
        
        # compute whitening estimator
        self.whitener_ = vec @ np.diag(inv_val) @ vec.T
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        '''
        Whiten the data in `X`.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        
        OUTPUTS:
            ^X  -   Whitened matrix/tensor ([... [samples]] x features x samples)
        '''
        
        # make sure covariance has been estimated
        if (hasattr(self, 'covariance_') == False) | (self.covariance_ is None):
            raise AttributeError('Covariance has not been estimated yet.')
        
        # make sure types match
        if isinstance(self.covariance_, np.ndarray) & (isinstance(X, np.ndarray) == False):
            raise ValueError(f'`X` and `covariance_` must be in same backend, but got `numpy` and type `{type(X)}`.')
        
        # check dims
        if len(X.shape) < 2:
            raise ValueError(f'`X` must be at least two-dimensional for whitening, but got shapee {X.shape}.')

        # reshape
        dims = X.shape
        
        if len(dims) > 2:
            X_h = X.swapaxes(-2, -1)
            dims = X_h.shape
            X_h = X_h.reshape(-1, X.shape[-2])
        else: 
            X_h = X.copy()
        
        # whiten
        X_w = X_h.dot(self.whitener_).reshape(dims)
        print(X_w.shape)
        
        # reshape
        if len(dims) > 2:
            X_w = X_w.swapaxes(-2, -1)
        
        return X_w

class _Whitener_torch(sklearn.base.BaseEstimator):
    '''
    Class to compute whitening estimator from covariance matrix.
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        
        super().__init__()
        
        self.whitener_ = None
    
    def fit(self, X: torch.Tensor):
        '''
        Fit the whitening estimator given covariance estimate.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        '''
        
        # make sure covariance has been estimated
        if (hasattr(self, 'covariance_') == False) | (self.covariance_ is None):
            raise AttributeError('Covariance has not been estimated yet.')
        
        # compute eigen-decomposition
        val, vec = torch.linalg.eigh(self.covariance_)
        
        # for stability, treat as only non-negative eigenvalues
        nonzero = val > 0
        val[~nonzero] = 0.0
        
        # take inverse square root
        inv_val = torch.zeros(val.shape)
        inv_val[nonzero] = 1 / torch.sqrt(val[nonzero])
        
        # compute whitening estimator
        self.whitener_ = vec @ torch.diag(inv_val) @ vec.T
        
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Whiten the data in `X`.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        
        OUTPUTS:
            ^X  -   Whitened matrix/tensor ([... [samples]] x features x samples)
        '''
        
        # make sure covariance has been estimated
        if (hasattr(self, 'covariance_') == False) | (self.covariance_ is None):
            raise AttributeError('Covariance has not been estimated yet.')
        
        # make sure types match
        if isinstance(self.covariance_, torch.Tensor) & (isinstance(X, torch.Tensor) == False):
            raise ValueError(f'`X` and `covariance_` must be in same backend, but got `numpy` and type `{type(X)}`.')
        
        # check dims
        if len(X.shape) < 2:
            raise ValueError(f'`X` must be at least two-dimensional for whitening, but got shapee {X.shape}.')
        
        # reshape
        dims = X.shape
        
        if len(dims) > 2:
            X_h = X.swapaxes(-2, -1)
            dims = X_h.shape
            X_h = X_h.reshape(-1, X.shape[-2])
        else:
            X_h = X.clone()
        
        # whiten
        X_w = (X_h @ self.whitener_).reshape(dims)
        
        # reshape
        if len(dims) > 2:
            X_w = X_w.swapaxes(-2, -1)
        
        return X_w

'''
Empirical estimators
'''

class _Empirical_numpy(_Whitener_numpy):
    '''
    Implements an empirical covariance estimator.
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        
        super().__init__()
        
        self.covariance_ = None
    
    def fit(self, X: np.ndarray):
        '''
        Fit the empirical estimator.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        '''
        
        # check dims
        if len(X.shape) < 2:
            raise ValueError(f'`X` must be at least two-dimensional for covariance estimation, but got shapee {X.shape}.')
        
        # reshape
        if len(X.shape) > 2: X_h = X.swapaxes(-2, -1).reshape((-1, X.shape[-2]))
        else: X_h = X.copy()
        
        # get dims
        N, F = X_h.shape
        
        # centre data
        X_h = X_h - X_h.mean(axis = 0)
        
        # compute sample covariance
        self.covariance_ = np.dot(X_h.T, X_h) / N
        
        # fit whitener class
        super().fit(X)
        
        return self
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        '''
        Fit the estimator and whiten data.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        
        OUTPUTS:
            ^X  -   Whitened matrix/tensor ([... [samples]] x features x samples)
        '''
        
        return self.fit(X).transform(X)

class _Empirical_torch(_Whitener_torch):
    '''
    Implements an empirical covariance estimator.
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        
        super().__init__()
        
        self.covariance_ = None
    
    def fit(self, X: torch.Tensor):
        '''
        Fit the empirical estimator.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        '''
        
        # check dims
        if len(X.shape) < 2:
            raise ValueError(f'`X` must be at least two-dimensional for covariance estimation, but got shapee {X.shape}.')
        
        # reshape
        if len(X.shape) > 2: X_h = X.swapaxes(-2, -1).reshape((-1, X.shape[-2]))
        else: X_h = X.clone()
        
        # get dims
        N, F = X_h.shape
        
        # centre data
        X_h = X_h - X_h.mean(0)
        
        # compute sample covariance
        self.covariance_ = torch.mm(X_h.T, X_h) / N
        
        # fit whitener class
        super().fit(X)
        
        return self
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        '''
        Fit the estimator and whiten data.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        
        OUTPUTS:
            ^X  -   Whitened matrix/tensor ([... [samples]] x features x samples)
        '''
        
        return self.fit(X).transform(X)

# add Empirical estimator
_ESTIMATORS.append('Empirical')

'''
LedoitWolf estimators
'''

class _LedoitWolf_numpy(_Whitener_numpy):
    '''
    Implements the Ledoit-Wolf shrinkage estimation as detailed in:

        Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88, 365-411. 10.1016/S0047-259X(03)00096-4
    
    This class is based on the numpy backend.
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        
        super().__init__()
        
        self.covariance_ = None
        self.shrinkage_ = None
    
    def fit(self, X: np.ndarray):
        '''
        Fit the LedoitWolf estimator.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        '''
        
        # check dims
        if len(X.shape) < 2:
            raise ValueError(f'`X` must be at least two-dimensional for covariance estimation, but got shapee {X.shape}.')
        
        # reshape
        if len(X.shape) > 2: X_h = X.swapaxes(-2, -1).reshape((-1, X.shape[-2]))
        else: X_h = X.copy()
        
        # get dims
        N, F = X_h.shape
        
        # centre data
        X_h = X_h - X_h.mean(axis = 0)
        
        # compute sample covariance
        S = np.dot(X_h.T, X_h) / N
        
        # compute target matrix from F = μI
        μ = np.trace(S) / N
        F = μ * np.eye(F)
        
        # compute ϕ
        X_h2 = X_h ** 2
        ϕ_m = np.dot(X_h2.T, X_h2) / N - S ** 2
        ϕ = np.sum(ϕ_m)
        
        # compute γ
        γ = np.linalg.norm(S - F, 'fro') ** 2
        
        # compute κ
        κ = ϕ / γ if γ != 0 else 0.0
        
        # compute shrinkage
        self.shrinkage_ = np.clip(κ / N, 0, 1)
        
        # compute covariance estimate
        self.covariance_ = self.shrinkage_ * F + (1 - self.shrinkage_) * S
        
        # fit whitener class
        super().fit(X)
        
        return self
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        '''
        Fit the estimator and whiten data.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        
        OUTPUTS:
            ^X  -   Whitened matrix/tensor ([... [samples]] x features x samples)
        '''
        
        return self.fit(X).transform(X)

class _LedoitWolf_torch(_Whitener_torch):
    '''
    Implements the Ledoit-Wolf shrinkage estimation as detailed in:

        Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88, 365-411. 10.1016/S0047-259X(03)00096-4
    
    This class is based on the torch backend.
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        
        super().__init__()
        
        self.covariance_ = None
        self.shrinkage_ = None
    
    def fit(self, X: torch.Tensor):
        '''
        Fit the LedoitWolf estimator.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        '''
        
        # check dims
        if X.ndim < 2:
            raise ValueError(f"`X` must be at least two-dimensional for covariance estimation, but got shape {X.shape}.")

        # reshape data if more than two dimensions
        if X.ndim > 2: X_h = X.transpose(-2, -1).reshape(-1, X.shape[-2])
        else: X_h = X.clone()

        # get dims
        N, F = X_h.shape

        # centre the data
        X_h = X_h - X_h.mean(0)

        # compute sample covariance
        S = torch.mm(X_h.T, X_h) / N

        # Compute target matrix from F = μI
        μ = torch.trace(S) / N
        F = μ * torch.eye(F, device=X.device, dtype=X.dtype)

        # compute φ
        X_h2 = X_h ** 2
        φ_m = torch.mm(X_h2.T, X_h2) / N - S ** 2
        φ = φ_m.sum()

        # compute γ
        γ = torch.norm(S - F, p = 'fro') ** 2

        # compute κ
        κ = φ / γ if γ != 0 else 0.0
        
        # compute shrinkage
        self.shrinkage_ = torch.clamp(κ / N, 0, 1)

        # compute covariance estimate
        self.covariance_ = self.shrinkage_ * F + (1 - self.shrinkage_) * S

        # fit whitener class
        super().fit(X)
        
        return self
    
    def fit_transform(self, X: torch.Tensor):
        '''
        Fit the estimator and whiten data.
        
        INPUTS:
            X   -   Matrix/Tensor ([... [samples]] x features x samples)
        
        OUTPUTS:
            ^X  -   Whitened matrix/tensor ([... [samples]] x features x samples)
        '''
        
        return self.fit(X).transform(X)

# add LedoitWolf estimator
_ESTIMATORS.append('LedoitWolf')

'''
Covariance estimator
'''

class Covariance(sklearn.base.BaseEstimator):
    '''
    Class for computing covariance estimates.
    '''
    
    def __init__(self, method: str = 'LedoitWolf'):
        '''
        Return the covariance class.
        
        INPUTS:
            method  -   Which method should be applied? (default = LedoitWolf, available = [Empirical, LedoitWolf])
        '''
        
        if method not in _ESTIMATORS:
            raise ValueError(f'Unknown covariance estimation method {method}. Available methods: {_ESTIMATORS}.')

        self.method = method
    
    def _get_estimator(self, X: Union[np.ndarray, torch.Tensor]) -> sklearn.base.BaseEstimator:
        '''
        Obtain an appropriate estimator.
        
        INPUTS:
            X   -   Matrix/Tensor
        
        OUTPUTS:
            estimator   -   Appropriate class to construct
        '''
        
        if isinstance(X, torch.Tensor) & (self.method == 'LedoitWolf'):
            return _LedoitWolf_torch
        elif isinstance(X, np.ndarray) & (self.method == 'LedoitWolf'):
            return _LedoitWolf_numpy
        elif isinstance(X, torch.Tensor) & (self.method == 'Empirical'):
            return _Empirical_torch
        elif isinstance(X, np.ndarray) & (self.method == 'Empirical'):
            return _Empirical_numpy
        
        raise ValueError(f'Got an unexpected combination of method=`{self.method}` and type=`{type(X)}`.') 
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], *args: Any):
        '''
        Fit the selected covariance estimator over `X`.
        
        INPUTS:
            X   -   Matrix/Tensor
        '''
        
        return self._get_estimator(X)().fit(X)
    
    def transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        '''
        Transform from estimator.

        INPUTS:
            x   -   Matrix/Tensor

        OUTPUTS:
            ^x  -   Whitened data
        '''

        return self._get_estimator(X)().fit_transform(X)

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        '''
        Fit and transform from estimator.
        
        INPUTS:
            x   -   Matrix/Tensor
        
        OUTPUTS:
            ^x  -   Whitened data
        '''
        
        return self._get_estimator(X)().fit_transform(X)