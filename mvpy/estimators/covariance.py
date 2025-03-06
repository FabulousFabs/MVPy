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
    """Implements a whitening estimator.
    """
    
    def __init__(self):
        """Obtain a whitening estimator class.
        """
        
        super().__init__()
        
        self.whitener_ = None
    
    def fit(self, X: np.ndarray):
        """Fit the whitening estimator.
        
        Parameters
        ----------
        X : np.ndarray
            Matrix/Tensor ([... [samples]] x features x samples)
        """
        
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
        """Whiten the data.
        
        Parameters
        ----------
        X : np.ndarray
            The data to be whitened.
        
        Returns
        -------
        np.ndarray
            The whitened data.
        """
        
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
        
        # reshape
        if len(dims) > 2:
            X_w = X_w.swapaxes(-2, -1)
        
        return X_w

class _Whitener_torch(sklearn.base.BaseEstimator):
    """Implements a whitening estimator.
    """
    
    def __init__(self):
        """Obtain a whitening estimator.
        """
        
        super().__init__()
        
        self.whitener_ = None
    
    def fit(self, X: torch.Tensor):
        """Fit the whitening estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            Matrix/Tensor ([... [samples]] x features x samples)
        """
        
        # make sure covariance has been estimated
        if (hasattr(self, 'covariance_') == False) | (self.covariance_ is None):
            raise AttributeError('Covariance has not been estimated yet.')
        
        # compute eigen-decomposition
        val, vec = torch.linalg.eigh(self.covariance_)
        
        # for stability, treat as only non-negative eigenvalues
        nonzero = val > 0
        val[~nonzero] = 0.0
        
        # take inverse square root
        inv_val = torch.zeros(val.shape, dtype = X.dtype, device = X.device)
        inv_val[nonzero] = 1 / torch.sqrt(val[nonzero])
        
        # compute whitening estimator
        self.whitener_ = vec @ torch.diag(inv_val) @ vec.T
        
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply whitening to the data.
        
        Parameters
        ----------
        X : torch.Tensor
            The data to be whitened.
        
        Returns
        -------
        torch.Tensor
            The whitened data.
        """
        
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
    """Implementation of an empirical covariance estimator.
    """
    
    def __init__(self):
        """Obtain an estimator class.
        """
        
        super().__init__()
        
        self.covariance_ = None
        self.precision_ = None
    
    def fit(self, X: np.ndarray):
        """Fit the estimator.
        
        Parameters
        ----------
        X : np.ndarray
            Training data.
        """
        
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
        
        # compute precision matrix
        self.precision_ = np.linalg.inv(self.covariance_)
        
        # fit whitener class
        super().fit(X)
        
        return self
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the estimator and whiten the data.
        
        Parameters
        ----------
        X : np.ndarray
            Matrix/Tensor ([... [samples]] x features x samples)
        
        Returns
        -------
        np.ndarray
            Whitened matrix/tensor ([... [samples]] x features x samples)
        """
        
        return self.fit(X).transform(X)
    
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        _Empirical_numpy
            The cloned object.
        """
        
        return _Empirical_numpy()

class _Empirical_torch(_Whitener_torch):
    """Implements an empirical, biassed covariance estimator.
    """
    
    def __init__(self):
        """Obtain an empirical estimator.
        """
        
        super().__init__()
        
        self.covariance_ = None
        self.precision_ = None
    
    def fit(self, X: torch.Tensor):
        """Fit the estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            Matrix/Tensor ([... [samples]] x features x samples)
        """
        
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
        
        # compute precision matrix
        self.precision_ = torch.linalg.inv(self.covariance_)
        
        # fit whitener class
        super().fit(X)
        
        return self
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit the estimator and whiten the data.
        
        Parameters
        ----------
        X : torch.Tensor
            Training data
        
        Returns
        -------
        W : torch.Tensor
            Whitened data
        """
        
        return self.fit(X).transform(X)
    
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        _Empirical_torch
            The cloned object.
        """
        
        return _Empirical_torch()

# add Empirical estimator
_ESTIMATORS.append('Empirical')

'''
LedoitWolf estimators
'''

class _LedoitWolf_numpy(_Whitener_numpy):
    """Implements the Ledoit-Wolf estimator.
    """
    
    def __init__(self):
        """Obtain a Ledoit-Wolf estimator.
        """
        
        super().__init__()
        
        self.covariance_ = None
        self.precision_ = None
        self.shrinkage_ = None
    
    def fit(self, X: np.ndarray):
        """Fit the estimator.
        
        Parameters
        ----------
        X : np.ndarray
            Training data
        """
        
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
        
        # compute precision matrix
        self.precision_ = np.linalg.inv(self.covariance_)
        
        # fit whitener class
        super().fit(X)
        
        return self
    
    def fit_transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        """Fit the estimator and whiten the data.
        
        Parameters
        ----------
        X : np.ndarray
            Matrix/Tensor ([... [samples]] x features x samples)
        
        Returns
        -------
        np.ndarray
            Whitened matrix/tensor ([... [samples]] x features x samples)
        """
        
        return self.fit(X).transform(X)
    
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        _LedoitWolf_numpy
            The cloned object.
        """
        
        return _LedoitWolf_numpy()

class _LedoitWolf_torch(_Whitener_torch):
    """Implementation of the Ledoit-Wolf shrinkage estimator.
    """
    
    def __init__(self):
        """Obtain the Ledoit-Wolf shrinkage estimator.
        """
        
        super().__init__()
        
        self.covariance_ = None
        self.precision_ = None
        self.shrinkage_ = None
    
    def fit(self, X: torch.Tensor):
        """Fit the estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            Matrix/Tensor ([... [samples]] x features x samples)
        """
        
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
        
        # compute precision matrix
        self.precision_ = torch.linalg.inv(self.covariance_)

        # fit whitener class
        super().fit(X)
        
        return self
    
    def fit_transform(self, X: torch.Tensor, *args: Any):
        """Fit the estimator and whiten the data.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data.

        Returns
        -------
        W : torch.Tensor
            Whitened data.
        """
        
        return self.fit(X).transform(X)
    
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        _LedoitWolf_torch
            The cloned object.
        """
        
        return _LedoitWolf_torch()

# add LedoitWolf estimator
_ESTIMATORS.append('LedoitWolf')

'''
Covariance estimator
'''

class Covariance(sklearn.base.BaseEstimator):
    """Class for computing covariance, precision and whitening matrices. Note that calling a transform from this clas will whiten the data.
    
    Parameters
    ----------
    method : str, default = 'LedoitWolf'
        Which method should be applied for estimation of covariance? (default = LedoitWolf, available = [Empirical, LedoitWolf])
    
    Attributes
    ----------
    covariance_ : Union[np.ndarray, torch.Tensor]
        Covariance matrix
    precision_ : Union[np.ndarray, torch.Tensor]
        Precision matrix (inverse of covariance matrix)
    whitener_ : Union[np.ndarray, torch.Tensor]
        Whitening matrix
    shrinkage_ : float, optional
        Shrinkage parameter, if used by method.
    
    Notes
    -----
    This class assumes features to be the second to last dimension of the data, unless there are only two dimensions (in which case it is assumed to be the last dimension).
    
    Currently, we support the following methods:
    
    - Empirical:
        This method simply computes the biassed empirical covariance matrix.

    - LedoitWolf:
        This method computes the Ledoit-Wolf shrinkage estimator as detailed in [1]_.
    
    References
    ----------
    .. [1] Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88, 365-411. 10.1016/S0047-259X(03)00096-4
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import Covariance
    >>> X = torch.normal(0, 1, (100, 10, 100))
    >>> cov = Covariance().fit(X)
    >>> cov.covariance_.shape
    torch.Size([10, 10])
    """
    
    def __init__(self, method: str = 'LedoitWolf'):
        """Create a new Covariance instance.
        
        Parameters
        ----------
        method : str, default = 'LedoitWolf'
            Which method should be applied for estimation of covariance? (default = LedoitWolf, available = [Empirical, LedoitWolf])
        """
        
        if method not in _ESTIMATORS:
            raise ValueError(f'Unknown covariance estimation method {method}. Available methods: {_ESTIMATORS}.')

        self.method = method
    
    def _get_estimator(self, X: Union[np.ndarray, torch.Tensor]) -> sklearn.base.BaseEstimator:
        """Obtain an estimator based on the data type.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Data to fit the estimator on.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            Estimator to use.
        """
        
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
        """Fit the covariance estimator.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Data to fit the estimator on.
        *args : Any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        self : Covariance
            Fitted covariance estimator.
        """
        
        return self._get_estimator(X)().fit(X)
    
    def transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Whiten data using the fitted covariance estimator.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Data to transform.
        *args : Any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        W : Union[np.ndarray, torch.Tensor]
            Whitened data.
        """
        return self._get_estimator(X)().fit_transform(X)

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Fit the covariance estimator and whiten the data.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Data to fit the estimator on and transform.
        *args : Any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        W : Union[np.ndarray, torch.Tensor]
            Whitened data.
        """
        
        return self._get_estimator(X)().fit_transform(X)
    
    def to_torch(self):
        """Create the torch estimator. Note that this function cannot be used for conversion.
        
        Returns
        -------
        Covariance
            The torch estimator.
        """
        
        return self._get_estimator(torch.tensor([1]))()
    
    def to_numpy(self):
        """Create the numpy estimator. Note that this function cannot be used for conversion.

        Returns
        -------
        Covariance
            The numpy estimator.
        """

        return self._get_estimator(np.array([1]))()
    
    def clone(self):
        """Obtain a clone of this class.
        
        Returns
        -------
        Covariance
            The cloned object.
        """
        
        return Covariance(method = self.method)