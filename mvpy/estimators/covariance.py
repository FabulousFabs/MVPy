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
_ESTIMATORS = {}

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
    
    def __init__(self, s_min: Union[float, None] = None, s_max: Union[float, None] = None):
        """Obtain an estimator class.
        """
        
        super().__init__()
        
        self.covariance_ = None
        self.precision_ = None
        
        self.s_min = s_min
        self.s_max = s_max
    
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
        if len(X.shape) > 2: 
            # set sample selection
            s = 0 if self.s_min is None else self.s_min
            e = X.shape[-1] if self.s_max is None else self.s_max
            
            # reshape
            X_h = X[...,s:e].swapaxes(-2, -1).reshape((-1, X.shape[-2]))
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
        
        return _Empirical_numpy(s_min = self.s_min, s_max = self.s_max)

class _Empirical_torch(_Whitener_torch):
    """Implements an empirical, biassed covariance estimator.
    """
    
    def __init__(self, s_min: Union[float, None] = None, s_max: Union[float, None] = None):
        """Obtain an empirical estimator.
        """
        
        super().__init__()
        
        self.covariance_ = None
        self.precision_ = None
        
        self.s_min = s_min
        self.s_max = s_max
    
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
        if len(X.shape) > 2: 
            # set sample selection
            s = 0 if self.s_min is None else self.s_min
            e = X.shape[-1] if self.s_max is None else self.s_max
            
            # reshape
            X_h = X[...,s:e].swapaxes(-2, -1).reshape((-1, X.shape[-2]))
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
        
        return _Empirical_torch(s_min = self.s_min, s_max = self.s_max)

# add empirical estimator
_ESTIMATORS['empirical'] = {
    'numpy': _Empirical_numpy,
    'torch': _Empirical_torch
}

'''
LedoitWolf estimators
'''

class _LedoitWolf_numpy(_Whitener_numpy):
    """Implements the Ledoit-Wolf estimator.
    """
    
    def __init__(self, s_min: Union[float, None] = None, s_max: Union[float, None] = None):
        """Obtain a Ledoit-Wolf estimator.
        """
        
        super().__init__()
        
        self.covariance_ = None
        self.precision_ = None
        self.shrinkage_ = None
        
        self.s_min = s_min
        self.s_max = s_max
    
    def fit(self, X: np.ndarray) -> "_LedoitWolf_numpy":
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
        if len(X.shape) > 2: 
            # set sample selection
            s = 0 if self.s_min is None else self.s_min
            e = X.shape[-1] if self.s_max is None else self.s_max
            
            # reshape
            X_h = X[...,s:e].swapaxes(-2, -1).reshape((-1, X.shape[-2]))
        else: X_h = X.copy()
        
        # get dims
        N, F = X_h.shape
        
        # centre data
        X_h = X_h - X_h.mean(axis = 0)
        
        # compute sample covariance
        S = np.dot(X_h.T, X_h) / N
        
        # compute target matrix from F = μI
        μ = np.trace(S) / F
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
    
    def clone(self) -> "_LedoitWolf_numpy":
        """Clone this class.
        
        Returns
        -------
        covariance : _LedoitWolf_numpy
            The cloned object.
        """
        
        return _LedoitWolf_numpy(
            s_min = self.s_min, 
            s_max = self.s_max
        )

class _LedoitWolf_torch(_Whitener_torch):
    """Implementation of the Ledoit-Wolf shrinkage estimator.
    """
    
    def __init__(self, s_min: Union[float, None] = None, s_max: Union[float, None] = None):
        """Obtain the Ledoit-Wolf shrinkage estimator.
        """
        
        super().__init__()
        
        self.covariance_ = None
        self.precision_ = None
        self.shrinkage_ = None
        
        self.s_min = s_min
        self.s_max = s_max
    
    def fit(self, X: torch.Tensor) -> "_LedoitWolf_torch":
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
        if X.ndim > 2: 
            # set sample selection
            s = 0 if self.s_min is None else self.s_min
            e = X.shape[-1] if self.s_max is None else self.s_max
            
            # reshape
            X_h = X[...,s:e].transpose(-2, -1).reshape(-1, X.shape[-2])
        else: X_h = X.clone()

        # get dims
        N, F = X_h.shape

        # centre the data
        X_h = X_h - X_h.mean(0)

        # compute sample covariance
        S = torch.mm(X_h.T, X_h) / N

        # compute target matrix from F = μI
        μ = torch.trace(S) / F
        F = μ * torch.eye(F, device=X.device, dtype=X.dtype)

        # compute φ
        X_h2 = X_h ** 2
        φ_m = torch.mm(X_h2.T, X_h2) / N - S ** 2
        φ = φ_m.sum()

        # compute γ
        γ = torch.norm(S - F, p = 'fro') ** 2

        # compute κ
        κ = φ / γ if γ != 0 else torch.tensor([0.0], dtype = X.dtype, device = X.device)
        
        # compute shrinkage
        self.shrinkage_ = torch.clamp(κ / N, 0, 1)

        # compute covariance estimate
        self.covariance_ = self.shrinkage_ * F + (1 - self.shrinkage_) * S
        
        # compute precision matrix
        self.precision_ = torch.linalg.inv(self.covariance_)

        # fit whitener class
        super().fit(X)
        
        return self
    
    def fit_transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
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
    
    def clone(self) -> "_LedoitWolf_torch":
        """Clone this class.
        
        Returns
        -------
        covariance : _LedoitWolf_torch
            The cloned object.
        """
        
        return _LedoitWolf_torch(
            s_min = self.s_min, 
            s_max = self.s_max
        )

# add LedoitWolf estimator
_ESTIMATORS['ledoitwolf'] = {
    'numpy': _LedoitWolf_numpy,
    'torch': _LedoitWolf_torch
}

'''
OAS estimators
'''

class _OAS_numpy(_Whitener_numpy):
    """Implements the oracle approximating shrinkage estimator.
    """
    
    def __init__(self, s_min: Union[float, None] = None, s_max: Union[float, None] = None):
        """Obtain an OAS estimator.
        """
        
        super().__init__()
        
        self.covariance_ = None
        self.precision_ = None
        self.shrinkage_ = None
        
        self.s_min = s_min
        self.s_max = s_max
    
    def fit(self, X: np.ndarray) -> "_OAS_numpy":
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
        if len(X.shape) > 2: 
            # set sample selection
            s = 0 if self.s_min is None else self.s_min
            e = X.shape[-1] if self.s_max is None else self.s_max
            
            # reshape
            X_h = X[...,s:e].swapaxes(-2, -1).reshape((-1, X.shape[-2]))
        else: X_h = X.copy()
        
        # get dims
        N, F = X_h.shape
        
        # centre data
        X_h = X_h - X_h.mean(axis = 0)
        
        # compute sample covariance
        S = np.dot(X_h.T, X_h) / N
        
        # compute target matrix from T = μI
        μ = np.trace(S) / F
        T = μ * np.eye(F)

        # compute traces
        tr_S = np.trace(S)
        tr_S2 = (S * S).sum()

        # compute OAS shrinkage
        n = (1 - 2.0 / F) * tr_S2 + tr_S**2
        d = (N + 1 - 2.0 / F) * (tr_S2 - (tr_S**2) / F)
        eps = np.finfo(S.dtype).eps
        self.shrinkage_ = np.clip(n / np.clip(d, a_min = eps, a_max = None), 0, 1)

        # compute covariance estimate
        self.covariance_ = (1 - self.shrinkage_) * S + self.shrinkage_ * T

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
    
    def clone(self) -> "_OAS_numpy":
        """Clone this class.
        
        Returns
        -------
        covariance : _OAS_numpy
            The cloned object.
        """
        
        return _OAS_numpy(
            s_min = self.s_min, 
            s_max = self.s_max
        )

class _OAS_torch(_Whitener_torch):
    """Implementation of the oracle approximating shrinkage estimator.
    """
    
    def __init__(self, s_min: Union[float, None] = None, s_max: Union[float, None] = None):
        """Obtain the Ledoit-Wolf shrinkage estimator.
        """
        
        super().__init__()
        
        self.covariance_ = None
        self.precision_ = None
        self.shrinkage_ = None
        
        self.s_min = s_min
        self.s_max = s_max
    
    def fit(self, X: torch.Tensor) -> "_OAS_torch":
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
        if X.ndim > 2: 
            # set sample selection
            s = 0 if self.s_min is None else self.s_min
            e = X.shape[-1] if self.s_max is None else self.s_max
            
            # reshape
            X_h = X[...,s:e].transpose(-2, -1).reshape(-1, X.shape[-2])
        else: X_h = X.clone()

        # get dims
        N, F = X_h.shape

        # centre the data
        X_h = X_h - X_h.mean(0)

        # compute sample covariance
        S = torch.mm(X_h.T, X_h) / N

        # compute target matrix from T = μI
        μ = torch.trace(S) / F
        T = μ * torch.eye(F, device = X_h.device, dtype = X_h.dtype)

        # compute traces
        tr_S = torch.trace(S)
        tr_S2 = (S * S).sum()

        # compute OAS shrinkage
        n = (1 - 2.0 / F) * tr_S2 + tr_S**2
        d = (N + 1 - 2.0 / F) * (tr_S2 - (tr_S**2) / F)
        eps = torch.finfo(S.dtype).eps
        self.shrinkage_ = torch.clamp(n / torch.clamp(d, min = eps), 0, 1)

        # compute covariance estimate
        self.covariance_ = (1 - self.shrinkage_) * S + self.shrinkage_ * T

        # compute precision matrix
        self.precision_ = torch.linalg.inv(self.covariance_)

        # fit whitener class
        super().fit(X)
        
        return self
    
    def fit_transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
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
    
    def clone(self) -> "_OAS_torch":
        """Clone this class.
        
        Returns
        -------
        covariance : _OAS_torch
            The cloned object.
        """
        
        return _OAS_torch(
            s_min = self.s_min, 
            s_max = self.s_max
        )

# add OAS estimator
_ESTIMATORS['oas'] = {
    'numpy': _OAS_numpy,
    'torch': _OAS_torch
}

'''
Covariance estimator
'''

class Covariance(sklearn.base.BaseEstimator):
    """Implements covariance and precision estimation as well as whitening of data.

    For covariance estimation, three methods are currently available through
    :py:attr:`~mvpy.estimators.Covariance.method`:
    
    1. ``empirical``
        This computes the empirical (sample) covariance matrix:
        
        .. math::

            \\Sigma = \\mathbb{E}\\left[(X - \\mathbb{E}[X])(X^T - \\mathbb{E}[X^T])\\right]
        
        This is computationally efficient, but produces estimates of the
        covariance :math:`\\Sigma` that may often be unfavourable: Given
        small datasets or noisy measurements, :math:`\\Sigma` may be ill-
        conditioned and not positive-definite with eigenvalues that tend
        to be systematically pushed towards the tails. In practice, this
        can make inversion challenging and hurts out-of-sample generalisation.
    
    2. ``ledoitwolf``
        This computes the LedoitWolf shrinkage estimator:
        
        .. math::
        
            \\hat\\Sigma = (1 - \\hat{\\delta})\\Sigma + \\hat\\delta T
        
        where :math:`\\hat{\\delta}\\in[0, 1]` is the data-driven shrinkage 
        intensity that minimises the the expected Frobenius-norm risk:
        
        .. math::
            \\hat\\delta = \\min\\left\\{1, \\max\\left\\{0, \\frac{\\hat\\pi}{\\hat\\rho}\\right\\}\\right\\},\\qquad
            \\hat\\rho = \\lvert\\lvert\\Sigma - T\\rvert\\rvert_F^2,\\qquad
            \\hat\\pi = \\frac{1}{n}\\sum_{k=1}^{n}\\lvert\\lvert x_k x_k^T - \\Sigma\\rvert\\rvert_F^2

        and where:
        
        .. math::

            T = \\mu I_p,\\qquad \\mu = \\frac{1}{p}\\textrm{tr}(\\Sigma)
        
        This produces estimates that are well-conditioned and positive-definite. 
        For more information on this procedure, please see [1]_.
    
    3. ``oas``
        This computes the oracle approximating shrinkage estimator:
        
        .. math::

            \\hat\\Sigma = (1 - \\hat{\\delta})\\Sigma + \\hat\\delta T
        
        where :math:`\\hat{\\delta}\\in[0, 1]` is the data-driven shrinkage:
        
        .. math::
            \\hat\\delta = \\frac{(1 - \\frac{2}{p}) \\textrm{tr}(\\Sigma^2) + \\textrm{tr}(\\Sigma)^2}{(n + 1 - \\frac{2}{p})\\left(\\textrm{tr}(\\Sigma^2) - \\frac{\\textrm{tr}(\\Sigma)^2}{p}\\right)},\\qquad
            T = \\mu I_p,\\qquad \\mu = \\frac{1}{p}\\textrm{tr}(\\Sigma)
        
        Like ``ledoitwolf``, this procedure produces estimates that are
        well-conditioned and positive-definite. Contrary to ``ledoitwolf``,
        shrinkage tends to be more aggressive in this procedure. For more 
        information, please see [2]_.
    
    When calling transform on this class, data will automatically be
    whitened based on the estimated covariance matrix. The whitening
    matrix is computed from the eigendecomposition as follows:
    
    .. math::
        \\Sigma = Q\\Lambda Q^T,\\qquad 
        \\Lambda = \\textrm{diag}(\\lambda_1, ..., \\lambda_p) \geq 0,\\qquad
        W = Q\\Lambda^{-\\frac{1}{2}}Q^T
    
    For more information on whitening, refer to [3]_.
    
    Parameters
    ----------
    method : {'empirical', 'ledoitwolf', 'oas'}, default = 'ledoitwolf'
        Which method should be applied for estimation of covariance?
    s_min : float, default = None
        What's the minimum sample we should consider in the time dimension?
    s_max : float, default = None
        What's the maximum sample we should consider in the time dimension?
    
    Attributes
    ----------
    covariance_ : np.ndarray | torch.Tensor
        Covariance matrix
    precision_ : np.ndarray | torch.Tensor
        Precision matrix (inverse of covariance matrix)
    whitener_ : np.ndarray | torch.Tensor
        Whitening matrix
    shrinkage_ : float, default=None
        Shrinkage parameter, if used by method.
    
    Notes
    -----
    This class assumes features to be the second to last dimension of the data, unless 
    there are only two dimensions (in which case it is assumed to be the last dimension).

    References
    ----------
    .. [1] Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88, 365-411. 10.1016/S0047-259X(03)00096-4
    .. [2] Chen, Y., Wiesel, A., Eldar, Y.C., & Hero, A.O. (2009). Shrinkage algorithms for MMSE covariance estimation. arXiv. 10.48550/arXiv.0907.4698
    .. [3] Kessy, A., Lewin, A., & Strimmer, K. (2016). Optimal whitening and decorrelation. arXiv. 10.48550/arXiv.1512.00809
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import Covariance
    >>> X = torch.normal(0, 1, (100, 10, 100))
    >>> cov = Covariance(s_max = 20).fit(X)
    >>> cov.covariance_.shape
    torch.Size([10, 10])
    """
    
    def __init__(self, method: str = 'ledoitwolf', s_min: Union[float, None] = None, s_max: Union[float, None] = None):
        """Create a new Covariance instance.
        
        Parameters
        ----------
        method : {'empirical', 'ledoitwolf', 'oas'}, default = 'ledoitwolf'
            Which method should be applied for estimation of covariance?
        """
        
        if method not in _ESTIMATORS:
            raise ValueError(f'Unknown covariance estimation method {method}. Available methods: {list(_ESTIMATORS.keys())}.')

        self.method = method
        self.s_min = s_min
        self.s_max = s_max
    
    def _get_estimator(self, X: Union[np.ndarray, torch.Tensor]) -> sklearn.base.BaseEstimator:
        """Obtain an estimator based on the data type.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Data to fit the estimator on.
        
        Returns
        -------
        cov : sklearn.base.BaseEstimator
            Covariance estimator to use.
        """
        
        # choose method
        method = _ESTIMATORS[self.method]
        
        # choose backend
        if isinstance(X, torch.Tensor):
            return method['torch']
        elif isinstance(X, np.ndarray):
            return method['numpy']
        
        raise ValueError(f'Got an unexpected combination of method=`{self.method}` and type=`{type(X)}`.') 
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> "Covariance":
        """Fit the covariance estimator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Data to fit the estimator on of shape ``(n_trials, n_features[, n_timepoints])``.
        *args : Any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        self : Covariance
            Fitted covariance estimator.
        """
        
        return self._get_estimator(X)(s_min = self.s_min, s_max = self.s_max).fit(X)
    
    def transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Whiten data using the fitted covariance estimator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Data to transform of shape ``(n_trials, n_features[, n_timepoints])``.
        *args : Any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        W : np.ndarray | torch.Tensor
            Whitened data of shape ``(n_trials, n_features[, n_timepoints])``.
        """
        return self._get_estimator(X)(s_min = self.s_min, s_max = self.s_max).fit_transform(X)

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Fit the covariance estimator and whiten the data.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Data to fit the estimator on and transform of shape ``(n_trials, n_features[, n_timepoints])``.
        *args : Any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        W : np.ndarray | torch.Tensor
            Whitened data of shape ``(n_trials, n_features[, n_timepoints])``.
        """
        
        return self._get_estimator(X)(s_min = self.s_min, s_max = self.s_max).fit_transform(X)
    
    def to_torch(self) -> sklearn.base.BaseEstimator:
        """Create the torch estimator. Note that this function cannot be used for conversion.
        
        Returns
        -------
        cov : mvpy.estimators.Covariance
            The torch estimator.
        """
        
        return self._get_estimator(torch.tensor([1]))(s_min = self.s_min, s_max = self.s_max)
    
    def to_numpy(self) -> sklearn.base.BaseEstimator:
        """Create the numpy estimator. Note that this function cannot be used for conversion.

        Returns
        -------
        cov : mvpy.estimators.Covariance
            The numpy estimator.
        """

        return self._get_estimator(np.array([1]))(s_min = self.s_min, s_max = self.s_max)
    
    def clone(self) -> "Covariance":
        """Obtain a clone of this class.
        
        Returns
        -------
        cov : mvpy.estimators.Covariance
            The cloned object.
        """
        
        return Covariance(method = self.method, s_min = self.s_min, s_max = self.s_max)