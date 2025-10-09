'''
A collection of estimators for common spatial patterns.
'''

import numpy as np
import torch
import sklearn

import warnings
from .covariance import _LedoitWolf_numpy, _LedoitWolf_torch, _Empirical_numpy, _Empirical_torch

from typing import Union, Any

class _CSP_numpy():
    def __init__(self):
        pass

def _ajd_pham_torch(X: torch.Tensor, init: Union[torch.Tensor, None] = None, eps: float = 1e-6, n_iter_max: int = 20, sample_weight: Union[torch.Tensor, None] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the approximate joint diagonalisation based on Pham's algorithm. This function ports the pyRiemann code to torch.
    
    Parameters
    ----------
    X : torch.Tensor
        Set of HPD matrices to diagnolise.
    init : torch.Tensor, default=None
        Initialisaiton for the diagonaliser.
    eps : float, default=1e-6
        Tolerance for convergence.
    n_iter_max : int, default=20
        Maximum number of iterations.
    sample_weight : torch.Tensor, default=None
        Sample weights.
    
    Returns
    -------
    V : torch.Tensor
        Diagonaliser.
    D : torch.Tensor
        Diagonalised matrices.
    """
    
    n_matrices, _, _ = X.shape
    normalised_weight = torch.ones((n_matrices, 1), dtype = X.dtype, device = X.device) / n_matrices if sample_weight is None else sample_weight / torch.sum(sample_weight)
    
    # reshape input matrix
    A = X.reshape((-1, X.shape[-1])).T
    n, n_matrices_x_n = A.shape
    
    # init variables
    V = torch.eye(n, dtype = X.dtype, device = X.device) if init is None else init
    epsilon = n * (n - 1) * eps
    
    for _ in range(n_iter_max):
        crit = 0
        
        for ii in range(1, n):
            for jj in range(ii):
                Ii = torch.arange(ii, n_matrices_x_n, n)
                Ij = torch.arange(jj, n_matrices_x_n, n)
                
                c1 = A[ii,Ii]
                c2 = A[jj,Ij]
                
                g12 = (A[ii,Ij] / c1) @ normalised_weight
                g21 = (A[ii,Ij] / c2) @ normalised_weight
                
                omega21 = (c1 / c2) @ normalised_weight
                omega12 = (c2 / c1) @ normalised_weight
                omega = torch.sqrt(omega12 * omega21)
                
                tmp = torch.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                if torch.isreal(X).all():
                    omega = torch.max(omega - 1, torch.tensor(1e-9))
                tmp2 = (tmp * g12 - g21) / omega
                
                h12 = tmp1 + tmp2
                h21 = torch.conj((tmp1 - tmp2) / tmp)
                
                crit += n_matrices * (g12 * torch.conj(h12) + g21 * h21) / 2.0
                
                tmp = 1 + 0.5j * torch.imag(h12 * h21)
                tmp = tmp + torch.sqrt(tmp ** 2 - h12 * h21)
                if torch.isreal(X).all():
                    tmp = torch.real(tmp)
                tau = torch.tensor([[1, torch.conj(-h12 / tmp)],
                                    [torch.conj(-h21 / tmp), 1]])
                
                A[[ii,jj],:] = tau.conj() @ A[[ii,jj],:]
                tmp = torch.cat(A[:,Ii].unsqueeze(1), A[:,Ij].unsqueeze(1), 1)
                tmp = tmp.reshape((n * n_matrices, 2), order = "F")
                tmp = tmp @ tau.T
                
                tmp = tmp.reshape((n, n_matrices * 2), order = "F")
                A[:,Ii] = tmp[:,:n_matrices]
                A[:,Ij] = tmp[:,n_matrices:]
                V[[ii,jj],:] = tau @ V[[ii,jj],:]
        
        if crit < epsilon:
            break
    else:
        warnings.warn('Convergence not reached.')
    
    D = A.reshape((n, -1, n)).transpose(1, 0, 2).conj()
    
    return V.to(X.dtype).to(X.device), D.to(X.dtype).to(X.device)

def _normalise_eigenvectors_torch(vec: torch.Tensor, covs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """"""
    
    cov = (covs * weights).sum(0) / weights.sum()
    
    for ii in range(vec.shape[1]):
        tmp = (vec[:,ii].T @ cov) @ vec[:,ii]
        vec[:,ii] = vec[:,ii] / torch.sqrt(tmp)
    
    return vec

class _CSP_torch():
    def __init__(self):
        pass
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """"""
        
        # get classes
        self.classes_, weights = torch.unique(y, return_counts = True)
        
        # check dimensions
        if len(self.classes_) < 2:
            raise ValueError('At least two classes are required.')
        
        # estimate covariance matrices
        classes_mask = [y == class_ for class_ in self.classes_]
        covs = torch.stack([_Empirical_torch().fit(X[mask]).covariance_ for mask in classes_mask], 0)
        
        # prewhiten
        #whitener_ = _Empirical_torch().fit(X).whitener_
        #covs = torch.stack([whitener_ @ cov for cov in covs], 0)
        
        # check multi-class or binary case
        if len(self.classes_) == 2:
            # in the binary case, we can just decompose
            val, vec = torch.linalg.eigh(covs[0])
        else:
            # in the multiclass case, we need to compute the approximate joint diagonalisation
            vec, d = _ajd_pham_torch(covs)
            vec = _normalise_eigenvectors_torch(vec.T, covs, weights)
            val = None
        #vec = whitener_ @ vec
        
        # order vectors
        if len(self.classes_) == 2:
            # for binary case, it's simple
            ix = torch.abs(val - 0.5).argsort().flip(-1)
        else:
            # for mutliclass, we take mutual information
            ps = weights / weights.sum()
            
            mi = []
            for jj in range(vec.shape[1]):
                aa, bb = 0, 0
                
                for cov, p in zip(covs, ps):
                    tmp = (vec[:,jj].T @ cov) @ vec[:,jj]
                    aa += p * torch.log(torch.sqrt(tmp))
                    bb += p * (tmp**2 - 1)
                
                mi.append(-(aa + (3.0 / 16) * (bb**2)))
            mi = torch.tensor(mi)
            
            ix = mi.argsort().flip(-1)
        
        vec = vec[:,ix]
        
        self.filters_ = vec.T
        self.patterns_ = torch.linalg.pinv(vec)
        
        return self

class CSP(sklearn.base.BaseEstimator):
    """Implements a simple linear ridge decoder.
    
    Parameters
    ----------
    alphas : Union[torch.Tensor, np.ndarray]
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to use a different penalty for each target.
    
    Attributes
    ----------
    estimator_ : mvpy.estimators.RidgeCV
        The ridge estimator.
    pattern_ : Union[torch.Tensor, np.ndarray]
        The decoded pattern.
    coef_ : Union[torch.Tensor, np.ndarray]
        The coefficeints of the decoder.
    intercept_ : Union[torch.Tensor, np.ndarray]
        The intercepts of the decoder.
    alpha_ : Union[torch.Tensor, np.ndarray]
        The penalties used for estimation.
    
    Notes
    -----
    After fitting the decoder, this class will also estimate the decoded patterns. This follows the approach detailed in [4]_. Please also be aware that, while this class supports decoding multiple features at once, these will principally be separate regressions wherein individual contributions are not disentangled. If you would like to do this, please consider using a back-to-back decoder.
    
    References
    ----------
    .. [4] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.D., Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96-110. 10.1016/j.neuroimage.2013.10.067
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import Decoder
    >>> X = torch.normal(0, 1, (100, 5))
    >>> ß = torch.normal(0, 1, (5, 60))
    >>> y = X @ ß + torch.normal(0, 1, (100, 60))
    >>> decoder = Decoder(alphas = torch.logspace(-5, 10, 20)).fit(y, X)
    >>> decoder.pattern_.shape
    torch.Size([60, 5])
    >>> decoder.predict(y).shape
    torch.size([100, 5])
    """
    
    def __new__(self, alphas: Union[torch.Tensor, np.ndarray, float, int] = 1, **kwargs) -> sklearn.base.BaseEstimator:
        """Obtain a new decoder.
        
        Parameters
        ----------
        alphas : Union[torch.Tensor, np.ndarray, float, int], default=1
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The decoder.
        """
        
        # check alphas
        if isinstance(alphas, float) | isinstance(alphas, int):
            alphas = torch.tensor([alphas])
        
        if isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        # determine estimator
        if isinstance(alphas, torch.Tensor):
            return _Decoder_torch(alpha = alphas, **kwargs)
        elif isinstance(alphas, np.ndarray):
            return _Decoder_numpy(alpha = alphas, **kwargs)
        
        raise ValueError(f'Alphas should be of type np.ndarray or torch.tensor, but got {type(alphas)}.')

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        """Fit the estimator.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The features.
        y : Union[np.ndarray, torch.Tensor]
            The targets.
        """

        raise NotImplementedError('This method is not implemented in the base class.')

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The features.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The predictions.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        Decoder
            The cloned object.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')