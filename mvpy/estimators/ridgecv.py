'''
A collection of estimators for fitting cross-validated ridge regressions.
'''

import numpy as np
import torch
import sklearn

from typing import Union, Any

from sklearn.linear_model import RidgeCV as _RidgeCV_numpy

class _RidgeCV_torch(sklearn.base.BaseEstimator):
    """Implements RidgeCV using torch as our backend. This class owes greatly to J.R. King's RidgeCV implementation[3]_.
    
    Attributes
    ----------
    alpha_ : torch.Tensor
        The penalties used for estimation.
    intercept_ : torch.Tensor
        The intercepts.
    coef_ : torch.Tensor
        The coefficients.
    
    References
    ----------
    .. [3] King, J.R. (2020). torch_ridge. https://github.com/kingjr/torch_ridge
    """
    
    def __init__(self, alphas: Union[torch.Tensor, list, float, int] = 1, fit_intercept: bool = True, normalise: bool = True, alpha_per_target: bool = False):
        """Obtain a RidgeCV estimator.
        
        Parameters
        ----------
        alphas : Union[torch.Tensor, list, float, int], default=1
            Penalties to use for estimation.
        fit_intercept : bool, default=True
            Whether to fit an intercept.
        normalise : bool, default=True
            Whether to normalise the data.
        alpha_per_target : bool, default=False
            Whether to use a different penalty for each target.
        """
        
        # check alphas
        if isinstance(alphas, int) | isinstance(alphas, float):
            alphas = torch.tensor([alphas])
        
        if isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalise = normalise
        self.alpha_per_target = alpha_per_target
        
        self.alpha_ = None
        self.coef_ = None
        self.intercept_ = None
    
    def _preprocess(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor]:
        """Preprocess the data.

        Parameters
        ----------
        X : torch.Tensor
            The features.
        y : torch.Tensor
            The targets.

        Returns
        -------
        X : torch.Tensor
            The preprocessed features.
        y : torch.Tensor
            The preprocessed targets.
        X_offset : torch.Tensor
            Offsets in X
        X_scale : torch.Tensor
            Scales in X
        y_offset : torch.Tensor
            Offsets in y
        y_scale : torch.Tensor
            Scales in y
        """

        if self.fit_intercept:
            # find and demean
            X_offset = X.mean(0)
            X -= X_offset
            
            # same for scale, if required
            if self.normalise:
                X_scale = torch.sqrt((X**2).sum(0))
                X /= X_scale
            else:
                X_scale = torch.ones(X.shape[1], dtype = X.dtype, device = X.device)
            
            # same for y
            y_offset = y.mean(0)
            y -= y_offset
        else:
            # otherwise, just zero out
            X_offset = torch.zeros(X.shape[1], dtype = X.dtype, device = X.device)
            X_scale = torch.ones(X.shape[1], dtype = X.dtype, device = X.device)
            y_offset = torch.zeros(y.shape[1], dtype = X.dtype, device = X.device)
        
        return (X, y, X_offset, X_scale, y_offset)
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the RidgeCV model.
        
        Parameters
        ----------
        X : torch.Tensor
            The features.
        y : torch.Tensor
            The targets.
        """
        
        # check shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'`X` and `y` must have the same number of samples, but got {X.shape[0]} and {y.shape[0]}')

        # check y shape
        if len(y.shape) == 1:
            y = y[:, None]
        
        # get dims
        n, n_x = X.shape
        _, n_y = y.shape
        
        # preprocess
        X, y = X.clone(), y.clone() # make sure we don't have in-place changes
        X, y, X_offset, X_scale, y_offset = self._preprocess(X, y)
        self.alphas = self.alphas.to(X.dtype).to(X.device)
        
        # decomposition
        U, S, _ = torch.linalg.svd(X, full_matrices = False)
        v = S ** 2
        Uy = U.transpose(0, 1) @ y
        
        # perform LOO per alpha
        cv_duals = torch.zeros((self.alphas.shape[0], n, n_y), dtype = X.dtype, device = X.device)
        cv_errors = torch.zeros_like(cv_duals, dtype = X.dtype, device = X.device)
        
        for a_i, alpha in enumerate(self.alphas):
            # solve for duals
            w = ((v + alpha) ** - 1) - alpha ** -1
            cv_duals[a_i] = U @ torch.diag(w) @ Uy + alpha ** -1 * y
            
            # compute errors
            G = ((w * U**2).sum(-1) + alpha ** -1)[:,None]
            cv_errors[a_i] = cv_duals[a_i] / G
        
        # if required, find alpha per feature
        if self.alpha_per_target:
            best = (cv_errors**2).mean(1).argmin(0)
            
            duals = torch.zeros((n, n_y), dtype = X.dtype, device = X.device)
            errors = torch.zeros_like(duals, dtype = X.dtype, device = X.device)
            
            for a_i, best_a in enumerate(best):
                duals[:,a_i] = cv_duals[best_a,:,a_i]
                errors[:,a_i] = cv_errors[best_a,:,a_i]
        else:
            best = (cv_errors.reshape(len(self.alphas), -1)**2).mean(1).argmin(0)
            duals = cv_duals[best]
            errors = cv_errors[best]
        
        # set alpha and coef
        self.alpha_ = self.alphas[best]
        self.coef_ = duals.transpose(0, 1) @ X
        
        # find intercept
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale[None,:]
            Xoff_coef = X_offset[None,:] @ self.coef_.transpose(1, 0)
            self.intercept_ = y_offset - Xoff_coef
        else:
            self.intercept_ = 0.0
        
        # fix on some numpy versions
        if len(self.coef_.shape) == 1:
            self.coef_ = self.coef_[None,:]
        
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions from fitted model.
        
        Parameters
        ----------
        X : torch.Tensor
            The features.
        
        Returns
        -------
        y : torch.Tensor
            The predictions.
        """
        
        # make sure model has been fit
        if (self.coef_ is None) | (self.intercept_ is None):
            raise ValueError('Model has not been fit yet.')

        return X @ self.coef_.transpose(1, 0) + self.intercept_

    def clone(self):
        """Make a clone of this class.
        
        Returns
        -------
        RidgeCV
            A clone of this class.
        """
        
        return _RidgeCV_torch(alphas = self.alphas, 
                              fit_intercept = self.fit_intercept, 
                              normalsie = self.normalise, 
                              alpha_per_target = self.alpha_per_target)

class RidgeCV(sklearn.base.BaseEstimator):
    """Implements RidgeCV using torch as our backend.
    
    Parameters
    ----------
    alphas : Union[torch.Tensor, list, float, int], default=torch.Tensor([1])
        Penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=True
        Whether to use a different penalty for each target.
    
    Attributes
    ----------
    alpha_ : torch.Tensor
        The penalties used for estimation.
    intercept_ : torch.Tensor
        The intercepts.
    coef_ : torch.Tensor
        The coefficients.
    
    Notes
    -----
    This class owes greatly to J.R. King's RidgeCV implementation[3]_. If data are supplied as numpy, this class will fall back to :func:`sklearn.linear_model.RidgeCV`[4]_.
    
    References
    ----------
    .. [3] King, J.R. (2020). torch_ridge. https://github.com/kingjr/torch_ridge
    .. [4] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import RidgeCV
    >>> ß = torch.normal(0, 1, size = (5,))
    >>> X = torch.normal(0, 1, size = (240, 5))
    >>> y = ß @ X.T + torch.normal(0, 0.5, size = (X.shape[0],))
    >>> model = RidgeCV().fit(X, y)
    >>> model.coef_
    """
    
    def __new__(self, alphas: Union[np.ndarray, torch.Tensor, list, float, int] = 1, fit_intercept: bool = True, normalise: bool = True, alpha_per_target: bool = False):
        """Obtain a RidgeCV estimator.
        
        Parameters
        ----------
        alphas : Union[torch.Tensor, list, float, int], default=torch.Tensor([1])
            Penalties to use for estimation.
        fit_intercept : bool, default=True
            Whether to fit an intercept.
        normalise : bool, default=True
            Whether to normalise the data.
        alpha_per_target : bool, default=False
            Whether to use a different penalty for each target.
        """
        
        # check alphas
        if isinstance(alphas, int) | isinstance(alphas, float):
            alphas = torch.tensor([alphas])
        
        if isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        # check model type
        if isinstance(alphas, torch.Tensor):
            return _RidgeCV_torch(alphas = alphas, fit_intercept = fit_intercept, normalise = normalise, alpha_per_target = alpha_per_target)
        elif isinstance(alphas, np.ndarray):
            return _RidgeCV_numpy(alphas = alphas, fit_intercept = fit_intercept, alpha_per_target = alpha_per_target)
        
        raise ValueError(f'Alphas should be of type np.ndarray or torch.tensor, but got {type(alphas)}.')
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        """Fit the estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            The features.
        y : torch.Tensor
            The targets.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            The features.
        
        Returns
        -------
        y : torch.Tensor
            The predictions.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def clone(self):
        """Make a clone of this class.
        
        Returns
        -------
        RidgeCV
            A clone of this class.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')