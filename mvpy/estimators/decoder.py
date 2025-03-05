'''
A collection of estimators for decoding features using ridge decoders.
'''

import numpy as np
import torch
import sklearn

from .ridgecv import RidgeCV

from typing import Union, Any

class _Decoder_numpy(sklearn.base.BaseEstimator):
    """Obtain a new ridge decoder.
    
    Parameters
    ----------
    alpha : np.ndarray
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
    pattern_ : np.ndarray
        The decoded pattern.
    coef_ : np.ndarray
        The coefficeints of the decoder.
    intercept_ : np.ndarray
        The intercepts of the decoder.
    alpha_ : np.ndarray
        The penalties used for estimation.
    """
    
    def __init__(self, alpha: np.ndarray, **kwargs):
        """Obtain a new decoder.
        
        Parameters
        ----------
        alphas : np.ndarray
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments.
        """
        
        # setup opts
        self.alpha = alpha
        self.fit_intercept = True if 'fit_intercept' not in kwargs else kwargs['fit_intercept']
        self.normalise = True if 'normalise' not in kwargs else kwargs['normalise']
        self.alpha_per_target = True if 'alpha_per_target' not in kwargs else kwargs['alpha_per_target']
        
        # setup estimator
        self.estimator = RidgeCV(alphas = alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)
        self.pattern_ = None
        self.coef_ = None
        self.intercept_ = None
        self.alpha_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the decoder.
        
        Parameters
        ----------
        X : np.ndarray
            The features.
        y : np.ndarray
            The targets
        """
        
        # check shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of samples.')

        # check y shape
        if len(y.shape) == 1:
            y = y[:, None]
        
        # fit the decoder
        self.estimator.fit(X, y)
        self.alpha_ = self.estimator.alpha_
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        
        # compute covariance of X
        X = (X - X.mean(axis = 0, keepdims = True))
        S_X = np.cov(X.T)
        
        if y.shape[1] > 1:
            # compute covariance of y
            y = (y - y.mean(axis = 0, keepdims = True))
            P_y = np.linalg.pinv(np.cov(y.T))
            
            # compute pattern
            self.pattern_ = S_X.dot(self.estimator.coef_.T).dot(P_y)
        else:
            # compute correction
            P_y = 1.0 / float(y.shape[0] - 1)
            
            # compute pattern
            self.pattern_ = S_X.dot(self.estimator.coef_.T) * P_y
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the targets.

        Parameters
        ----------
        X : np.ndarray
            The features.

        Returns
        -------
        y : np.ndarray
        """
        
        # check fit
        if self.pattern_ is None:
            raise ValueError('The decoder has not been fitted yet.')
        
        # check shapes
        if X.shape[1] != self.estimator.coef_.shape[1]:
            raise ValueError('X and ß must have the same number of features.')
        
        return self.estimator.predict(X)
    
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        _Decoder_numpy
            The cloned object.
        """
        
        return _Decoder_numpy(alpha = self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)

class _Decoder_torch(sklearn.base.BaseEstimator):
    """Obtain a new ridge decoder.
    
    Parameters
    ----------
    alpha : torch.Tensor
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
    pattern_ : torch.Tensor
        The decoded pattern.
    coef_ : torch.Tensor
        The coefficeints of the decoder.
    intercept_ : torch.Tensor
        The intercepts of the decoder.
    alpha_ : torch.Tensor
        The penalties used for estimation.
    """
    
    def __init__(self, alpha: torch.Tensor, **kwargs):
        """Obtain a new decoder.
        
        Parameters
        ----------
        alphas : torch.Tensor
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments.
        """
        
        # setup opts
        self.alpha = alpha
        self.fit_intercept = True if 'fit_intercept' not in kwargs else kwargs['fit_intercept']
        self.normalise = True if 'normalise' not in kwargs else kwargs['normalise']
        self.alpha_per_target = True if 'alpha_per_target' not in kwargs else kwargs['alpha_per_target']
        
        # setup estimator
        self.estimator = RidgeCV(alphas = alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)
        self.pattern_ = None
        self.coef_ = None
        self.intercept_ = None
        self.alpha_ = None
        
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the decoder.
        
        Parameters
        ----------
        X : torch.Tensor
            The features.
        y : torch.Tensor
            The targets
        """
        
        # check shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of samples.')

        # check y shape
        if len(y.shape) == 1:
            y = y[:, None]
        
        # fit the decoder
        self.estimator.fit(X, y)
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        self.alpha_ = self.estimator.alpha_
        
        # compute covariance of X
        X = (X - X.mean(0, keepdim = True))
        S_X = torch.cov(X.T)
        
        if y.shape[1] > 1:
            # compute covariance of y
            y = (y - y.mean(0, keepdim = True))
            P_y = torch.linalg.pinv(torch.cov(y.T))
            
            # compute pattern
            self.pattern_ = S_X.mm(self.estimator.coef_.T).mm(P_y)
        else:
            # compute correction
            P_y = 1.0 / float(y.shape[0] - 1)
            
            # compute pattern
            self.pattern_ = S_X.mm(self.estimator.coef_.T) * P_y
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the targets.

        Parameters
        ----------
        X : torch.Tensor
            The features.

        Returns
        -------
        y : torch.Tensor
        """
        
        # check fit
        if self.pattern_ is None:
            raise ValueError('The decoder has not been fitted yet.')
        
        # check shapes
        if X.shape[1] != self.estimator.coef_.shape[1]:
            raise ValueError('X and ß must have the same number of features.')
        
        return self.estimator.predict(X)
    
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        _Decoder_torch
            The cloned object.
        """
        
        return _Decoder_torch(alpha = self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)
    
class Decoder(sklearn.base.BaseEstimator):
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