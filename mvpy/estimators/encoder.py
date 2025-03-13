'''
A collection of estimators for encoding features using ridge regressions.
'''

import numpy as np
import torch
import sklearn

from .ridgecv import RidgeCV

from typing import Union, Any

class _Encoder_numpy(sklearn.base.BaseEstimator):
    """Implements a simple ridge encoder.
    
    Parameters
    ----------
    alphas : np.ndarray, default=np.array([1.])
        The penalties to use for estimation.
    kwargs : Any
        Additional arguments for the estimator.
    
    Attributes
    ----------
    alphas : np.ndarray
        The penalties to use for estimation.
    kwargs : Any
        Additional arguments for the estimator.
    estimator :  mvpy.estimators.RidgeCV
        The estimator to use.
    intercept_ : np.ndarray
        The intercepts of the encoder.
    coef_ : np.ndarray
        The coefficients of the encoder.
    """
    
    def __init__(self, alphas: np.ndarray = np.array([1.]), **kwargs):
        """Initialise a new encoder.
        
        Parameters
        ----------
        alphas : np.ndarray, default=np.array([1.])
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments for the estimator.
        """
        
        # setup opts
        self.alphas = alphas
        self.kwargs = kwargs
        
        # setup the estimator
        self.estimator = RidgeCV(alphas = self.alphas, **self.kwargs)
        self.expanded_ = None
        self.intercept_ = None
        self.coef_ = None
        self.n_ = None
        self.f_ = None
        self.t_ = None
        self.d_ = None
    
    def _expand_matrices(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> Union[np.ndarray, tuple[np.ndarray]]:
        """Expand the design matrix and outcomes.
        
        Parameters
        ----------
        X : np.ndarray
            The features.
        y : Union[np.ndarray, None], optional
            The targets.
        
        Returns
        -------
        Union[np.ndarray, tuple[np.ndarray]]
            The expanded design matrix and outcomes.
        """
        
        # setup design matrix & outcomes
        X_t = np.zeros((X.shape[0] * X.shape[2], X.shape[1] * X.shape[2]))
        if y is not None: y_t = np.zeros((y.shape[0] * y.shape[2], y.shape[1]))

        # loop over trials
        for i in range(X.shape[0]):
            # loop over time points
            for j in range(X.shape[2]):
                # grab index in matrix
                ij = i * X.shape[2] + j
                
                # write data
                X_t[ij,j*X.shape[1]:(j+1)*X.shape[1]] = X[i,:,j]
                if y is not None: y_t[ij,:] = y[i,:,j]
        
        if y is not None: return (X_t, y_t)
        return X_t

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the encoder.
        
        Parameters
        ----------
        X : np.ndarray
            The features.
        y : np.ndarray
            The targets.
        """
        
        # check dims
        if len(y.shape) == 1:
            y = y[:, None]
        
        # check if we want an expanded encoder
        if (len(X.shape) == 3) & (len(y.shape) == 3):
            # track dimensions
            self.expanded_ = True
            self.n_, self.f_, self.t_ = X.shape
            self.d_ = y.shape[1]
            
            # expand the data
            X, y = self._expand_matrices(X, y)
        
        # fit the encoder
        self.estimator.fit(X, y)
        
        # save the data
        self.intercept_ = self.estimator.intercept_.copy()
        self.coef_ = self.estimator.coef_.copy()
        
        # check if we need to reshape data
        if self.expanded_:
            self.coef_ = self.coef_.reshape((self.d_, self.t_, self.f_)).swapaxes(1, 2)

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions from encoder.
        
        Parameters
        ----------
        X : np.ndarray
            The features.
        
        Returns
        -------
        y : np.ndarray
            The predictions.
        """
        
        # check fit
        if (self.intercept_ is None) | (self.coef_ is None):
            raise ValueError('Encoder has not been fitted yet.')
        
        # grab dims
        dims = X.shape
        
        # check if we need to expand data
        if self.expanded_:
            X = self._expand_matrices(X)
        
        # make predictions
        y = self.estimator.predict(X)
        
        # check if we need to reshape data
        if self.expanded_:
            y = y.reshape((dims[0], dims[-1], self.d_)).swapaxes(1, 2)
        
        return y
    
    def clone(self):
        """Obtain a clone of the estimator.
        
        Returns
        -------
        _Encoder_numpy
            The cloned estimator.
        """
        
        return _Encoder_numpy(alphas = self.alphas, **self.kwargs)

class _Encoder_torch(sklearn.base.BaseEstimator):
    """Implements a simple ridge encoder using torch.
    
    Parameters
    ----------
    alphas : torch.Tensor, default=torch.tensor([1.])
        The penalties to use for estimation.
    kwargs : Any
        Additional arguments for the estimator.
    
    Attributes
    ----------
    alphas : torch.Tensor
        The penalties to use for estimation.
    kwargs : Any
        Additional arguments for the estimator.
    estimator :  mvpy.estimators.RidgeCV
        The estimator to use.
    intercept_ : torch.Tensor
        The intercepts of the encoder.
    coef_ : torch.Tensor
        The coefficients of the encoder.
    """
    
    def __init__(self, alphas: torch.Tensor = torch.tensor([1.]), **kwargs):
        """Initialise a new encoder.
        
        Parameters
        ----------
        alphas : torch.Tensor, default=torch.tensor([1.])
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments for the estimator.
        """
        
        # setup opts
        self.alphas = alphas
        self.kwargs = kwargs
        
        # setup the estimator
        self.estimator = RidgeCV(alphas = self.alphas, **self.kwargs)
        self.expanded_ = None
        self.intercept_ = None
        self.coef_ = None
        self.n_ = None
        self.f_ = None
        self.t_ = None
        self.d_ = None
        
    def _expand_matrices(self, X: torch.Tensor, y: Union[torch.Tensor, None] = None) -> Union[torch.Tensor, tuple[torch.Tensor]]:
        """Expand the design matrix and outcomes.
        
        Parameters
        ----------
        X : torch.Tensor
            The features.
        y : Union[torch.Tensor, None], optional
            The targets.
        
        Returns
        -------
        Union[torch.Tensor, tuple[torch.Tensor]]
            The expanded design matrix and outcomes.
        """
        
        # setup design matrix & outcomes
        X_t = torch.zeros((X.shape[0] * X.shape[2], X.shape[1] * X.shape[2]), dtype = X.dtype, device = X.device)
        if y is not None: y_t = torch.zeros((y.shape[0] * y.shape[2], y.shape[1]), dtype = y.dtype, device = y.device)

        # loop over trials
        for i in range(X.shape[0]):
            # loop over time points
            for j in range(X.shape[2]):
                # grab index in matrix
                ij = i * X.shape[2] + j
                
                # write data
                X_t[ij,j*X.shape[1]:(j+1)*X.shape[1]] = X[i,:,j]
                if y is not None: y_t[ij,:] = y[i,:,j]
        
        if y is not None: return (X_t, y_t)
        return X_t

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the encoder.
        
        Parameters
        ----------
        X : torch.Tensor
            The features.
        y : torch.Tensor
            The targets.
        """
        
        # check dims
        if len(y.shape) == 1:
            y = y[:, None]
        
        # check if we want an expanded encoder
        if (len(X.shape) == 3) & (len(y.shape) == 3):
            # track dimensions
            self.expanded_ = True
            self.n_, self.f_, self.t_ = X.shape
            self.d_ = y.shape[1]
            
            # expand the data
            X, y = self._expand_matrices(X, y)
        
        # fit the encoder
        self.estimator.fit(X, y)
        
        # save the data
        self.intercept_ = self.estimator.intercept_.clone()
        self.coef_ = self.estimator.coef_.clone()
        
        # check if we need to reshape data
        if self.expanded_:
            self.coef_ = self.coef_.reshape((self.d_, self.t_, self.f_)).swapaxes(1, 2)

        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions from encoder.
        
        Parameters
        ----------
        X : torch.Tensor
            The features.
        
        Returns
        -------
        y : torch.Tensor
            The predictions.
        """
        
        # check fit
        if (self.intercept_ is None) | (self.coef_ is None):
            raise ValueError('Encoder has not been fitted yet.')
        
        # grab dims
        dims = X.shape
        
        # check if we need to expand data
        if self.expanded_:
            X = self._expand_matrices(X)
        
        # make predictions
        y = self.estimator.predict(X)
        
        # check if we need to reshape data
        if self.expanded_:
            y = y.reshape((dims[0], dims[-1], self.d_)).swapaxes(1, 2)
        
        return y
    
    def clone(self):
        """Obtain a clone of the estimator.
        
        Returns
        -------
        _Encoder_torch
            The cloned estimator.
        """
        
        return _Encoder_torch(alphas = self.alphas, **self.kwargs)

class Encoder(sklearn.base.BaseEstimator):
    """Implements a simple linear ridge encoder. This class essentially just wraps `mvpy.estimators.RidgeCV`, exposing this as a more convenient name.
    
    Parameters
    ----------
    alphas : Union[torch.Tensor, np.ndarray, float, int], default=1
        The penalties to use for estimation.
    kwargs : Any
        Additional arguments.
    
    Attributes
    ----------
    alphas : torch.Tensor
        The penalties to use for estimation.
    kwargs : Any
        Additional arguments for the estimator.
    estimator :  mvpy.estimators.RidgeCV
        The estimator to use.
    intercept_ : torch.Tensor
        The intercepts of the encoder.
    coef_ : torch.Tensor
        The coefficients of the encoder.
    
    Notes
    -----
    This class is a wrapper around `mvpy.estimators.RidgeCV`. Really, it exists mostly to make the code more readable.
    
    However, this class may also be used for a temporally expanded encoder. This may be useful in cases where you would like to encode in time, but would like to impose the constraint that alpha should be fit over all time points. To use this class in this manner, simply supply 3D `X` and `y` tensors.
    
    Examples
    --------
    Let's say we want to do a very simple encoding:
    
    >>> import torch
    >>> from mvpy.estimators import Encoder
    >>> ß = torch.normal(0, 1, (50,))
    >>> X = torch.normal(0, 1, (100, 50))
    >>> y = X @ ß
    >>> y = y[:,None] + torch.normal(0, 1, (100, 1))
    >>> encoder = Encoder().fit(X, y)
    >>> encoder.coef_.shape
    torch.Size([1, 50])
    
    Next, let's assume we want to do a temporally expanded encoding instead:
    
    >>> import torch
    >>> from mvpy.estimators import Encoder
    >>> X = torch.normal(0, 1, (240, 5, 100))
    >>> ß = torch.normal(0, 1, (60, 5, 100))
    >>> y = torch.stack([torch.stack([X[:,:,i] @ ß[j,:,i] for i in range(X.shape[2])], 0) for j in range(ß.shape[0])], 0).swapaxes(0, 2).swapaxes(1, 2)
    >>> y = y + torch.normal(0, 1, y.shape)
    >>> encoder = Encoder().fit(X, y)
    >>> encoder.coef_.shape
    torch.Size([60, 5, 100])
    """
    
    def __new__(self, alphas: Union[torch.Tensor, np.ndarray, float, int] = 1, **kwargs) -> sklearn.base.BaseEstimator:
        """Obtain a new encoder.
        
        Parameters
        ----------
        alphas : Union[torch.Tensor, np.ndarray, float, int], default=1
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The encoder.
        """
        
        # check alphas
        if isinstance(alphas, float) | isinstance(alphas, int):
            alphas = torch.tensor([alphas])
        
        if isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        # determine estimator
        if isinstance(alphas, torch.Tensor):
            return _Encoder_torch(alphas = alphas, **kwargs)
        elif isinstance(alphas, np.ndarray):
            return _Encoder_numpy(alphas = alphas, **kwargs)
        
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
        Encoder
            The cloned object.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')