'''
A collection of estimators for decoding and disentangling features using back2back regression.
'''

import numpy as np
import torch
import sklearn

from .ridgecv import _RidgeCV_numpy, _RidgeCV_torch

from typing import Union, Any

class _TimeDelayed_numpy(sklearn.base.BaseEstimator):
    """Initialise a new TimeDelayed estimator using numpy backend.
    
    Parameters
    ----------
    t_min : float
        The minimum time delay.
    t_max : float
        The maximum time delay.
    fs : int
        The sampling frequency.
    alphas : np.ndarray, default=np.array([1])
        The penalties to use for estimation.
    patterns : bool, default=False
        Should patterns be estimated?
    kwargs : Any
        Additional arguments for the estimator.
    
    Attributes
    ----------
    alphas : np.ndarray
        The penalties to use for estimation.
    kwargs : Any
        Additional arguments.
    patterns : bool
        Should patterns be estimated?
    t_min : float
        The minimum time delay.
    t_max : float
        The maximum time delay.
    fs : int
        The sampling frequency.
    window : np.ndarray
        The window to use for estimation.
    estimator : mvpy.estimators.RidgeCV
        The estimator to use.
    f_ : int
        The number of output features.
    c_ : int
        The number of input features.
    w_ : int
        The number of time delays.
    intercept_ : np.ndarray
        The intercepts of the estimator.
    coef_ : np.ndarray
        The coefficients of the estimator.
    pattern_ : np.ndarray
        The patterns of the estimator.
    """
    
    def __init__(self, t_min: float, t_max: float, fs: int, alphas: np.ndarray = np.array([1]), patterns: bool = False, **kwargs):
        """Initialise a new TimeDelayed estimator.
        
        Parameters
        ----------
        t_min : float
            The minimum time delay.
        t_max : float
            The maximum time delay.
        fs : int
            The sampling frequency.
        alphas : np.ndarray, default=np.array([1])
            The penalties to use for estimation.
        patterns : bool, default=False
            Should patterns be estimated?
        kwargs : Any
            Additional arguments for the estimator.
        """
        
        # setup opts
        self.alphas = alphas
        self.kwargs = kwargs
        self.patterns = patterns
        
        # setup timing
        self.t_min, self.t_max = t_min, t_max
        self.fs = fs
        
        # setup window
        t_neg = np.arange(np.abs(np.ceil(self.t_min * self.fs)) + 1)
        t_pos = np.arange(np.abs(np.ceil(self.t_max * self.fs)) + 1)
        self.window = np.unique(np.concatenate((-t_neg[::-1], t_pos))).astype(int)

        # setup estimator
        self.estimator = _RidgeCV_numpy(alphas = self.alphas, **self.kwargs)
        
        # setup attributes
        self.f_ = None
        self.c_ = None
        self.w_ = None
        self.intercept_ = None
        self.coef_ = None
        self.pattern_ = None
    
    def _delay_matrices(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> Union[np.ndarray, tuple[np.ndarray]]:
        """Create a temporally expanded design matrix.
        
        Parameters
        ----------
        X : np.ndarray
            The data to delay.
        y : np.ndarray, optional
            The target data to delay, if required.
        
        Returns
        -------
        Union[np.ndarray, tuple[np.ndarray]]
            The delayed data, either X or (X, y)
        """
        
        # obtain dimensions
        trials, channels, time = X.shape
        window_size = self.window.shape[0]
        
        # generate indices for time windows
        win_indices = np.arange(time).repeat(window_size).reshape(time, window_size) + self.window
        win_indices = np.clip(win_indices, 0, time - 1)  # Ensure indices are within valid range
        
        # expand X to extract windows efficiently
        X_win = X[:, :, win_indices]  # (trials, channels, time, window_size)
        X_win[:, :, (win_indices < 0) | (win_indices >= time)] = 0  # zero out edges
        
        # reshape to flattened design matrix
        X_t = X_win.transpose(0, 2, 1, 3).reshape(trials * time, channels * window_size)
        
        # check if y is provided
        if y is not None:
            # expand y accordingly
            y_t = y.transpose(0, 2, 1).reshape(trials * time, -1)
            
            return X_t, y_t
        
        return X_t
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the estimator.
        
        Parameters
        ----------
        X : np.ndarray
            The features
        y : np.ndarray
            The targets
        """
        
        # check dimensions
        if (len(X.shape) != 3) | (len(y.shape) != 3):
            raise ValueError(f'X and y must have 3 dimensions (trials, channels, time), but got {X.shape} and {y.shape}.')
        
        # check shapes
        if (X.shape[0] != y.shape[0]) | (X.shape[2] != y.shape[2]):
            raise ValueError(f'X and y must have the same number of trials and time points, but got {X.shape} and {y.shape}.')
        
        # keep track of dimensions
        self.f_, self.c_, self.w_ = y.shape[1], X.shape[1], self.window.shape[0]
        
        # obtain delayed matrices
        X, y = self._delay_matrices(X, y)
        
        # fit estimator
        self.estimator.fit(X, y)
        
        # set intercepts
        self.intercept_ = self.estimator.intercept_
        
        # obtain coefficients
        self.coef_ = self.estimator.coef_.reshape((self.f_, self.c_, self.w_))

        # if desired, also get the patterns
        if self.patterns:
            # demean
            X = X - X.mean(axis = 0, keepdims = True)
            
            # get covariance of X
            S_X = np.cov(X.T)
            
            # get precision of y
            if y.shape[1] > 1:
                # demean
                y = y - y.mean(axis = 0, keepdims = True)
                
                # get covariance of y
                P_y = np.linalg.inv(np.cov(y.T))
            else:
                P_y = 1.0 / float(y.shape[0] - 1)
            
            # obtain inverse patterns
            if y.shape[1] > 1: 
                self.pattern_ = S_X.dot(self.estimator.coef_.T).dot(P_y)
            else: 
                self.pattern_ = S_X.dot(self.estimator.coef_.T) * P_y
            
            # reshape patterns
            self.pattern_ = self.pattern_.reshape((self.f_, self.c_, self.w_))

        return self
    
    def predict(self, X: np.ndarray, reshape: bool = True) -> np.ndarray:
        """Make predictions from estimator.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        reshape : bool
            If True, reshape output to original shape.
        
        Returns
        -------
        y : np.ndarray
            Predictions.
        """
        
        # check fit
        if (self.coef_ is None) or (self.intercept_ is None):
            raise ValueError('Estimator has not been fitted yet.')
        
        # check dimensions
        if len(X.shape) != 3:
            raise ValueError(f'X must have 3 dimensions (trials, channels, time), but got {X.shape}.')
        
        # keep track of dimensions
        n, c, t = X.shape
        
        # obtain delayed matrices
        X = self._delay_matrices(X)
        
        # make predictions
        y = self.estimator.predict(X)
        
        # reshape to original shape, if desired
        if reshape: 
            y = y.reshape((n, t, self.f_)).swapaxes(1, 2)
        
        return y
    
    def clone(self):
        """Obtain a clone of the estimator.
        
        Returns
        -------
        _TimeDelayed_numpy
            A clone of the estimator.
        """
        
        return _TimeDelayed_numpy(self.t_min, self.t_max, self.fs, alphas = self.alphas, patterns = self.patterns, **self.kwargs)

class _TimeDelayed_torch(sklearn.base.BaseEstimator):
    """Initialise a new TimeDelayed estimator using torch backend.
    
    Parameters
    ----------
    t_min : float
        The minimum time delay.
    t_max : float
        The maximum time delay.
    fs : int
        The sampling frequency.
    alphas : torch.Tensor, default=torch.tensor([1])
        The penalties to use for estimation.
    patterns : bool, default=False
        Should patterns be estimated?
    kwargs : Any
        Additional arguments for the estimator.
    
    Attributes
    ----------
    alphas : torch.Tensor
        The penalties to use for estimation.
    kwargs : Any
        Additional arguments.
    patterns : bool
        Should patterns be estimated?
    t_min : float
        The minimum time delay.
    t_max : float
        The maximum time delay.
    fs : int
        The sampling frequency.
    window : torch.Tensor
        The window to use for estimation.
    estimator : mvpy.estimators.RidgeCV
        The estimator to use.
    f_ : int
        The number of output features.
    c_ : int
        The number of input features.
    w_ : int
        The number of time delays.
    intercept_ : torch.Tensor
        The intercepts of the estimator.
    coef_ : torch.Tensor
        The coefficients of the estimator.
    pattern_ : torch.Tensor
        The patterns of the estimator.
    """
    
    def __init__(self, t_min: float, t_max: float, fs: int, alphas: torch.Tensor = torch.tensor([1]), patterns: bool = False, **kwargs):
        """Initialise a new TimeDelayed estimator.
        
        Parameters
        ----------
        t_min : float
            The minimum time delay.
        t_max : float
            The maximum time delay.
        fs : int
            The sampling frequency.
        alphas : torch.Tensor, default=torch.tensor([1])
            The penalties to use for estimation.
        patterns : bool, default=False
            Should patterns be estimated?
        kwargs : Any
            Additional arguments for the estimator.
        """
        
        # setup opts
        self.alphas = alphas
        self.kwargs = kwargs
        self.patterns = patterns
        
        # setup timing
        self.t_min, self.t_max = t_min, t_max
        self.fs = fs
        
        # setup window
        t_neg = np.arange(np.abs(np.ceil(self.t_min * self.fs)) + 1)
        t_pos = np.arange(np.abs(np.ceil(self.t_max * self.fs)) + 1)
        self.window = np.unique(np.concatenate((-t_neg[::-1], t_pos)))
        self.window = torch.from_numpy(self.window).to(torch.int32).to(self.alphas.device)
        
        # setup estimator
        self.estimator = _RidgeCV_torch(alphas = self.alphas, **self.kwargs)
        
        # setup attributes
        self.f_ = None
        self.c_ = None
        self.w_ = None
        self.intercept_ = None
        self.coef_ = None
        self.pattern_ = None
    
    def _delay_matrices(self, X: torch.Tensor, y: Union[torch.Tensor, None] = None) -> Union[torch.Tensor, tuple[torch.Tensor]]:
        """Create a temporally expanded design matrix.
        
        Parameters
        ----------
        X : torch.Tensor
            The data to delay.
        y : torch.Tensor, optional
            The target data to delay, if required.
        
        Returns
        -------
        Union[torch.Tensor, tuple[torch.Tensor]]
            The delayed data, either X or (X, y)
        """
        
        # obtain dimensions
        trials, channels, time = X.shape
        window_size = self.window.shape[0]
        
        # generate time window indices
        win_indices = torch.arange(time, device = X.device).repeat(window_size, 1) + self.window[:, None]
        win_indices = torch.clamp(win_indices, 0, time - 1)  # ensure indices are within valid range
        
        # expand X to extract windows efficiently
        X_win = X[:, :, win_indices] # (trials, channels, window_size, time)
        X_win[:, :, (win_indices < 0) | (win_indices >= time)] = 0 # zero out for edge correction
        
        # reshape as flattened design matrix
        X_t = X_win.permute(0, 3, 1, 2).reshape(trials * time, channels * window_size)
        
        # check if y also needs to be transformed
        if y is not None:
            y_t = y.permute(0, 2, 1).reshape(trials * time, -1) # flatten y accordingly
            
            return X_t, y_t
        
        return X_t
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            The features
        y : torch.Tensor
            The targets
        """
        
        # check dimensions
        if (len(X.shape) != 3) | (len(y.shape) != 3):
            raise ValueError(f'X and y must have 3 dimensions (trials, channels, time), but got {X.shape} and {y.shape}.')
        
        # check shapes
        if (X.shape[0] != y.shape[0]) | (X.shape[2] != y.shape[2]):
            raise ValueError(f'X and y must have the same number of trials and time points, but got {X.shape} and {y.shape}.')
        
        # keep track of dimensions
        self.f_, self.c_, self.w_ = y.shape[1], X.shape[1], self.window.shape[0]
        
        # obtain delayed matrices
        X, y = self._delay_matrices(X, y)
        
        # fit estimator
        self.estimator.fit(X, y)
        
        # set intercepts
        self.intercept_ = self.estimator.intercept_
        
        # obtain coefficients
        self.coef_ = self.estimator.coef_.reshape((self.f_, self.c_, self.w_))
        
        # if desired, also get the patterns
        if self.patterns:
            # demean
            X = X - X.mean(0, keepdim = True)
            
            # get covariance of X
            S_X = torch.cov(X.T)
            
            # get precision of y
            if y.shape[1] > 1:
                # demean
                y = y - y.mean(0, keepdim = True)
                
                # get covariance of y
                P_y = torch.linalg.inv(torch.cov(y.T))
            else:
                P_y = 1.0 / float(y.shape[0] - 1)
            
            # obtain inverse patterns
            if y.shape[1] > 1: 
                self.pattern_ = S_X.mm(self.estimator.coef_.T).mm(P_y)
            else: 
                self.pattern_ = S_X.mm(self.estimator.coef_.T) * P_y
            
            # reshape patterns
            self.pattern_ = self.pattern_.reshape((self.f_, self.c_, self.w_))

        return self
    
    def predict(self, X: torch.Tensor, reshape: bool = True) -> torch.Tensor:
        """Make predictions from estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data.
        reshape : bool
            If True, reshape output to original shape.
        
        Returns
        -------
        y : torch.Tensor
            Predictions.
        """
        
        # check fit
        if (self.coef_ is None) or (self.intercept_ is None):
            raise ValueError('Estimator has not been fitted yet.')
        
        # check dimensions
        if len(X.shape) != 3:
            raise ValueError(f'X must have 3 dimensions (trials, channels, time), but got {X.shape}.')
        
        # keep track of dimensions
        n, c, t = X.shape
        
        # obtain delayed matrices
        X = self._delay_matrices(X)
        
        # make predictions
        y = self.estimator.predict(X)
        
        # reshape to original shape, if desired
        if reshape: 
            y = y.reshape((n, t, self.f_)).swapaxes(1, 2)
        
        return y
    
    def clone(self):
        """Obtain a clone of the estimator.
        
        Returns
        -------
        _TimeDelayed_torch
            A clone of the estimator.
        """
        
        return _TimeDelayed_torch(self.t_min, self.t_max, self.fs, alphas = self.alphas, patterns = self.patterns, **self.kwargs)

class TimeDelayed(sklearn.base.BaseEstimator):
    r"""Implements TimeDelayed regression.
    
    Parameters
    ----------
    t_min : float
        The minimum time delay. Note that positive values indicate X is delayed relative to y. This is unlike MNE's behaviour.
    t_max : float
        The maximum time delay. Note that positive values indicate X is delayed relative to y. This is unlike MNE's behaviour.
    fs : int
        The sampling frequency.
    alphas : Union[np.ndarray, torch.Tensor], default=torch.tensor([1])
        The penalties to use for estimation.
    patterns : bool, default=False
        Should patterns be estimated?
    kwargs : Any
        Additional arguments for the estimator.
    
    Attributes
    ----------
    alphas : Union[np.ndarray, torch.Tensor]
        The penalties to use for estimation.
    kwargs : Any
        Additional arguments.
    patterns : bool
        Should patterns be estimated?
    t_min : float
        The minimum time delay. Note that positive values indicate X is delayed relative to y. This is unlike MNE's behaviour.
    t_max : float
        The maximum time delay. Note that positive values indicate X is delayed relative to y. This is unlike MNE's behaviour.
    fs : int
        The sampling frequency.
    window : Union[np.ndarray, torch.Tensor]
        The window to use for estimation.
    estimator : mvpy.estimators.RidgeCV
        The estimator to use.
    f_ : int
        The number of output features.
    c_ : int
        The number of input features.
    w_ : int
        The number of time delays.
    intercept_ : Union[np.ndarray, torch.Tensor]
        The intercepts of the estimator.
    coef_ : Union[np.ndarray, torch.Tensor]
        The coefficients of the estimator.
    pattern_ : Union[np.ndarray, torch.Tensor]
        The patterns of the estimator.
    
    Notes
    -----
    This class allows estimation of either multivariate temporal response functions (mTRF) or stimulus reconstruction (SR) models.
    
    mTRFs are estimated as:

    .. math::
        r(t,n) = \\sum_\\tau w(\\tau, n) s(t - \\tau) + \\epsilon
    
    where :math:`r(t,n)` is the reconstructed signal at timepoint :math:`t` for channel :math:`n`, :math:`s(t)` is the stimulus at time :math:`t`, :math:`w(\tau, n)` is the weight at time delay :math:`\tau` for channel :math:`n`, and :math:`\epsilon` is the error.
    
    SR models are estimated as:
    
    .. math::
        s(t) = \\sum_n\\sum_\\tau r(t + \\tau, n) g(\\tau, n)
    
    where :math:`s(t)` is the reconstructed stimulus at time :math:`t`, :math:`r(t,n)` is the neural response at :math:`t` and lagged by :math:`\\tau` for channel :math:`n`, :math:`g(\tau, n)` is the weight at time delay :math:`\tau` for channel :math:`n`.
    
    For more information on mTRF or SR models, see [1]_.
    
    Note that for SR models it is recommended to also pass `patterns=True` to estimate not only the coefficients but also the patterns that were actually used for reconstructing stimuli. For more information, see [2]_.
    
    References
    ----------
    .. [1] Crosse, M.J., Di Liberto, G.M., Bednar, A., & Lalor, E.C. (2016). The multivariate temporal response function (mTRF) toolbox: A MATLAB toolbox for relating neural signals to continuous stimuli. Frontiers in Human Neuroscience, 10, 604. 10.3389/fnhum.2016.00604
    .. [2] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.D., Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96-110. 10.1016/j.neuroimage.2013.10.067
    
    Examples
    --------
    For mTRF estimation, we can do:
    
    >>> import torch
    >>> from mvpy.estimators import TimeDelayed
    >>> ß = torch.tensor([1., 2., 3., 2., 1.])
    >>> X = torch.normal(0, 1, (100, 1, 50))
    >>> y = torch.nn.functional.conv1d(X, ß[None,None,:], padding = 'same')
    >>> y = y + torch.normal(0, 1, y.shape)
    >>> trf = TimeDelayed(-2, 2, 1, alphas = 1e-5)
    >>> trf.fit(X, y).coef_
    tensor([[[0.9290, 1.9101, 2.8802, 1.9790, 0.9453]]])
    
    For stimulus reconstruction, we can do:
    
    >>> import torch
    >>> from mvpy.estimators import TimeDelayed
    >>> ß = torch.tensor([1., 2., 3., 2., 1.])
    >>> X = torch.arange(50)[None,None,:] * torch.ones((100, 1, 50))
    >>> y = torch.nn.functional.conv1d(X, ß[None,None,:], padding = 'same')
    >>> y = y + torch.normal(0, 1, y.shape)
    >>> X, y = y, X
    >>> sr = TimeDelayed(-2, 2, 1, alphas = 1e-3, patterns = True).fit(X, y)
    >>> sr.predict(X).mean(0)[0,:]
    tensor([ 1.3591,  1.2549,  1.5662,  2.3544,  3.3440,  4.3683,  5.4097,  6.4418, 7.4454,  8.4978,  9.5206, 10.5374, 11.5841, 12.6102, 13.6254, 14.6939, 15.6932, 16.7168, 17.7619, 18.8130, 19.8182, 20.8687, 21.8854, 22.9310, 23.9270, 24.9808, 26.0085, 27.0347, 28.0728, 29.0828, 30.1400, 31.1452, 32.1793, 33.2047, 34.2332, 35.2717, 36.2945, 37.3491, 38.3800, 39.3817, 40.3962, 41.4489, 42.4854, 43.4965, 44.5346, 45.5716, 46.7301, 47.2251, 48.4449, 48.8793])
    """
    
    def __new__(self, t_min: float, t_max: float, fs: int, alphas: torch.Tensor = torch.tensor([1]), patterns: bool = False, **kwargs) -> sklearn.base.BaseEstimator:
        """Obtain a new TimeDelayed estimator.
        
        Parameters
        ----------
        t_min : float
            The minimum time delay.
        t_max : float
            The maximum time delay.
        fs : int
            The sampling frequency.
        alphas : Union[torch.Tensor, np.ndarray, float, int], default=1
            The penalties to use for estimation.
        patterns : bool, default=False
            Should patterns be estimated?
        kwargs : Any
            Additional arguments for estimator.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The TimeDelayed estimator.
        """
        
        # check alphas
        if isinstance(alphas, float) | isinstance(alphas, int):
            alphas = torch.tensor([alphas])
        
        if isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        # determine estimator
        if isinstance(alphas, torch.Tensor):
            return _TimeDelayed_torch(t_min, t_max, fs, alphas = alphas, patterns = patterns, **kwargs)
        elif isinstance(alphas, np.ndarray):
            return _TimeDelayed_numpy(t_min, t_max, fs, alphas = alphas, patterns = patterns, **kwargs)
        
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