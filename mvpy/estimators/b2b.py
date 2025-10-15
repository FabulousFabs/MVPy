'''
A collection of estimators for decoding and disentangling features using back2back regression.
'''

import numpy as np
import torch
import sklearn

from .ridgedecoder import _RidgeDecoder_torch, _RidgeDecoder_numpy
from ..preprocessing.scaler import _Scaler_torch, _Scaler_numpy

from typing import Union, Any

class _B2B_numpy(sklearn.base.BaseEstimator):
    """Initialise a new estimator using numpy as our backend.
    
    Parameters
    ----------
    alphas : np.ndarray, default=np.array([1])
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to use a different penalty for each target.
    normalise_decoder : bool, default=True
        Whether to normalise decoder ouputs.
    
    Attributes
    ----------
    alphas : np.ndarray
        The penalties to use for estimation.
    fit_intercept : bool
        Whether to fit an intercept.
    normalise : bool
        Whether to normalise the data
    alpha_per_target : bool
        Whether to use a different penalty for each target.
    normalise_decoder : bool
        Whether to normalise decoder ouputs.
    decoder_ : mvpy.estimators.RidgeDecoder
        The decoder.
    encoder_ : mvpy.estimators.RidgeDecoder
        The encoder.
    scaler_ : mvpy.estimators.Scaler
        The scaler.
    causal_ : np.ndarray
        The causal contribution of each feature.
    pattern_ : np.ndarray
        The decoded patterns.
    """
    
    def __init__(self, alphas: np.ndarray = np.array([1]), fit_intercept: bool = True, normalise: bool = True, alpha_per_target: bool = False, normalise_decoder: bool = True):
        """Initialise a new estimator.
        
        Parameters
        ----------
        alphas : np.ndarray, default=np.array([1])
            The penalties to use for estimation.
        fit_intercept : bool, default=True
            Whether to fit an intercept.
        normalise : bool, default=True
            Whether to normalise the data.
        alpha_per_target : bool, default=False
            Whether to use a different penalty for each target.
        normalise_decoder : bool, default=True
            Whether to normalise decoder ouputs.
        """
        
        # setup opts
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalise = normalise
        self.alpha_per_target = alpha_per_target
        self.normalise_decoder = normalise_decoder

        # setup model
        self.decoder_ = _RidgeDecoder_numpy(
            alpha = alphas, 
            fit_intercept = fit_intercept, 
            normalise = normalise, 
            alpha_per_target = alpha_per_target
        )
        
        self.encoder_ = _RidgeDecoder_numpy(
            alpha = alphas, 
            fit_intercept = fit_intercept, 
            normalise = normalise, 
            alpha_per_target = alpha_per_target
        )
        
        self.scaler_ = _Scaler_numpy(
            with_mean = normalise_decoder, 
            with_std = normalise_decoder
        )
        
        self.causal_ = None
        self.pattern_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_B2B_numpy":
        """Fit the back-to-back estimator.
        
        Parameters
        ----------
        X : np.ndarray
            The data to fit.
        y : np.ndarray
            The targets to fit.
        
        Returns
        -------
        b2b : _B2B_numpy
            The fitted back-to-back model.
        """
        
        # check data
        if len(y.shape) == 1:
            y = y[:,None]
        
        # setup folds
        indc = np.arange(X.shape[0])
        fold_a, fold_b = indc[0:X.shape[0] // 2], indc[X.shape[0] // 2:]
        
        # fit decoder
        self.decoder_.fit(X[fold_a], y[fold_a])
        
        # fit scaler
        self.scaler_.fit(self.decoder_.predict(X[fold_b]))
        
        # fit encoder
        self.encoder_.fit(self.scaler_.transform(self.decoder_.predict(X[fold_b])), y[fold_b])
        
        # compute contributions
        self.causal_ = self.encoder_.coef_.diagonal()
        
        # clone patterns
        self.pattern_ = self.decoder_.pattern_.copy()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            The data to decode.
        
        Returns
        -------
        y_h : np.ndarray
            Predictions from encoder.
        """
        
        # make sure estimator has been fit
        if (self.causal_ is None) or (self.pattern_ is None):
            raise ValueError(f'Estimator has not been fit yet.')
        
        # make predictions
        return self.encoder_.predict(self.scaler_.transform(self.decoder_.predict(X)))
    
    def clone(self) -> "_B2B_numpy":
        """Obtain a clone of the estimator.
        
        Returns
        -------
        b2b : _B2B_numpy
            A clone of the estimator.
        """
        
        return _B2B_numpy(
            alphas = self.alphas, 
            fit_intercept = self.fit_intercept, 
            normalise = self.normalise, 
            alpha_per_target = self.alpha_per_target, 
            normalise_decoder = self.normalise_decoder
        )

class _B2B_torch(sklearn.base.BaseEstimator):
    """Initialise a new estimator using torch as our backend.
    
    Parameters
    ----------
    alphas : torch.Tensor, default=torch.tensor([1])
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to use a different penalty for each target.
    normalise_decoder : bool, default=True
        Whether to normalise decoder ouputs.
    
    Attributes
    ----------
    alphas : torch.Tensor
        The penalties to use for estimation.
    fit_intercept : bool
        Whether to fit an intercept.
    normalise : bool
        Whether to normalise the data
    alpha_per_target : bool
        Whether to use a different penalty for each target.
    normalise_decoder : bool
        Whether to normalise decoder ouputs.
    decoder_ : mvpy.estimators.RidgeDecoder
        The decoder.
    encoder_ : mvpy.estimators.RidgeDecoder
        The encoder.
    scaler_ : mvpy.estimators.Scaler
        The scaler.
    causal_ : torch.Tensor
        The causal contribution of each feature.
    pattern_ : torch.Tensor
        The decoded patterns.
    """
    
    def __init__(self, alphas: torch.Tensor = torch.tensor([1]), fit_intercept: bool = True, normalise: bool = True, alpha_per_target: bool = False, normalise_decoder: bool = True):
        """Initialise a new estimator.
        
        Parameters
        ----------
        alphas : torch.Tensor, default=torch.tensor([1])
            The penalties to use for estimation.
        fit_intercept : bool, default=True
            Whether to fit an intercept.
        normalise : bool, default=True
            Whether to normalise the data.
        alpha_per_target : bool, default=False
            Whether to use a different penalty for each target.
        normalise_decoder : bool, default=True
            Whether to normalise decoder ouputs.
        """
        
        # setup opts
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalise = normalise
        self.alpha_per_target = alpha_per_target
        self.normalise_decoder = normalise_decoder

        # setup model
        self.decoder_ = _RidgeDecoder_torch(
            alpha = alphas, 
            fit_intercept = fit_intercept, 
            normalise = normalise, 
            alpha_per_target = alpha_per_target
        )
        
        self.encoder_ = _RidgeDecoder_torch(
            alpha = alphas, 
            fit_intercept = fit_intercept, 
            normalise = normalise, 
            alpha_per_target = alpha_per_target
        )
        
        self.scaler_ = _Scaler_torch(
            with_mean = normalise_decoder, 
            with_std = normalise_decoder
        )
        
        self.causal_ = None
        self.pattern_ = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "_B2B_torch":
        """Fit the back-to-back estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            The data to fit.
        y : torch.Tensor
            The targets to fit.
        
        Returns
        -------
        b2b : _B2B_torch
            The fitted back-to-back model.
        """
        
        # check data
        if len(y.shape) == 1:
            y = y[:,None]
        
        # setup folds
        indc = torch.arange(X.shape[0], dtype = torch.int32, device = X.device)
        fold_a, fold_b = indc[0:X.shape[0] // 2], indc[X.shape[0] // 2:]
        
        # fit decoder
        self.decoder_.fit(X[fold_a], y[fold_a])
        
        # fit scaler
        self.scaler_.fit(self.decoder_.predict(X[fold_b]))
        
        # fit encoder
        self.encoder_.fit(self.scaler_.transform(self.decoder_.predict(X[fold_b])), y[fold_b])
        
        # compute contributions
        self.causal_ = self.encoder_.coef_.diagonal()
        
        # clone patterns
        self.pattern_ = self.decoder_.pattern_.clone()
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions.
        
        Parameters
        ----------
        X : torch.Tensor
            The data to decode.
        
        Returns
        -------
        y_h : torch.Tensor
            Predictions from encoder.
        """
        
        # make sure estimator has been fit
        if (self.causal_ is None) or (self.pattern_ is None):
            raise ValueError(f'Estimator has not been fit yet.')
        
        # make predictions
        return self.encoder_.predict(self.scaler_.transform(self.decoder_.predict(X)))
    
    def clone(self) -> "_B2B_torch":
        """Obtain a clone of the estimator.
        
        Returns
        -------
        b2b : _B2B_torch
            A clone of the estimator.
        """
        
        return _B2B_torch(
            alphas = self.alphas, 
            fit_intercept = self.fit_intercept, 
            normalise = self.normalise, 
            alpha_per_target = self.alpha_per_target, 
            normalise_decoder = self.normalise_decoder
        )

class B2B(sklearn.base.BaseEstimator):
    """Implements a back-to-back regression to disentangle causal contributions of correlated features.
    
    The back-to-back estimator is a two-step estimator that consists of a decoder and an encoder. 
    Effectively, the idea is to split the data :math:`X` and :math:`y` into two folds, decode all 
    features in fold a, then use predictions from the decoder to encode all true features from all 
    predictions in fold b. Consequently, this allows us to obtain a disentangled estimate of the 
    causal contribution of each feature.
    
    In practice, this is implemented as:

    .. math::

        \\hat{G} = (Y^T Y + \\alpha_Y)^{-1}Y^T X
    
    .. math::
        \\hat{H} = (X^T X + \\alpha_X)^{-1}X^T Y\\hat{G}
    
    where :math:`\\hat{G}` is the decoder and :math:`\\hat{H}` is the encoder, and :math:`\\alpha` 
    are regularisation parameters. Consequently, the diagonal of :math:`\\hat{H}` contains the 
    estimated causal contributions of our features. 
    
    For more information on B2B regression, please see [1]_.
    
    Parameters
    ----------
    alphas : torch.Tensor | np.ndarray, default=torch.tensor([1])
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to use a different penalty for each target.
    normalise_decoder : bool, default=True
        Whether to normalise decoder ouputs.
    
    Attributes
    ----------
    alphas : torch.Tensor | np.ndarray
        The penalties to use for estimation.
    fit_intercept : bool
        Whether to fit an intercept.
    normalise : bool
        Whether to normalise the data
    alpha_per_target : bool
        Whether to use a different penalty for each target.
    normalise_decoder : bool
        Whether to normalise decoder ouputs.
    decoder_ : mvpy.estimators.RidgeDecoder
        The decoder.
    encoder_ : mvpy.estimators.RidgeDecoder
        The encoder.
    scaler_ : mvpy.estimators.Scaler
        The scaler.
    causal_ : torch.Tensor | np.ndarray
        The causal contribution of each feature of shape ``(n_features,)``.
    pattern_ : torch.Tensor | np.ndarray
        The decoded patterns of shape ``(n_channels, n_features)``.
    
    See also
    --------
    mvpy.preprocessing.Scaler : If applied, scalers used in this class.
    mvpy.estimators.RidgeDecoder : Ridge decoders used for the two-step procedure here.
    
    Notes
    -----
    When penalising per target by setting :py:attr:`~mvpy.estimators.B2B.alpha_per_target`
    to ``True``, you may want to consider normalising the decoder by also setting 
    :py:attr:`~mvpy.estimators.B2B.normalise_decoder` to ``True``. This is because otherwise 
    decoder outputs may live on very different scales, potentially distorting the causal 
    estimates per predictor.
    
    Patterns are computed as per [2]_. However, these patterns are not disentangled and 
    may, consequently, be less informative than desired, depending on strength of existing
    correlations.
    
    References
    ----------
    .. [1] King, J.R., Charton, F., Lopez-Paz, D., & Oquab, M. (2020). Back-to-back regression: Disentangling the influence of correlated factors from multivariate observations. NeuroImage, 220, 117028. 10.1016/j.neuroimage.2020.117028
    .. [2] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.D., Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96-110. 10.1016/j.neuroimage.2013.10.067
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import B2B
    >>> ß = torch.normal(0, 1, (2, 60))
    >>> X = torch.normal(0, 1, (100, 2))
    >>> y = X @ ß + torch.normal(0, 1, (100, 60))
    >>> X, y = y, X
    >>> y = torch.cat((y, y.mean(1).unsqueeze(-1) + torch.normal(0, 5, (100, 1))), 1)
    >>> b2b = B2B()
    >>> b2b.fit(X, y).causal_
    tensor([0.4470, 0.4594, 0.0060])
    """
    
    def __new__(self, alphas: Union[torch.Tensor, np.ndarray, float, int] = 1, **kwargs) -> sklearn.base.BaseEstimator:
        """Obtain a new B2B estimator.
        
        Parameters
        ----------
        alphas : torch.Tensor | np.ndarray | float | int, default=1
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The back-to-back estimator.
        """
        
        # check alphas
        if isinstance(alphas, float) | isinstance(alphas, int):
            alphas = torch.tensor([alphas])
        
        if isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        # determine estimator
        if isinstance(alphas, torch.Tensor):
            return _B2B_torch(alphas = alphas, **kwargs)
        elif isinstance(alphas, np.ndarray):
            return _B2B_numpy(alphas = alphas, **kwargs)
        
        raise ValueError(f'Alphas should be of type np.ndarray or torch.tensor, but got {type(alphas)}.')

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> "B2B":
        """Fit the estimator.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The neural data of shape ``(n_trials, n_channels)``.
        y : np.ndarray | torch.Tensor
            The targets of shape ``(n_trials, n_features)``.
        
        Returns
        -------
        b2b : mvpy.estimators.B2B
            The fitted estimator.
        """

        raise NotImplementedError('This method is not implemented in the base class.')

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The neural data of shape ``(n_trials, n_channels)``.
        
        Returns
        -------
        y_h : np.ndarray | torch.Tensor
            The predictions of shape ``(n_trials, n_features)``.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def clone(self) -> "B2B":
        """Clone this class.
        
        Returns
        -------
        b2b : mvpy.estimators.B2B
            The cloned estimator.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')