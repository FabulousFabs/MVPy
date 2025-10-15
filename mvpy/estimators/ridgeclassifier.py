'''
A collection of estimators for ridge classification.
'''

import numpy as np
import torch
import sklearn

from .ridgedecoder import _RidgeDecoder_numpy, _RidgeDecoder_torch
from .classifier import _Classifier_numpy, _Classifier_torch
from ..preprocessing.labelbinariser import _LabelBinariser_numpy, _LabelBinariser_torch
from .. import metrics

from typing import Union, Dict, Tuple, Optional

class _RidgeClassifier_numpy(sklearn.base.BaseEstimator):
    r"""Implements a ridge classifier with numpy backend.
    
    Parameters
    ----------
    alpha : np.ndarray
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to fit individual alphas per target.
    
    Attributes
    ----------
    alpha : np.ndarray
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to fit individual alphas per target.
    estimator : _RidgeDecoder_numpy
        The ridge estimator.
    binariser_ : _LabelBinariser_numpy
        The label binariser used internally.
    intercept_ : np.ndarray
        The intercepts of the classifier.
    coef_ : np.ndarray
        The coefficients of the classifier.
    pattern_ : np.ndarray
        The patterns of the classifier.
    metric_ : Metric
        The default metric to use.
    """
    
    def __init__(self, alpha: np.ndarray, fit_intercept: bool = True, normalise: bool = True, alpha_per_target: bool = False):
        """Obtain a ridge classifier.
        
        Parameters
        ----------
        alpha : np.ndarray
            The penalties to use for estimation.
        fit_intercept : bool, default=True
            Whether to fit an intercept.
        normalise : bool, default=True
            Whether to normalise the data.
        alpha_per_target : bool, default=False
            Whether to fit individual alphas per target.
        """
        
        # setup opts
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalise = normalise
        self.alpha_per_target = alpha_per_target
        
        # setup estimator
        self.estimator = _RidgeDecoder_numpy(
            self.alpha,
            fit_intercept = self.fit_intercept, 
            normalise = self.normalise, 
            alpha_per_target = self.alpha_per_target
        )
        
        # setup binariser
        self.binariser_ = _LabelBinariser_numpy(
            neg_label = -1.0,
            pos_label = 1.0
        )
        
        # setup attributes
        self.intercept_ = None
        self.coef_ = None
        self.pattern_ = None
        self.metric_ = metrics.accuracy
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RidgeClassifier_numpy":
        """Fit the classifier.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_channels).
        y : np.ndarray
            Label data of shape (n_samples[, n_features]).
        """
        
        # check shape of X
        if len(X.shape) != 2:
            raise ValueError(f'X must be of shape (n_samples, n_features), but got {X.shape}.')

        # check shape of y
        if len(y.shape) == 1:
            y = y[:,None]
        
        # transform labels
        L = self.binariser_.fit_transform(y)
        
        # fit estimator
        self.estimator.fit(X, L)
        
        # copy data
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        self.pattern_ = self.estimator.pattern_
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the decision values for inputs.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_channels).
        
        Returns
        -------
        df : np.ndarray
            Decision values of shape (n_samples, n_classes).
        """
        
        # check fit
        if self.coef_ is None or self.intercept_ is None or self.pattern_ is None:
            raise ValueError(f'The RidgeClassifier has not been fitted yet.')

        # compute decision values
        return self.estimator.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_channels).
        
        Returns
        -------
        y : np.ndarray
            Predicted labels of shape (n_samples, n_features).
        """
        
        # compute decision values
        df = self.decision_function(X)
        
        # prepare y
        y = np.zeros_like(df)
        
        # loop over our features
        for i in range(self.binariser_.n_features_):
            # find start and end of feature
            s = self.binariser_.C_[i]
            e = s + len(self.binariser_.classes_[i])
            
            # find maximum for feature
            idx = np.argmax(df[:,s:e], axis = 1)
            
            # set max values
            y[np.arange(idx.shape[0]),s + idx] = 1.0
        
        # return labels
        return self.binariser_.inverse_transform(y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute the probabilities assigned to each class.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape ``(n_samples, n_channels)``.

        Returns
        -------
        p : np.ndarray
            Predicted class probabilities shape ``(n_samples, n_classes)``.
        
        .. warning::
            Probabilities are computed from ``expit()`` over outputs of
            :py:meth:`~mvpy.estimators.RidgeClassifier.decision_function`.
            Consequently, probability estimates returned by this class 
            are not calibrated.
        """
        
        # compute decision values
        df = self.decision_function(X)

        # compute logistic sigmoid
        p = 1 / (1 + np.exp(-df))
        
        # loop over our features
        for i in range(self.binariser_.n_features_):
            # find start and end of feature
            s = self.binariser_.C_[i]
            e = s + len(self.binariser_.classes_[i])
            
            # normalise within feature
            p[...,s:e] /= p[...,s:e].sum(-1, keepdims = True)
        
        return p
    
    def score(self, X: np.ndarray, y: np.ndarray, metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Make predictions from :math:`X` and score against :math:`y`.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape ``(n_samples, n_channels)``.
        y : np.ndarray
            Output data of shape ``(n_samples, n_features)``.
        metric : Optional[Metric], default=None
            Metric or tuple of metrics to compute. If ``None``, defaults to :py:attr:`~mvpy.estimators.RidgeClassifier.metric_`.
        
        Returns
        -------
        score : np.ndarray | Dict[str, np.ndarray]
            Scores of shape ``(n_features,)`` or, for multiple metrics, a dictionary of metric names and scores of shape ``(n_features,)``.
        
        .. warning::
            If multiple values are supplied for ``metric``, this function will
            output a dictionary of ``{Metric.name: score, ...}`` rather than
            a stacked array. This is to provide consistency across cases where
            metrics may or may not differ in their output shapes.
        """
        
        # check metric
        if metric is None:
            metric = self.metric_
        
        return metrics.score(self, metric, X, y)

    def clone(self) -> "_RidgeClassifier_numpy":
        """Obtain a clone of this class.
        
        Returns
        -------
        clf : _RidgeClassifier_numpy
            The cloned class.
        """
        
        return _RidgeClassifier_numpy(
            self.alphas,
            fit_intercept = self.fit_intercept,
            normalise = self.normalise,
            alpha_per_target = self.alpha_per_target
        )

class _RidgeClassifier_torch(sklearn.base.BaseEstimator):
    r"""Implements a ridge classifier with torch backend.
    
    Parameters
    ----------
    alpha : torch.Tensor
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to fit individual alphas per target.
    
    Attributes
    ----------
    alpha : torch.Tensor
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to fit individual alphas per target.
    estimator : _RidgeDecoder_torch
        The ridge estimator.
    binariser_ : _LabelBinariser_torch
        The label binariser used internally.
    intercept_ : torch.Tensor
        The intercepts of the classifier.
    coef_ : torch.Tensor
        The coefficients of the classifier.
    pattern_ : torch.Tensor
        The patterns of the classifier.
    metric_ : Metric
        The default metric to use.
    """
    
    def __init__(self, alpha: torch.Tensor, fit_intercept: bool = True, normalise: bool = True, alpha_per_target: bool = False):
        """Obtain a ridge classifier.
        
        Parameters
        ----------
        alpha : torch.Tensor
            The penalties to use for estimation.
        fit_intercept : bool, default=True
            Whether to fit an intercept.
        normalise : bool, default=True
            Whether to normalise the data.
        alpha_per_target : bool, default=False
            Whether to fit individual alphas per target.
        """
        
        # setup opts
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalise = normalise
        self.alpha_per_target = alpha_per_target
        
        # setup estimator
        self.estimator = _RidgeDecoder_torch(
            self.alpha,
            fit_intercept = self.fit_intercept, 
            normalise = self.normalise, 
            alpha_per_target = self.alpha_per_target
        )
        
        # setup binariser
        self.binariser_ = _LabelBinariser_torch(
            neg_label = -1.0,
            pos_label = 1.0
        )
        
        # setup attributes
        self.intercept_ = None
        self.coef_ = None
        self.pattern_ = None
        self.metric_ = metrics.accuracy
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "_RidgeClassifier_torch":
        """Fit the classifier.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_channels).
        y : torch.Tensor
            Label data of shape (n_samples[, n_features]).
        """
        
        # check shape of X
        if len(X.shape) != 2:
            raise ValueError(f'X must be of shape (n_samples, n_features), but got {X.shape}.')

        # check shape of y
        if len(y.shape) == 1:
            y = y[:,None]
        
        # transform labels
        L = self.binariser_.fit_transform(y)
        
        # fit estimator
        self.estimator.fit(X, L)
        
        # copy data
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        self.pattern_ = self.estimator.pattern_
        
        return self
    
    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the decision values for inputs.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_channels).
        
        Returns
        -------
        df : torch.Tensor
            Decision values of shape (n_samples, n_classes).
        """
        
        # check fit
        if self.coef_ is None or self.intercept_ is None or self.pattern_ is None:
            raise ValueError(f'The RidgeClassifier has not been fitted yet.')

        # compute decision values
        return self.estimator.predict(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the target.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_channels).
        
        Returns
        -------
        y : torch.Tensor
            Predicted labels of shape (n_samples, n_features).
        """
        
        # compute decision values
        df = self.decision_function(X)
        
        # prepare y
        y = torch.zeros_like(df)
        
        # loop over our features
        for i in range(self.binariser_.n_features_):
            # find start and end of feature
            s = self.binariser_.C_[i].long().item()
            e = s + len(self.binariser_.classes_[i])
            
            # find maximum for feature
            idx = torch.argmax(df[:,s:e], dim = 1)
            
            # set max values
            y[torch.arange(idx.shape[0]),s + idx] = 1.0
        
        # return labels
        return self.binariser_.inverse_transform(y)
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the probabilities assigned to each class.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.

        Returns
        -------
        p : torch.Tensor
            Predicted class probabilities shape ``(n_samples, n_classes)``.
        
        .. warning::
            Probabilities are computed from ``expit()`` over outputs of
            :py:meth:`~mvpy.estimators.RidgeClassifier.decision_function`.
            Consequently, probability estimates returned by this class 
            are not calibrated.
        """
        
        # compute decision values
        df = self.decision_function(X)

        # compute logistic sigmoid
        p = 1 / (1 + torch.exp(-df))
        
        # loop over our features
        for i in range(self.binariser_.n_features_):
            # find start and end of feature
            s = self.binariser_.C_[i].long().item()
            e = s + len(self.binariser_.classes_[i])
                        
            # normalise within feature
            p[...,s:e] = p[...,s:e] / p[...,s:e].sum(-1, keepdim = True)
        
        return p
    
    def score(self, X: torch.Tensor, y: torch.Tensor, metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Make predictions from :math:`X` and score against :math:`y`.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        y : torch.Tensor
            Output data of shape ``(n_samples, n_features)``.
        metric : Optional[Metric], default=None
            Metric or tuple of metrics to compute.  If ``None``, defaults to :py:attr:`~mvpy.estimators.RidgeClassifier.metric_`.
        
        Returns
        -------
        score : torch.Tensor | Dict[str, torch.Tensor]
            Scores of shape ``(n_features,)`` or, for multiple metrics, a dictionary of metric names and scores of shape ``(n_features,)``.
        
        .. warning::
            If multiple values are supplied for ``metric``, this function will
            output a dictionary of ``{Metric.name: score, ...}`` rather than
            a stacked array. This is to provide consistency across cases where
            metrics may or may not differ in their output shapes.
        """
        
        # check metric
        if metric is None:
            metric = self.metric_
        
        return metrics.score(self, metric, X, y)

    def clone(self) -> "_RidgeClassifier_torch":
        """Obtain a clone of this class.
        
        Returns
        -------
        clf : _RidgeClassifier_torch
            The cloned class.
        """
        
        return _RidgeClassifier_torch(
            self.alphas,
            fit_intercept = self.fit_intercept,
            normalise = self.normalise,
            alpha_per_target = self.alpha_per_target
        )

class RidgeClassifier(sklearn.base.BaseEstimator):
    """Implements a linear ridge classifier.
    
    Ridge classifiers effectively frame a classification problem as a simple
    linear ridge regression, mapping from neural data :math:`X` to labels
    :math:`y` through spatial filters :math:`\\beta`:

    .. math::

        y = \\beta X + \\varepsilon\\quad\\textrm{where}\\quad y\\in\\{-1, 1\\}
    
    Consequently, we solve for spatial filters through:
    
    .. math::

        \\arg\\min_{\\beta} \\sum_{i}(y_i - \\beta^TX_i)^2 + \\alpha_\\beta\\lvert\\lvert\\beta\\rvert\\rvert^2
    
    where :math:`\\alpha_\\beta` are the penalties to test in LOO-CV.
    
    This linear filter estimation is extremely convenient for neural decoding 
    because, unlike other decoding approaches such as :py:class:`~mvpy.estimators.SVC`,
    this can be solved extremely efficiently and, for many decoding tasks,
    will perform well.
    
    Parameters
    ----------
    alpha : np.ndarray | torch.Tensor
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to fit individual alphas per target.
    
    Attributes
    ----------
    alpha : np.ndarray | torch.Tensor
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to fit individual alphas per target.
    estimator : mvpy.estimators.RidgeDecoder
        The ridge estimator.
    binariser_ : mvpy.preprocessing.LabelBinariser
        The label binariser used internally.
    intercept_ : np.ndarray | torch.Tensor
        The intercepts of the classifier.
    coef_ : np.ndarray | torch.Tensor
        The coefficients of the classifier.
    pattern_ : np.ndarray | torch.Tensor
        The patterns of the classifier.
    metric_ : mvpy.metrics.Accuracy
        The default metric to use.
    
    Notes
    -----
    By default, this will not allow alpha values to differ between targets. In certain situations, 
    this may be desirable, however. In the multi-class case, it should be carefully evaluated
    whether or not :py:attr:`~mvpy.estimators.RidgeClassifier.alpha_per_target` should be enabled,
    as here it may also hurt decoding performance if penalties are on different scales and 
    :py:attr:`~mvpy.estimators.Classifier.method` is ``OvR``.
    
    Coefficients are transformed to patterns to facilitate interpretation thereof. For more 
    information, please see [1]_.
    
    References
    ----------
    .. [1] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.D., Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96-110. 10.1016/j.neuroimage.2013.10.067
    
    Examples
    --------
    We can either do classification over a single feature, like so:
    
    >>> import torch
    >>> from mvpy.estimators import RidgeClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y = True)
    >>> X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    >>> clf = RidgeClassifier(torch.logspace(-5, 10, 20)).fit(X, y)
    >>> y_h = clf.predict(X)
    >>> mv.math.accuracy(y_h.squeeze(), y)
    tensor(0.8533)
    
    Or we can also do classification over multiple features, like so:
    
    >>> import torch
    >>> from mvpy.estimators import RidgeClassifier
    >>> from sklearn.datasets import make_classification
    >>> X0, y0 = make_classification(n_classes = 3, n_informative = 6)
    >>> X1, y1 = make_classification(n_classes = 4, n_informative = 8)
    >>> X = torch.from_numpy(np.concatenate((X0, X1), axis = -1)).float()
    >>> y = torch.from_numpy(np.stack((y0, y1), axis = -1)).float()
    >>> clf = RidgeClassifier(torch.logspace(-5, 10, 20)).fit(X, y)
    >>> y_h = clf.predict(X)
    >>> mv.math.accuracy(y_h.T, y.T)
    torch.tensor([0.82, 0.75])
    """
    
    def __new__(self, alphas: Union[torch.Tensor, np.ndarray, float, int] = 1, method: str = 'OvR', fit_intercept: bool = True, normalise: bool = True, alpha_per_target: bool = False) -> sklearn.base.BaseEstimator:
        """Obtain a new ridge classifier.
        
        Parameters
        ----------
        alpha : np.ndarray | torch.Tensor
            The penalties to use for estimation.
        method : str, default='OvR'
            Should we solve multiclass problems through one-versus-rest (OvR) or one-versus-one (OvO) classifiers?
        fit_intercept : bool, default=True
            Whether to fit an intercept.
        normalise : bool, default=True
            Whether to normalise the data.
        alpha_per_target : bool, default=False
            Whether to fit individual alphas per target.
        """
        
        # check alphas
        if isinstance(alphas, float) | isinstance(alphas, int):
            alphas = torch.tensor([alphas])
        
        if isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        # create estimator
        if isinstance(alphas, torch.Tensor):
            return _Classifier_torch(
                _RidgeClassifier_torch,
                method = method,
                arguments = [alphas],
                kwarguments = dict(
                    fit_intercept = fit_intercept,
                    normalise = normalise,
                    alpha_per_target = alpha_per_target
                )
            )
        elif isinstance(alphas, np.ndarray):
            return _Classifier_numpy(
                _RidgeClassifier_numpy,
                method = method,
                arguments = [alphas],
                kwarguments = dict(
                    fit_intercept = fit_intercept,
                    normalise = normalise,
                    alpha_per_target = alpha_per_target
                )
            )
        
        raise TypeError(f'`alphas` must be of type torch.Tensor, np.ndarray, float or int (the latter two defaulting to torch.Tensor), but got {type(alphas)}.')
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> sklearn.base.BaseEstimator:
        """Fit the estimator.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The features of shape ``(n_samples, n_channels)``.
        y : np.ndarray | torch.Tensor
            The targets of shape ``(n_samples[, n_features])``.
        
        Returns
        -------
        clf : Classifier
            The classifier.
        """

        raise NotImplementedError(f'Method not implemented in the base class.')
    
    def decision_function(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The features ``(n_samples, n_channels)``.

        Returns
        -------
        df : np.ndarray | torch.Tensor
            The predictions of shape ``(n_samples, n_classes)``.
        """

        raise NotImplementedError(f'Method not implemented in the base class.')

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The features (n_samples, n_channels).
        
        Returns
        -------
        y_h : np.ndarray | torch.Tensor
            The predictions of shape ``(n_samples, n_features)``.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The features ``(n_samples, n_channels)``.

        Returns
        -------
        p : np.ndarray | torch.Tensor
            The predictions of shape ``(n_samples, n_classes)``.
        
        .. warning::
            Probabilities are computed from ``expit()`` over outputs of
            :py:meth:`~mvpy.estimators.RidgeClassifier.decision_function`.
            Consequently, probability estimates returned by this class 
            are not calibrated. See :py:class:`~mvpy.estimators.Classifier` 
            for more information.
        """

        raise NotImplementedError('This method is not implemented in the base class.')
    
    def score(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None) -> Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
        """Make predictions from :math:`X` and score against :math:`y`.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        y : np.ndarray | torch.Tensor
            Output data of shape ``(n_samples, n_features)``.
        metric : Optional[Metric | Tuple[Metric]], default=None
            Metric or tuple of metrics to compute. If ``None``, defaults to :py:attr:`~mvpy.estimators.RidgeClassifier.metric_`.
        
        Returns
        -------
        score : np.ndarray | torch.Tensor | Dict[str, np.ndarray], Dict[str, torch.Tensor]
            Scores of shape ``(n_features,)`` or, for multiple metrics, a dictionary of metric names and scores of shape ``(n_features,)``.
        
        .. warning::
            If multiple values are supplied for ``metric``, this function will
            output a dictionary of ``{Metric.name: score, ...}`` rather than
            a stacked array. This is to provide consistency across cases where
            metrics may or may not differ in their output shapes.
        """
        
        raise NotImplementedError(f'Method not implemented in the base class.')
    
    def clone(self) -> "RidgeClassifier":
        """Clone this class.
        
        Returns
        -------
        clf : RidgeClassifier
            The cloned object.
        """
        
        raise NotImplementedError(f'Method not implemented in the base class.')
    
    def copy(self) -> "RidgeClassifier":
        """Clone this class.
        
        Returns
        -------
        clf : RidgeClassifier
            The cloned object.
        """
        
        return self.clone()