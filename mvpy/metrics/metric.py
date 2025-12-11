'''
Base metric class.
'''

import torch
import numpy as np

from dataclasses import dataclass, replace
from typing import Union, Tuple, Any, Callable, Optional

@dataclass
class Metric:
    """Implements an interface class for mutable metrics of model evaluation.
    
    Generally, when we fit models, we want to be able to evaluate how well they explain
    some facet of our data. Many individual functions for doing so are implemented in 
    :py:mod:`~mvpy.math`. However, depending on the exact dimensionality of our data
    and the exact outcome we want to measure, it is convenient to have automatic control
    over what kinds of data are required for a given metric, what dimensions should be
    considered features, and what exact function should be called.
    
    For example, metric functions in :py:mod:`~mvpy.math` generally expect features to 
    live in the final dimension of a tensor. However, if we have neural data :math:`X`
    of shape ``(n_trials, n_channels, n_timepoints)`` and a semantic embedding :math:`y`
    of shape ``(n_trials, n_features, n_timepoints)`` that we decode using a pipeline
    wrapping :py:class:`~mvpy.estimators.RidgeDecoder` in :py:class:`~mvpy.estimators.Sliding`, 
    the results from calling :py:meth:`~mvpy.estimators.Sliding.predict` will be of shape
    ``(n_trials, n_features, n_timepoints)``. In practice, we often want to compute a metric
    over the first dimension that is returned--i.e., trials. Consequently, we can specify
    a metric where :py:attr:`~mvpy.metrics.Metric.request` is ``('y', 'predict')``, 
    :py:attr:`~mvpy.metrics.Metric.reduce` is ``(0,)``, and :py:attr:`~mvpy.metrics.Metric.f` is 
    ``mvpy.math.r2``. In fact, this metric is already implemented as :py:class:`~mvpy.metrics.R2`, 
    which will conveniently handle all of these steps for us.
    
    At the same time, we may want to use this existing :py:class:`~mvpy.metrics.R2` metric, but 
    may also be interested in how well the embedding geometry was decoded. For this, we may
    want to compute our metric over the feature dimension instead. For such cases, 
    :py:class:`~mvpy.metrics.Metric` exposes :py:meth:`~mvpy.metrics.Metric.mutate`, which allows
    us to obtain a new :py:class:`~mvpy.metrics.Metric` with, for example, a mutated structure
    in :py:attr:`~mvpy.metrics.Metric.reduce` which is ``(1,)``.
    
    Parameters
    ----------
    name : str, default='metric'
        The name of the metric. If models are scored using multiple metrics, the name of any metric will be a key in the resulting output dictionary. See :py:meth:`~mvpy.metrics.score` for more information.
    request : Tuple[{'X', 'y', 'decision_function', 'predict', 'transform', 'predict_proba', str}], default=('y', 'predict')
        A tuple of strings defining what measures are required for computing this metric. Generally, this will first try to find a corresponding method or, alternatively, a corresponding attribute in the class. All requested attributes will be supplied to the metric function in order of request.
    reduce : int | Tuple[int], default=(0,)
        Dimensions that should be reduced when computing this metric. In practice, this means that the specified dimensions will be flattened and moved to the final dimension where :py:mod:`~mvpy.math` expects features to live. See :py:meth:`~mvpy.metrics.score` for more information.
    f : Callable, default=lambda x: x
        The math function that should be used for computing this particular metric.
    
    Attributes
    ----------
    name : str, default='metric'
        The name of the metric. If models are scored using multiple metrics, the name of any metric will be a key in the resulting output dictionary. See :py:meth:`~mvpy.metrics.score` for more information.
    request : Tuple[{'X', 'y', 'decision_function', 'predict', 'transform', 'predict_proba', str}], default=('y', 'predict')
        A tuple of strings defining what measures are required for computing this metric. Generally, this will first try to find a corresponding method or, alternatively, a corresponding attribute in the class. All requested attributes will be supplied to the metric function in order of request.
    reduce : int | Tuple[int], default=(0,)
        Dimensions that should be reduced when computing this metric. In practice, this means that the specified dimensions will be flattened and moved to the final dimension where :py:mod:`~mvpy.math` expects features to live. See :py:meth:`~mvpy.metrics.score` for more information.
    f : Callable, default=lambda x: x
        The math function that should be used for computing this particular metric.
    
    Notes
    -----
    All metrics implemented in this submodule are implemented as dataclasses. Instances of these
    classes are automatically instantiated and are available as snake cased variables. For example,
    for the metric class :py:class:`~mvpy.metrics.Roc_auc`, we can directly access 
    :py:data:`~mvpy.metrics.roc_auc`.
    
    See also
    --------
    mvpy.metrics.score : The function handling the logic before :py:class:`~mvpy.metrics.Metric` is called.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.dataset import make_meeg_continuous
    >>> from mvpy.preprocessing import Scaler
    >>> from mvpy.estimators import TimeDelayed
    >>> from mvpy import metrics
    >>> from sklearn.pipeline import make_pipeline
    >>> fs = 200
    >>> X, y = make_meeg_continuous(fs = fs)
    >>> trf = make_pipeline(
    >>>     Scaler().to_torch(),
    >>>     TimeDelayed(
    >>>         -1.0, 0.0, fs,
    >>>         alphas = torch.logspace(-5, 5, 10)
    >>>     )
    >>> ).fit(X, y)
    >>> scores = metrics.score(
    >>>     trf,
    >>>     (
    >>>         metrics.r2, 
    >>>         metrics.r2.mutate(
    >>>             name = 'r2_time', 
    >>>             reduce = (2,)
    >>>         )
    >>>     ),
    >>>     X, y
    >>> )
    >>> scores['r2'].shape, scores['r2_time'].shape
    (torch.Size([64, 400]), torch.Size([120, 64]))
    """

    # defaults are set for sphinx linkcode to work
    name: str = 'metric'
    request: Tuple[str] = ('y', 'predict')
    reduce: Union[int, Tuple[int]] = (0,)
    f: Callable = lambda x: x
    
    def __call__(self, *args: Any, **kwargs: Any) -> Union[np.ndarray, torch.Tensor]:
        """Compute the desired metric.
        
        Parameters
        ----------
        *args : List[Any]
            Arguments to pass to the metric function.
        **kwargs : Dict[str, Any]
            Keyword arguments to pass to the metric function.
        
        Returns
        -------
        metric : np.ndarray | torch.Tensor
            The computed metric of arbitrary shape.
        """
        
        return self.f(*args, **kwargs)
    
    def mutate(self, name: Optional[str] = None, request: Optional[str] = None, reduce: Optional[Union[int, Tuple[int]]] = None, f: Optional[Callable] = None) -> "Metric":
        """Mutate an existing metric.
        
        Parameters
        ----------
        name : Optional[str], default=None
            If not ``None``, the name of the mutated metric.
        request : Optional[str], default=None
            If not ``None``, the features to request for the mutated metric.
        reduce : Optional[int | Tuple[int]], default=None
            If not ``None``, the dimensions to reduce for the mutated metric.
        f : Optional[Callable], default=None
            If not ``None``, the underlying math function to use for the mutated metric.
        
        Returns
        -------
        metric : Metric
            The mutated metric.
        """
        
        return Metric(
            name = name or self.name,
            request = request or self.request,
            reduce = reduce or self.reduce,
            f = f or self.f
        )