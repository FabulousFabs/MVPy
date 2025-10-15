'''
A collection of classes and functions to automatically perform cross-validation.
'''

import torch
import numpy as np
import sklearn

from .kfold import KFold
from .validator import Validator
from ..metrics import Metric
from sklearn.pipeline import Pipeline

from typing import Optional, Union, Dict, Tuple, Any

def cross_val_score(model: Union[Pipeline, sklearn.base.BaseEstimator], X: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]] = None, cv: Optional[Union[int, Any]] = 5, metric: Optional[Union[Metric, Tuple[Metric]]] = None, return_validator: bool = True, n_jobs: Optional[int] = None, verbose: Union[int, bool] = False) -> Union[np.ndarray, torch.Tensor, Dict, Tuple[Validator, Union[np.ndarray, torch.Tensor, Dict]]]:
    """Implements a shorthand for automated cross-validation scoring over estimators or pipelines.
    
    This function acts as a shorthand for :py:class:`~mvpy.crossvalidation.Validator` 
    where it will automatically create and fit the validator, returning either only
    its output scores or, if ``return_validator`` is ``True``, both the fitted 
    validator object and the scores in a tuple.
    
    For more information, please see :py:class:`~mvpy.crossvalidation.Validator`.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline | sklearn.base.BaseEstimator
        The model to fit and score. Can be either a pipeline or estimator object.
    X : np.ndarray | torch.Tensor
        The input data of arbitrary shape.
    y : Optional[np.ndarray | torch.Tensor], default=None
        The outcome data of arbitrary shape.
    cv : Optional[int | Any], default=5
        The cross-validation procedure to follow. Either an object exposing a ``split()`` method, such as :py:class:`~mvpy.crossvalidation.KFold` or an integer specifying the number of folds to use in :py:class:`~mvpy.crossvalidation.KFold`.
    metric : Optional[mvpy.metrics.Metric, Tuple[mvpy.metrics.Metric]], default=None
        The metric to use for scoring. If ``None``, this will default to the ``score()`` method exposed by ``model``.
    return_validator : bool, default=True
        Should the underlying validator object be returned?
    n_jobs : Optional[int], default=None
        How many jobs should be used to parallelise the cross-validation procedure?
    verbose : int | bool, default=False
        Should progress be reported verbosely?
    
    Returns
    -------
    validator : Optional[mvpy.crossvalidation.Validator]
        If ``return_validator`` is ``True``, the underlying validator object.
    score : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
        The scores from cross-validation of arbitrary output shape.
    
    .. warning::
        If multiple values are supplied for ``metric``, this function will
        output a dictionary of ``{Metric.name: score, ...}`` rather than
        a stacked array. This is to provide consistency across cases where
        metrics may or may not differ in their output shapes.
    
    .. warning::
        When specifying ``n_jobs`` here, be careful not to specify any number of jobs
        in the model. Otherwise, this will lead to a situation where individual jobs
        each try to initialise more low-level jobs, severely hurting performance.
    
    See also
    --------
    mvpy.crossvalidation.Validator : The underlying Validator class.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import ReceptiveField
    >>> from mvpy.crossvalidation import cross_val_score
    >>> ß = torch.tensor([1., 2., 3., 2., 1.])
    >>> X = torch.normal(0, 1, (100, 1, 50))
    >>> y = torch.nn.functional.conv1d(X, ß[None,None,:], padding = 'same')
    >>> y = y + torch.normal(0, 1, y.shape)
    >>> trf = ReceptiveField(-2, 2, 1, alpha = 1e-5)
    >>> validator, scores = cross_val_score(trf, X, y)
    >>> scores.mean()
    tensor(0.9432)
    """
    
    # setup and fit validator
    validator = Validator(
        model = model,
        cv = cv,
        metric = metric,
        n_jobs = n_jobs,
        verbose = verbose
    ).fit(X, y)
    
    # check return value
    if return_validator:
        # if desired, return both
        return validator, validator.score_

    # otherwise, return only scores
    return validator.score_
    