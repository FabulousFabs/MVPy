import torch
import numpy as np

import sklearn
from sklearn.pipeline import Pipeline

from ..crossvalidation import Validator
from .shapley import Shapley
from .. import metrics

from typing import Union, Tuple, Dict, List, Optional, Any

def shapley_score(model: Union[Pipeline, sklearn.base.BaseEstimator], X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], groups: Optional[Union[List, np.ndarray, torch.Tensor]] = None, dim: Optional[int] = None, cv: Union[int, Any] = 5, metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None, return_shapley: bool = True, n_permutations: int = 10, n_jobs: Optional[int] = None, n_jobs_validator: Optional[int] = None, verbose: Union[int, bool] = False, verbose_validator: Union[int, bool] = False) -> Union[np.ndarray, torch.Tensor, Dict, Tuple[Shapley, np.ndarray, torch.Tensor, Dict]]:
    """Implements a shorthand for Shapley scoring over all feature permutations in :math:`X` describing :math:`y`.
    
    This function acts as a shorthand for :py:class:`~mvpy.model_selection.Shapley` 
    where it will automatically create and fit all groups of predictors specified
    in :math:`X` following a shapley procedure. Returns either only the output scores
    or, if ``return_shapley`` is ``True``, both the fitted shapley object and
    the scores in a tuple.
    
    For more information, please consult :py:class:`~mvpy.model_selection.Shapley`.
    
    .. warning::
        This performs :math:`n k p` model fits where :math:`n` is the number of permutations, :math:`k` 
        is the total number of cross-validation steps and :math:`p` is the number of unique groups of
        predictors. For large :math:`p` or :math:`n`, this becomes expensive to solve.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline | sklearn.base.BaseEstimator
        The model to fit and score. Can be either a pipeline or estimator object.
    X : np.ndarray | torch.Tensor
        The input data of arbitrary shape.
    y : np.ndarray | torch.Tensor
        The output data of arbitrary shape.
    groups : Optional[List | np.ndarray | torch.Tensor], default=None
        Matrix describing all groups of interest of shape ``(n_groups, n_predictors)``. If ``None``, this will default to the identity matrix ``(n_predictors, n_predictors)``.
    dim : Optional[int], default=None
        The dimension in :math:`X` that describes the predictors. If ``None``, this will assume ``-1`` for 2D data and ``-2`` otherwise.
    cv : int | Any, default=5
        The cross-validation procedure to follow. Either an object exposing a ``split()`` method, such as :py:class`~mvpy.crossvalidation.KFold`, or an integer specifying the number of folds to use in :py:class:`~mvpy.estimators.KFold`.
    metric : Optional[mvpy.metrics.Metric, Tuple[mvpy.metrics.Metric]], default=None
        The metric to use for scoring. If ``None``, this will default to the ``score()`` method exposed by ``model``.
    return_shapley : bool, default=True
        Should the underlying :py:class:`~mvpy.model_selection.Shapley` object be returned?
    n_permutations : int, default=10
        How many permutations should we run? A higher number of permutations yields better estimates. Generally, the higher the number of predictor groups, the higher the number of permutations used.
    n_jobs : Optional[int], default=None
        How many jobs should be used to parallelise the hierarchical fitting procedure?
    n_jobs_validator : Optional[int], default=None
        How many jobs should be used to parallelise the cross-validation procedure?
    verbose : int | bool, default=False
        Should progress be reported verbosely?
    verbose_validator : int | bool, default=False
        Should progress in individual :py:class:`~mvpy.crossvalidation.Validator` objects be reported verbosely?
    
    Returns
    -------
    shapley : Optional[mvpy.model_selection.Shapley]
        If ``return_shapley`` is ``True``, the underlying :py:class:`~mvpy.model_selection.Shapley` object.
    score : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
        The all hierarchical scores of shape ``(n_permutations, n_sets, n_cv[, ...])`` or a dictionary containing each individual :py:class:`~mvpy.metrics.Metric`.
    
    See also
    --------
    mvpy.model_selection.hierarchical_score, mvpy.model_selection.Hierarchical : An alternative scoring method computing the full permutation of feature combinations.
    mvpy.model_selection.Shapley : The underlying shapley scoring object.
    mvpy.crossvalidation.Validator : The cross-validation objects used in :py:class:`~mvpy.model_selection.Shapley`.
    
    Notes
    -----
    All entries of scores are relative to the baseline group, except for, of course, 
    the baseline group itself.
    
    .. warning::
        If multiple values are supplied for ``metric``, this class will
        produce a dictionary of ``{Metric.name: score, ...}`` rather than
        a stacked array. This is to provide consistency across cases where
        metrics may or may not differ in their output shapes.
    
    .. warning::
        When specifying ``n_jobs`` here, be careful not to specify any number of jobs
        in the model or any ``n_jobs_validator``. Otherwise, this will lead to a situation 
        where individual jobs each try to initialise more low-level jobs, severely hurting 
        performance.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy import metrics
    >>> from mvpy.dataset import make_meeg_continuous
    >>> from mvpy.preprocessing import Scaler
    >>> from mvpy.estimators import TimeDelayed
    >>> from mvpy.model_selection import shapley_score
    >>> from sklearn.pipeline import make_pipeline
    >>> # create dataset
    >>> fs = 200
    >>> X, y = make_meeg_continuous(fs = fs, n_features = 5)
    >>> # setup pipeline for estimation of multivariate temporal response functions
    >>> trf = make_pipeline(
    >>>     Scaler().to_torch(),
    >>>     TimeDelayed(
    >>>         -1.0, 0.0, fs, 
    >>>         alphas = torch.logspace(-5, 5, 10, device = device)
    >>>     )
    >>> )
    >>> # setup groups of predictors
    >>> groups = torch.tensor(
    >>>     [
    >>>         [1, 1, 1, 0, 0],
    >>>         [1, 1, 1, 1, 0],
    >>>         [1, 1, 1, 0, 1]
    >>>     ], 
    >>>     dtype = torch.long,
    >>>     device = device
    >>> )
    >>> # score predictors hierarchically
    >>> shapley, score = shapley_score(
    >>>     trf, X, y, 
    >>>     groups = groups,
    >>>     metric = (metrics.r2, metrics.pearsonr)
    >>>     verbose = True
    >>> )
    >>> score['r2'].shape
    torch.size([10, 4, 5, 64, 400])
    """
    
    # setup validator
    validator = Validator(
        model,
        cv = cv,
        metric = metric,
        n_jobs = n_jobs_validator,
        verbose = verbose_validator
    )
    
    # setup shapley
    shapley = Shapley(
        validator,
        n_permutations = n_permutations,
        n_jobs = n_jobs,
        verbose = verbose
    ).fit(X, y, groups = groups, dim = dim)
    
    # check outputs
    if return_shapley:
        # if desired, return both
        return (shapley, shapley.score_)
    
    # return just scores
    return shapley.score_