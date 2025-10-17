import torch
import numpy as np

import sklearn
from sklearn.pipeline import Pipeline
from itertools import chain, combinations

import warnings
from joblib import Parallel, delayed

from ..utilities import Progressbar
from ..crossvalidation import Validator
from .. import metrics

from typing import Union, Tuple, Dict, List, Optional

def fit_validator_(validator: Validator, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], mask: Union[np.ndarray, torch.Tensor], dim: int) -> Validator:
    """Fit an individual validator object.
    
    Parameters
    ----------
    validator : mvpy.crossvalidation.Validator
        The validator object to be fit. Note that this is automatically cloned here.
    X : np.ndarray | torch.Tensor
        The input data of arbitrary shape.
    y : np.ndarray | torch.Tensor
        The output data of arbitrary shape.
    mask : np.ndarray | torch.Tensor
        The mask to apply to the input data.
    dim : int
        The dimension along which to apply the mask.
    
    Returns
    -------
    validator : mvpy.crossvalidation.Validator
        The fitted validator object.
    """
    
    # apply mask
    if isinstance(X, torch.Tensor):
        # on device selecting in torch
        X_i = torch.index_select(
            X, 
            dim = dim, 
            index = torch.nonzero(
                mask, 
                as_tuple = False
            ).squeeze(-1)
        )
    else:
        # otherwise, it must be numpy (as per previous type checks)
        X_i = np.compress(
            mask, 
            X, 
            axis = dim
        )
    
    # clone validator
    validator = validator.clone()
    validator.fit(X_i, y)
    
    return validator

def fit_permutation_(validator: Validator, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], groups: Union[np.ndarray, torch.Tensor], dim: int, i: int) -> Dict:
    """
    """
    
    # check type
    is_torch = isinstance(X, torch.Tensor) & isinstance(y, torch.Tensor)
    
    # we treat group zero as zero set
    if is_torch:
        # make baseline mask
        mask_i = groups[0,:].clone()
        
        # make permuted indices
        perm_i = 1 + torch.randperm(groups.shape[0] - 1, device = X.device)
        sort_i = [0] + (perm_i.argsort() + 1).cpu().tolist()
        sort_i = torch.tensor(sort_i, dtype = torch.long, device = X.device)
        
        # make data permutation
        data_i = torch.randperm(X.shape[0], device = X.device)
    else:
        # make baseline mask
        mask_i = groups[0,:].copy()
        
        # make permuted indices
        perm_i = 1 + np.random.choice(
            np.arange(groups.shape[0] - 1), 
            replace = False, 
            size = (groups.shape[0] - 1,)
        )
        sort_i = perm_i.argsort() + 1
        sort_i = np.concatenate((
            np.array([0]), sort_i
        ))
        
        # make data permutation
        data_i = np.random.choice(
            np.arange(X.shape[0]),
            replace = False,
            size = (X.shape[0],)
        )
    
    # permute the data
    X, y = X[data_i], y[data_i]
    
    # fit zero set
    validator_i = fit_validator_(validator, X, y, mask_i, dim)
    φ_i = validator_i.score_
    
    # setup phi
    is_dict = isinstance(φ_i, Dict)
    φ = [] if not is_dict else {name: [] for name in φ_i}
    
    # setup validators
    validators = []
    
    # add zero-set (for posterity)
    validators.append(validator_i)
    if is_dict:
        for name in φ_i:
            φ[name].append(φ_i[name])
    else:
        φ.append(φ_i)
    
    # fit updates
    for p_i in perm_i:
        # update mask and fit
        mask_i = mask_i | groups[p_i]
        validator_i = fit_validator_(validator, X, y, mask_i, dim)
        validators.append(validator_i)
        φ_j = validator_i.score_
        
        # add scores
        if is_dict:
            # loop over metrics
            for name in φ_i:
                # add delta
                φ[name].append(
                    φ_j[name] - φ_i[name]
                )
        else:
            # add delta
            φ.append(
                φ_j - φ_i
            )
        
        # update phi
        φ_i = φ_j
    
    # sort the scores by group again
    if is_dict:
        # loop over metrics
        for name in φ:
            # stack data
            if is_torch:
                φ[name] = torch.stack(φ[name], dim = 0)
            else:
                φ[name] = np.array(φ[name])
            
            # sort groups
            φ[name] = φ[name][sort_i]
    else:
        # stack data
        if is_torch:
            φ = torch.stack(φ, dim = 0)
        else:
            φ = np.array(φ)
        
        # sort groups
        φ = φ[sort_i]
    
    # sort validators by group again
    sorted_validators = [validators[i] for i in sort_i]
    
    return dict(
        i = i,
        validator = sorted_validators,
        score = φ,
        order = perm_i
    )

class Shapley:
    """Implements a Shapley value scoring procedure over all feature permutations in :math:`X` describing :math:`y`.
    
    When modeling outcomes :math:`y`, a common question to ask is to what degree individual 
    predictors in :math:`X` contribute to :math:`y`. To do this, we group predictors according
    to :py:attr:`~mvpy.model_selection.Shapley.groups` that may, for example, specify:
    
    .. math::
        G = \\begin{bmatrix}
            1 & 1 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & 0 & 1
        \\end{bmatrix}
    
    Or, in other words, here we have three groups of predictors. To compute Shapley values, we
    always assume that group zero (in this case, including the first three predictors), is some
    baseline, relative to which we measure the contribution of other predictors. If no such baseline 
    exists, it should simply be an intercept predictor. Next, we perform :py:attr:`~mvpy.model_selection.Shapley.n_permutations`
    where we fit and score the zero model, then loop over a permutation of our other predictors that
    we add one by one, measuring how they affect the outcome score relative to our last fitted model.
    
    By repeating this procedure many times, we obtain Shapley values for each predictor that represent
    the fair contribution of each predictor to outcome scores, invariant of the order in which they
    may be included. The first score will always correspond to the full baseline performance, whereas
    the others are relative improvements over baseline.
    
    .. warning::
        This performs :math:`n k p` model fits where :math:`n` is the number of permutations, :math:`k` 
        is the total number of cross-validation steps and :math:`p` is the number of unique groups of
        predictors. For large :math:`p` or :math:`n`, this becomes expensive to solve.
    
    .. warning::
        The default behaviour of this class is to check whether all predictors in :math:`X` appear in
        the group specification :py:attr:`~mvpy.model_selection.Shapley.groups` at least once.
        If this is not the case, the class will ``raise`` an exception. If you would like to mutate
        this behaviour to either ignore or warn about these cases only, you may want to supply the 
        corresponding :py:attr:`~mvpy.model_selection.Shapley.on_underspecified` value.
    
    .. warning::
        When specifying ``n_jobs`` here, be careful not to specify any number of jobs
        in the model or underlying validator. Otherwise, this will lead to a situation 
        where individual jobs each try to initialise more low-level jobs, severely hurting 
        performance.
    
    Parameters
    ----------
    validator : mvpy.crossvalidation.Validator
        The validator object that should be used in this procedure.
    n_permutations : int, default=10
        How many permutations should we run? A higher number of permutations yields better estimates. Generally, the higher the number of predictor groups, the higher the number of permutations used.
    n_jobs : Optional[int], default=None
        How many jobs should be used to parallelise the hierarchical fitting procedure?
    verbose : int | bool, default=False
        Should progress be reported verbosely?
    on_underspecified : {'raise', 'warn', 'ignore'}, default='raise'
        If we detect an underspecified grouping--i.e., not all available predictors are used--to what level should we escalate things?
    
    Attributes
    ----------
    validator : mvpy.crossvalidation.Validator
        The validator object that should be used in this procedure.
    n_permutations : int, default=10
        How many permutations should we run? A higher number of permutations yields better estimates. Generally, the higher the number of predictor groups, the higher the number of permutations used.
    n_jobs : Optional[int], default=None
        How many jobs should be used to parallelise the hierarchical fitting procedure?
    verbose : int | bool, default=False
        Should progress be reported verbosely?
    on_underspecified : {'raise', 'warn', 'ignore'}, default='raise'
        If we detect an underspecified grouping--i.e., not all available predictors are used--to what level should we escalate things?
    validator_ : List[List[mvpy.crossvalidation.Validator]]
        A list containing, per permutation, another list containing Validators for each model group, ordered by group identity.
    score_ : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
        The shapley scores of shape ``(n_permutations, n_sets, n_cv[, ...])`` or a dictionary containing each individual :py:class:`~mvpy.metrics.Metric`.
    order_ : np.ndarray | torch.Tensor
        A matrix containing the order in which groups were added to the baseline group of shape ``(n_permutations, n_groups - 1)``.
    
    See also
    --------
    mvpy.model_selection.hierarchical_score, mvpy.model_selection.Hierarchical : An alternative scoring method computing the full permutation over features.
    mvpy.model_selection.shapley_score : A shorthand for fitting this class.
    mvpy.crossvalidation.Validator : The cross-validation object required by :py:class:`~mvpy.model_selection.Shapley`.
    
    Notes
    -----
    All entries of scores are relative to the baseline group, except for, of course, 
    the baseline group itself.
    
    .. warning::
        If multiple values are supplied for ``metric``, this class will
        produce a dictionary of ``{Metric.name: score, ...}`` rather than
        a stacked array. This is to provide consistency across cases where
        metrics may or may not differ in their output shapes.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy import metrics
    >>> from mvpy.dataset import make_meeg_continuous
    >>> from mvpy.preprocessing import Scaler
    >>> from mvpy.estimators import TimeDelayed
    >>> from mvpy.crossvalidation import Validator
    >>> from mvpy.model_selection import Shapley
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
    >>> # setup validator
    >>> validator = Validator(
    >>>     trf,
    >>>     metric = (metrics.r2, metrics.pearsonr),
    >>> )
    >>> # setup groups
    >>> groups = torch.tensor(
    >>>     [
    >>>         [1, 1, 1, 0, 0],
    >>>         [0, 0, 0, 1, 0],
    >>>         [0, 0, 0, 0, 1]
    >>>     ], 
    >>>     dtype = torch.long
    >>> )
    >>> # score individual predictors using Shapley
    >>> shapley = Shapley(validator, n_permutations = 3, verbose = True).fit(
    >>>     X, y,
    >>>     groups = groups
    >>> )
    >>> shapley.score_['r2'].shape
    torch.Size([10, 3, 5, 64, 400])
    """
    
    def __init__(self, validator: Validator, n_permutations: int = 10, n_jobs: Optional[int] = None, verbose: Union[int, bool] = False, on_underspecified: str = 'raise'):
        """ Obtain a new Shapley model selector.
        
        Parameters
        ----------
        validator : mvpy.crossvalidation.Validator
            The validator object that should be used in this procedure.
        n_permutations : int, default=10
            How many permutations should we run? A higher number of permutations yields better estimates. Generally, the higher the number of predictor groups, the higher the number of permutations used.
        n_jobs : Optional[int], default=None
            How many jobs should be used to parallelise the hierarchical fitting procedure?
        verbose : int | bool, default=False
            Should progress be reported verbosely?
        on_underspecified : {'raise', 'warn', 'ignore'}, default='raise'
            If we detect an underspecified grouping--i.e., not all available predictors are used--to what level should we escalate things?
        """
        
        # setup parameters
        self.validator = validator
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.on_underspecified = on_underspecified
        
        # setup internals
        self.validator_ = []
        self.score_ = []
        self.order_ = []
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], groups: Optional[Union[List, np.ndarray, torch.Tensor]] = None, dim: Optional[int] = None) -> "Shapley":
        """ Fit the models to obtain Shapley values.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The input data of arbitray shape.
        y : np.ndarray | torch.Tensor
            The output data of arbitrary shape.
        groups : Optional[List | np.ndarray | torch.Tensor], default=None
            Matrix describing all groups of interest of shape ``(n_groups, n_predictors)``. If ``None``, this will default to the identity matrix of ``(n_predictors, n_predictors)``.
        dim : Optional[int], default=None
            The dimension in :math:`X` that describes the predictors. If ``None``, this will assume ``-1`` for 2D data and ``-2`` otherwise.
        
        Returns
        -------
        shapley : mvpy.model_selection.Shapley
            The fitted shapley model selector.
        """
        
        # check type
        is_torch = isinstance(X, torch.Tensor) & isinstance(y, torch.Tensor)
        
        if (not is_torch) and not (isinstance(X, np.ndarray) & isinstance(y, np.ndarray)):
            raise ValueError(f'Both `X` and `y` must be of type np.ndarray or torch.Tensor, but got {type(X)} and {type(y)}.')
        
        # check dims
        if dim is None:
            if X.ndim > 2:
                dim = -2
            else:
                dim = -1
        
        # check groups
        if groups is None:
            # check type
            if is_torch:
                # make torch identity
                groups = torch.eye(X.shape[dim], dtype = torch.bool, device = X.device)
            else:
                # make numpy identity
                groups = np.eye(X.shape[dim])
        else:
            # convert, if required
            if isinstance(groups, List):
                if is_torch:
                    groups = torch.tensor(groups, dtype = torch.long, device = X.device)
                else:
                    groups = np.array(groups, dtype = int)
            
            # verify grouping length matches dim in X
            if groups.shape[1] != X.shape[dim]:
                raise ValueError(f'Expected `groups` to be of shape (n_groups, n_features) matching n_features in `X`, but got groups={groups.shape} and X={X.shape} with dim={dim}.')
            
            # verify all predictors are grouped at least once
            if not (groups.sum(0) > 0).all():
                if self.on_underspecified == 'warn':
                    warnings.warn(f'Supplied `groups` do not include all available predictors. Counted {groups.sum(0)} instances.')
                elif self.on_underspecified == 'ignore':
                    pass
                else:
                    raise ValueError(f'When supplying `groups`, each predictor must appear in at least one grouping.')
            
            # check data type
            if is_torch:
                groups = groups.to(torch.bool)
            else:
                groups = groups.astype(bool)
        
        # check size
        n_groups = groups.shape[0]
        n_total = self.n_permutations
        if (self.n_jobs is None) or (self.n_jobs < 2):
            n_total = n_total * n_groups * self.validator.cv_n_ + n_total
        
        # setup progressbar 
        with Progressbar(
            enabled = self.verbose, 
            desc = "Shapley", 
            total = n_total
        ) as progress_bar:
            # perform permutation
            results = [
                Parallel(n_jobs = self.n_jobs)(
                    delayed(fit_permutation_)(
                        self.validator,
                        X,
                        y,
                        groups,
                        dim,
                        p_i
                    )
                    for p_i in range(self.n_permutations)
                )
            ][0]
        
        # reset internals
        self.validator_ = []
        self.score_ = [] if self.validator.metric is None or len(self.validator.metric) == 1 else {metric.name: [] for metric in self.validator.metric}
        self.order_ = []
        
        # collect results
        for r_i, result in enumerate(results):
            # grab results
            validator_ = result['validator']
            score_ = result['score']
            order_ = result['order']
            i = result['i']

            # stack data
            self.validator_.append(validator_)
            if self.validator.metric is not None and len(self.validator.metric) > 1:
                for metric in self.validator.metric:
                    self.score_[metric.name].append(score_[metric.name])
            else:
                self.score_.append(score_)
            self.order_.append(order_)
        
        # stack order
        if is_torch:
            self.order_ = torch.stack(order_, dim = 0)
        else:
            self.order_ = np.array(order_)
        
        # check metric
        if self.validator.metric is not None and len(self.validator.metric) > 1:
            # stack within dicts
            for metric in self.validator.metric:
                # check data
                if is_torch:
                    self.score_[metric.name] = torch.stack(self.score_[metric.name], dim = 0)
                else:
                    self.score_[metric.name] = np.array(self.score_[metric.name])
        else:
            # if not dict, stack regularly
            if is_torch:
                self.score_ = torch.stack(self.score_, dim = 0)
            else:
                self.score_ = np.array(self.score_)
        
        return self