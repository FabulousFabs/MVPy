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

def check_dims_and_groups_(X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], groups: Optional[Union[List, np.ndarray, torch.Tensor]] = None, dim: Optional[int] = None, on_underspecified: str = 'raise') -> Tuple[int, Union[np.ndarray, torch.Tensor], List, Union[np.ndarray, torch.Tensor]]:
    """Check dimension and groups and create corresponding sets and masks.
    
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
    on_underspecified : {'raise', 'warn', 'ignore'}, default='raise'
        If we detect an underspecified grouping--i.e., not all available predictors are used--to what level should we escalate things?
    
    Returns
    -------
    dim : int
        The dimension along which groups exist.
    group_ids : np.ndarray | torch.Tensor
        Vector containing group identifiers.
    sets : List[Tuple[int]]
        List containing tuples of group identifiers, together forming all combinations.
    masks : np.ndarray | torch.Tensor
        The masks corresponding to each feature combination in sets.
    """
    
    # check torch
    is_torch = isinstance(X, torch.Tensor) & isinstance(y, torch.Tensor)
    
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
            if on_underspecified == 'warn':
                warnings.warn(f'Supplied `groups` do not include all available predictors. Counted {groups.sum(0)} instances.')
            elif on_underspecified == 'ignore':
                pass
            else:
                raise ValueError(f'When supplying `groups`, each predictor must appear in at least one grouping.')

        # check data type
        if is_torch:
            groups = groups.to(torch.bool)
        else:
            groups = groups.astype(bool)
    
    # setup all group combinations
    n_groups = groups.shape[0]
    group_ids = list(range(n_groups))
    sets = list(chain.from_iterable(combinations(group_ids, r) for r in range(1, n_groups + 1)))
    masks = []
    
    # setup masks
    for i, set_i in enumerate(sets):
        # create empty mask
        if is_torch:
            mask = groups[0,:].clone()
        else:
            mask = groups[0,:].copy()
        mask[:] = False
        
        # bit-combine current masks
        for group_i in set_i:
            mask = mask | groups[group_i]
        
        # put on stack
        masks.append(mask)
    
    # remove duplicates (if any)
    if is_torch:
        # make unique through bit representation
        masks = torch.stack(masks, dim = 0)
        w = (2 ** torch.arange(masks.shape[1], device = X.device))[None,:]
        tmp = (masks * w).sum(1).long()
        
        # grab indices
        val, inv = torch.unique(tmp, return_inverse = True)
        pos = torch.arange(tmp.numel(), device = tmp.device)
        first_idx = torch.full((val.numel(),), tmp.numel(), device = tmp.device)
        first_idx.scatter_reduce_(0, inv, pos, reduce = 'amin', include_self = False)
        
        # order appropriately
        first_idx = first_idx.sort().values
        
        # mask out
        masks = masks[first_idx]
        sets = [sets[i] for i in first_idx]
    else:
        # make unique through bit representation
        masks = np.array(masks)
        w = (2 ** np.arange(masks.shape[1]))[None,:]
        tmp = (masks * w).sum(1).astype(int)
        
        # grab indices
        _, first_idx = np.unique(tmp, return_index = True)
        
        # order appropriately
        first_idx.sort()
        
        # mask out
        masks = masks[first_idx]
        sets = [sets[i] for i in first_idx]
    
    return dim, group_ids, sets, masks

def fit_validator_(validator: Validator, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], mask: Union[np.ndarray, torch.Tensor], dim: int, i: int) -> Dict:
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
    i : int
        The identifier to return.
    
    Returns
    -------
    results : Dict
        A dictionary containing the results:
            i : int
                The identifier for this validator.
            validator : mvpy.crossvalidation.Validator
                The fitted validator.
            score : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
                The score of the validator.
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
    
    return dict(
        i = i,
        validator = validator,
        score = validator.score_
    )

class Hierarchical:
    """Implements a hierarchical scoring procedure over all feature permutations in :math:`X` describing :math:`y`.
    
    When modeling outcomes :math:`y`, a common question to ask is what specific combination
    of predictors in :math:`X` explains the observed data best. One way to tackle this question
    is to iteratively cross-validate the scoring of predictions :math:`\\hat{y}` from each 
    possible feature combination in :math:`X`. For example, if we have three features in 
    :math:`X`, we would model y as a function of feature combinations :math:`\\left[(0), (1), (2), (0, 1), (0, 2), (1, 2), (0, 1, 2)\\right]`
    such that we can now compare how well each individual predictor and combination of 
    predictors explain the data.
    
    This class implements precisely this hierarchical modeling procedure, but allows creation
    of groups of predictors. For example, we might have several predictors in :math:`X` that,
    together, form some kind of baseline. We might then specify:
    
    .. math::
        G = \\begin{bmatrix}
            1 & 1 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & 0 & 1
        \\end{bmatrix}
    
    to signal that there are three total groups of predictors that we want to permute together,
    the first one including predictors 1-3, and the following two including one novel predictor 
    each. :py:class:`~mvpy.model_selection.Hierarchical` would now compute :math:`2^3 - 1` 
    combinations instead of the full :math:`2^6 - 1` combinations. As described before, this
    yields the feature combinations :math:`\\left[(0), (1), (2), (0, 1), (0, 2), (1, 2), (0, 1, 2)\\right]`
    where feature :math:`(0,)` groups predictors :math:`\\{0, 1, 2\\}`.
    
    Observe, however, that this now means that the permutations include those permutations where
    the baseline predictors are not included in all other models--for example, :math:`(1,)` which
    would evaluate to :math:`[0, 0, 0, 1, 0]`. If we want to enforce that all models include the 
    baseline, we should make them part of every other group:
    
    .. math::
        G = \\begin{bmatrix}
            1 & 1 & 1 & 0 & 0 \\\\
            1 & 1 & 1 & 1 & 0 \\\\
            1 & 1 & 1 & 0 & 1
        \\end{bmatrix}
    
    The backend will automatically remove duplicates, leaving us with only the contrasts that are
    of interest to us :math:`\\left[(0,), (0, 1), (0, 2), (0, 1, 2)\\right]` or, expressed as
    boolean masks:
    
    .. math::
        M = \\begin{bmatrix}
            1 & 1 & 1 & 0 & 0 \\\\
            1 & 1 & 1 & 1 & 0 \\\\
            1 & 1 & 1 & 0 & 1 \\\\
            1 & 1 & 1 & 1 & 1 \\\\
        \\end{bmatrix}
    
    In code, you can confirm the desired grouping in :py:attr:`~mvpy.model_selection.Hierarchical.group_`,
    the resulting feature combinations in :py:attr:`~mvpy.model_selection.Hierarchical.set_` and the masks
    that were applied in :py:attr:`~mvpy.model_selection.Hierarchial.mask_`.
    
    .. warning::
        This performs :math:`k\\left(2^p - 1\\right)` individual model fits where :math:`k` is the
        total number of cross-validation steps and :math:`p` is the number of unique groups of 
        predictors. For large :math:`p`, this becomes exponentially more expensive to solve. If you
        are interested in the unique contribution of each feature rather than separate estimates for
        all combinations, consider using :py:class:`~mvpy.model_selection.Shapley` instead.
    
    .. warning::
        The default behaviour of this class is to check whether all predictors in :math:`X` appear in
        the group specification :py:attr:`~mvpy.model_selection.Hierarchial.groups` at least once.
        If this is not the case, the class will ``raise`` an exception. If you would like to mutate
        this behaviour to either ignore or warn about these cases only, you may want to supply the 
        corresponding :py:attr:`~mvpy.model_selection.Hierarchial.on_underspecified` value.
    
    .. warning::
        When specifying ``n_jobs`` here, be careful not to specify any number of jobs
        in the model or underlying validator. Otherwise, this will lead to a situation 
        where individual jobs each try to initialise more low-level jobs, severely hurting 
        performance.
    
    Parameters
    ----------
    validator : mvpy.crossvalidation.Validator
        The validator object that should be used in this procedure.
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
    n_jobs : Optional[int], default=None
        How many jobs should be used to parallelise the hierarchical fitting procedure?
    verbose : int | bool, default=False
        Should progress be reported verbosely?
    on_underspecified : {'raise', 'warn', 'ignore'}, default='raise'
        If we detect an underspecified grouping--i.e., not all available predictors are used--to what level should we escalate things?
    validator_ : List[mvpy.crossvalidation.Validator]
        A list of all fitted validators.
    score_ : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
        The hierarchical scores of shape ``(n_sets, n_cv[, ...])`` or a dictionary containing each individual :py:class:`~mvpy.metrics.Metric`.
    mask_ : np.ndarray | torch.Tensor
        A matrix where each row corresponds to one boolean mask used to fit one validator.
    group_ : np.ndarray | torch.Tensor
        A matrix where each row corresponds to the boolean mask for one group.
    group_id_: np.ndarray | torch.Tensor
        A vector containing group identifiers used in sets.
    set_ : List[Tuple[int]]
        A list including all group combinations that were tested.
    
    See also
    --------
    mvpy.model_selection.shapley_score, mvpy.model_selection.Shapley : An alternative scoring method computing unique contributions of each feature rather than the full permutation.
    mvpy.model_selection.hierarchical_score : A shorthand for fitting this class.
    mvpy.crossvalidation.Validator : The cross-validation object required by :py:class:`~mvpy.model_selection.Hierarchical`.
    
    Notes
    -----
    Currently this does not automatically select the best model for you. Instead,
    it will return all scores, leaving further decisions up to you. This is because, 
    for most applications, the scores of all permutations are actually of interest
    and may need to be reported.
    
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
    >>> from mvpy.model_selection import Hierarchical
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
    >>> hierarchical = Hierarchical(validator, verbose = True).fit(
    >>>     X, y,
    >>>     groups = groups
    >>> )
    >>> hierarchical.score_['r2'].shape
    torch.size([4, 5, 64, 400])
    """
    
    def __init__(self, validator: Validator, n_jobs: Optional[int] = None, verbose: Union[int, bool] = False, on_underspecified: str = 'raise', ):
        """Obtain a new hierarchical model selector.
        
        Parameters
        ----------
        validator : mvpy.crossvalidation.Validator
            The validator object that should be used in this procedure.
        n_jobs : Optional[int], default=None
            How many jobs should be used to parallelise the hierarchical fitting procedure?
        verbose : int | bool, default=False
            Should progress be reported verbosely?
        on_underspecified : {'raise', 'warn', 'ignore'}, default='raise'
            If we detect an underspecified grouping--i.e., not all available predictors are used--to what level should we escalate things?
        """
        
        # setup parameters
        self.validator = validator
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.on_underspecified = on_underspecified
        
        # setup internals
        self.validator_ = []
        self.score_ = []
        self.mask_ = []
        self.group_ = []
        self.group_id_ = []
        self.set_ = []
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], groups: Optional[Union[List, np.ndarray, torch.Tensor]] = None, dim: Optional[int] = None) -> "Hierarchical":
        """ Fit all models in a hierarchical manner.
        
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
        hierarchical : mvpy.model_selection.Hierarchical
            The fitted hierarchical model selector.
        """
        
        # check type
        is_torch = isinstance(X, torch.Tensor) & isinstance(y, torch.Tensor)
        
        if (not is_torch) and not (isinstance(X, np.ndarray) & isinstance(y, np.ndarray)):
            raise ValueError(f'Both `X` and `y` must be of type np.ndarray or torch.Tensor, but got {type(X)} and {type(y)}.')
        
        # make dims, sets and groups
        dim, group_ids, sets, masks = check_dims_and_groups_(
            X, y, 
            groups = groups,
            dim = dim,
            on_underspecified = self.on_underspecified
        )
        
        # reset containers
        self.validator_ = []
        self.score_ = [] if self.validator.metric is None or len(self.validator.metric) == 1 else {metric.name: [] for metric in self.validator.metric}
        self.group_ = []
        self.group_id_ = []
        self.mask_ = []
        self.set_ = []
        
        # count tasks
        n_total = len(masks)
        if (self.n_jobs is None) or (self.n_jobs < 2):
            n_total = n_total * (self.validator.cv_n_ + 1)
        
        # setup progressbar 
        with Progressbar(
            enabled = self.verbose, 
            desc = "Hierarchical", 
            total = n_total
        ) as progress_bar:
            # perform cross-validation
            results = [
                Parallel(n_jobs = self.n_jobs)(
                    delayed(fit_validator_)(
                        self.validator,
                        X,
                        y,
                        mask,
                        dim,
                        i
                    )
                    for i, mask in enumerate(masks)
                )
            ][0]
        
        # collect results
        for r_i, result in enumerate(results):
            # grab results
            validator_ = result['validator']
            score_ = result['score']
            i = result['i']

            # stack data
            self.validator_.append(validator_)
            if self.validator.metric is not None and len(self.validator.metric) > 1:
                for metric in self.validator.metric:
                    self.score_[metric.name].append(score_[metric.name])
            else:
                self.score_.append(score_)
            self.mask_.append(masks[i])
            self.set_.append(sets[i])
        
        # set groups
        self.group_ = groups
        self.group_id_ = group_ids
        
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