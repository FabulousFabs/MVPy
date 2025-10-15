'''
A collection of classes and functions to automatically perform cross-validation.
'''

import torch
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from copy import deepcopy

from joblib import Parallel, delayed

from ..utilities import Progressbar
from .kfold import KFold
from ..metrics import Metric, score

from typing import Optional, Union, Any, Tuple, List, Dict

def fit_model_(model: Union[sklearn.base.BaseEstimator, Pipeline], train: Union[np.ndarray, torch.Tensor], test: Union[np.ndarray, torch.Tensor], X: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]] = None, metric: Optional[Union[Metric, Tuple[Metric]]] = None) -> Dict:
    """Implements a single model fitting and scoring procedure.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline | sklearn.base.BaseEstimator
        The model to fit. Can be either pipeline or estimator object.
    train : np.ndarray | torch.Tensor
        The training data indicies.
    test : np.ndarray | torch.Tensor
        The testing data indices.
    X : np.ndarray | torch.Tensor
        The input data of arbitrary shape.
    y : Optional[np.ndarray | torch.Tensor], default=None
        The outcome data of arbitrary shape.
    metric : Optional[mvpy.metrics.Metric, Tuple[mvpy.metrics.Metric]], default=None
        The metric to use for scoring. If ``None``, this will default to the ``score()`` method exposed by ``model``.
    
    Returns
    -------
    out : Dict
        Output dictionary containing:
            model : sklearn.pipeline.Pipeline | sklearn.base.BaseEstimator
                The fitted model or pipeline.
            score_ : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
                The scores from cross-validation.
            test : np.ndarray | torch.Tensor
                The test indices.
    """
    
    # clone estimator or pipeline
    if hasattr(model, 'clone'):
        model = model.clone()
    else:
        model = deepcopy(model)
    
    # fit model
    fit_args = (X[train], y[train]) if y is not None else (X[train],)
    model.fit(*fit_args)
    
    # score model
    score_args = (X[test], y[test]) if y is not None else (X[test],)
    if metric is not None:
        # if metrics were supplied, call score
        score_i = score(model, metric, *score_args)
    else:
        # otherwise, default to internal scorer
        score_i = model.score(*score_args)
    
    # check metric
    if metric is not None and len(metric) > 1:
        # setup
        score_ = {}
        
        # add to dict
        for metric in metric:
            score_[metric.name] = score_i[metric.name]
    else:
        score_ = score_i
    
    return dict(model = model, score_ = score_, test = test)

class Validator(sklearn.base.BaseEstimator):
    """Implements automated cross-validation and scoring over estimators or pipelines.
    
    This allows for easy cross-validated evaluation of models or pipelines, without having
    to explicitly write the code therefore--a common source of mistakes--and while still
    having access to the underlying fitted pipelines. This allows not only model evaluation
    but also evaluation on, for example, new unseen data or inspection of model parameters.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline | sklearn.base.BaseEstimator
        The model to fit and score. Can be either a pipeline or estimator object.
    cv : int | Any, default=5
        The cross-validation procedure to follow. Either an object exposing a ``split()`` method, such as :py:class:`~mvpy.crossvalidation.KFold`, or an integer specifying the number of folds to use in :py:class:`~mvpy.crossvalidation.KFold`.
    metric : Optional[mvpy.metrics.Metric | Tuple[mvpy.metrics.Metric]], default=None
        The metric to use for scoring. If ``None``, this will default to the ``score()`` method exposed by ``model``.
    n_jobs : Optional[int], default=None
        How many jobs should be used to parallelise the cross-validation procedure?
    verbose : int | bool, default=False
        Should progress be reported verbosely?
    
    Attributes
    ----------
    model : sklearn.pipeline.Pipeline | sklearn.base.BaseEstimator
        The model to fit and score. Can be either a pipeline or estimator object.
    cv : int | Any, default=5
        The cross-validation procedure to follow. Either an object exposing a ``split()`` method, such as :py:class:`~mvpy.crossvalidation.KFold`, or an integer specifying the number of folds to use in :py:class:`~mvpy.crossvalidation.KFold`.
    metric : Optional[mvpy.metrics.Metric | Tuple[mvpy.metrics.Metric]], default=None
        The metric to use for scoring. If ``None``, this will default to the ``score()`` method exposed by ``model``.
    n_jobs : Optional[int], default=None
        How many jobs should be used to parallelise the cross-validation procedure?
    verbose : int | bool, default=False
        Should progress be reported verbosely?
    cv_ : Any
        The instantiated cross-validation object exposing ``split()``.
    cv_n_ : int
        The number of cross-validation steps used by :py:attr:`~mvpy.crossvalidation.Validator.cv_`.
    model_ : List[sklearn.pipeline.Pipeline | sklearn.base.BaseEstimator]
        The models fit during cross-validation.
    score_ : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
        The scores of all models on test data of an arbitrary output shape.
    test_ : List[np.ndarray | torch.Tensor]
        A list of test indices used for scoring.
    
    See also
    --------
    mvpy.crossvalidation.cross_val_score : A shorthand for fitting a :py:class:`~mvpy.crossvalidation.Validator`.

    Notes
    -----
    When trying to access individual functions or attributes within estimators of a pipeline, make sure to
    indicate the pipeline step in ``from_step`` when calling either :py:meth:`~mvpy.crossvalidation.Validator.call`
    or :py:meth:`~mvpy.crossvalidation.Validator.collect`.
    
    .. warning::
        If multiple values are supplied for ``metric``, this function will
        output a dictionary of ``{Metric.name: score, ...}`` rather than
        a stacked array. This is to provide consistency across cases where
        metrics may or may not differ in their output shapes.
    
    .. warning::
        When specifying ``n_jobs`` here, be careful not to specify any number of jobs
        in the model. Otherwise, this will lead to a situation where individual jobs
        each try to initialise more low-level jobs, severely hurting performance.
    
    Examples
    --------
    In the simplest case where we have one estimator, we can do:
    
    >>> import torch
    >>> from mvpy.estimators import ReceptiveField
    >>> from mvpy.crossvalidation import Validator
    >>> ß = torch.tensor([1., 2., 3., 2., 1.])
    >>> X = torch.normal(0, 1, (100, 1, 50))
    >>> y = torch.nn.functional.conv1d(X, ß[None,None,:], padding = 'same')
    >>> y = y + torch.normal(0, 1, y.shape)
    >>> trf = ReceptiveField(-2, 2, 1, alpha = 1e-5)
    >>> validator = Validator(model = trf).fit(X, y)
    >>> validator.score_.shape
    torch.size([5, 1, 50])
    >>> validator.collect('coef_').shape
    torch.size([5, 1, 1, 5])
    >>> X_new = torch.normal(0, 1, (100, 1, 50))
    >>> y_new = torch.nn.functional.conv1d(X, ß[None,None,:], padding = 'same')
    >>> y_new = y_new + torch.normal(0, 1, y_new.shape)
    >>> validator.score(X_new, y_new).shape_
    torch.size([5, 1, 50])

    If we have a pipeline, we may want to do:
    
    >>> import torch
    >>> from mvpy.preprocessing import Scaler
    >>> from mvpy.estimators import ReceptiveField
    >>> from mvpy.crossvalidation import Validator
    >>> from sklearn.pipeline import make_pipeline
    >>> ß = torch.tensor([1., 2., 3., 2., 1.])
    >>> X = torch.normal(0, 1, (100, 1, 50))
    >>> y = torch.nn.functional.conv1d(X, ß[None,None,:], padding = 'same')
    >>> y = y + torch.normal(0, 1, y.shape)
    >>> trf = make_pipeline(
    >>>     Scaler().to_torch(),
    >>>     ReceptiveField(-2, 2, 1, alpha = 1e-5)
    >>> )
    >>> validator = Validator(model = trf).fit(X, y)
    >>> validator.score_.shape
    torch.size([5, 1, 50])
    >>> validator.collect('coef_', from_step = -1).shape
    torch.size([5, 1, 1, 5])
    >>> X_new = torch.normal(0, 1, (100, 1, 50))
    >>> y_new = torch.nn.functional.conv1d(X, ß[None,None,:], padding = 'same')
    >>> y_new = y_new + torch.normal(0, 1, y_new.shape)
    >>> validator.score(X_new, y_new).shape_
    torch.size([5, 1, 50])
    
    Whereas, in the more complicated case of also using :py:class:`~mvpy.estimators.Sliding`, for example, we can do:
    
    >>> import torch
    >>> from mvpy.dataset import make_meeg_categorical
    >>> from mvpy.preprocessing import Scaler
    >>> from mvpy.estimators import Sliding, RidgeClassifier
    >>> from mvpy.crossvalidation import Validator
    >>> from mvpy import metrics
    >>> from sklearn.pipeline import make_pipeline
    >>> y, X = make_meeg_categorical()
    >>> clf = make_pipeline(
    >>>     Scaler().to_torch(),
    >>>     Sliding(
    >>>         RidgeClassifier(
    >>>             torch.logspace(-5, 5, 10)
    >>>         ),
    >>>         dims = (-1,),
    >>>         verbose = True,
    >>>         n_jobs = None
    >>>     )
    >>> )
    >>> validator = Validator(
    >>>     model = clf, 
    >>>     metric = (metrics.accuracy, metrics.roc_auc),
    >>>     verbose = True
    >>> ).fit(X, y)
    >>> validator.score_['roc_auc'].shape
    torch.Size([5, 1, 400])
    >>> validator.call('collect', 'pattern_', from_step = -1).shape
    torch.Size([5, 400, 64, 2])
    """
    
    def __init__(self, model: Union[sklearn.base.BaseEstimator, Pipeline], cv: Union[int, Any] = 5, metric: Optional[Union[Metric, Tuple[Metric]]] = None, n_jobs: Optional[int] = None, verbose: Union[int, bool] = False):
        """Obtain a new Validator.
        
        Parameters
        ----------
        model : sklearn.pipeline.Pipeline | sklearn.base.BaseEstimator
            The model to fit and score. Can be either a pipeline or estimator object.
        cv : int | Any, default=5
            The cross-validation procedure to follow. Either an object exposing a ``split()`` method, such as :py:class:`~mvpy.crossvalidation.KFold`, or an integer specifying the number of folds to use in :py:class:`~mvpy.crossvalidation.KFold`.
        metric : Optional[mvpy.metrics.Metric | Tuple[mvpy.metrics.Metric]], default=None
            The metric to use for scoring. If ``None``, this will default to the ``score()`` method exposed by ``model``.
        n_jobs : Optional[int], default=None
            How many jobs should be used to parallelise the cross-validation procedure?
        verbose : int | bool, default=False
            Should progress be reported verbosely?
        """
        
        # setup parameters
        self.model = model
        self.cv = cv
        self.metric = metric
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # check metric
        if self.metric is None:
            if not hasattr(model, 'score'):
                raise ValueError(f'If no `metric` is specified, `model` must expose a `score()`-method, but none was found.')
        
        if isinstance(self.metric, Metric):
            self.metric = (self.metric,)

        # check cv
        if isinstance(cv, int):
            self.cv_ = KFold(n_splits = self.cv)
        else:
            self.cv_ = cv
        
        if not hasattr(self.cv_, 'split'):
            raise ValueError(f'Expected `cv` to be either an integer or expose a `split()` method.')
        
        if not hasattr(self.cv_, 'n_splits'):
            raise ValueError(f'Expceted `cv` to expose `n_splits`.')

        # check CV number
        self.cv_n_ = self.cv_.n_splits
        
        if hasattr(self.cv_, 'n_repeats'):
            self.cv_n_ = self.cv_n_ * self.cv_.n_repeats
        
        # setup internals
        self.model_ = []
        self.score_ = []
        self.test_ = []
        
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]] = None) -> "Validator":
        """Fit and score the validator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of arbitrary shape.
        y : Optional[np.ndarray | torch.Tensor], default=None
            Output data of arbitrary shape.
        
        Returns
        -------
        validator : mvpy.crossvalidation.Validator
            The fitted validator.
        """
        
        # reset containers
        self.model_ = []
        self.score_ = [] if self.metric is None or len(self.metric) == 1 else {metric.name: [] for metric in self.metric}
        self.test_ = []
        
        # setup arguments for split
        split_args = (X, y) if y is not None else (X,)
        
        # setup progressbar
        context = Progressbar(enabled = self.verbose, desc = "Cross-validating...", total = self.cv_n_)
        with context as progress_bar:
            # perform cross-validation
            results = [
                Parallel(n_jobs = self.n_jobs)(
                    delayed(fit_model_)(
                        self.model,
                        train,
                        test,
                        X,
                        y = y,
                        metric = self.metric
                    )
                    for f_i, (train, test) in enumerate(self.cv_.split(*split_args))
                )
            ][0]
        
        # collect results
        for r_i, result in enumerate(results):
            # grab results
            model = result['model']
            score_ = result['score_']
            test = result['test']

            # stack data
            self.model_.append(model)
            if self.metric is not None and len(self.metric) > 1:
                for metric in self.metric:
                    self.score_[metric.name].append(score_[metric.name])
            else:
                self.score_.append(score_)
            self.test_.append(test)
        
        # check metric
        if self.metric is not None and len(self.metric) > 1:
            # stack within dicts
            for metric in self.metric:
                # check data
                if isinstance(X, torch.Tensor):
                    self.score_[metric.name] = torch.stack(self.score_[metric.name], dim = 0)
                else:
                    self.score_[metric.name] = np.array(self.score_[metric.name])
        else:
            # if not dict, stack regularly
            if isinstance(X, torch.Tensor):
                self.score_ = torch.stack(self.score_, dim = 0)
            else:
                self.score_ = np.array(self.score_)
        
        return self
    
    def call(self, method: str, *args: Any, from_step: Optional[int] = None, **kwargs: Any) -> Union[np.ndarray, torch.Tensor]:
        """Call ``method`` for all fitted estimators, pipelines or specific estimators within a pipeline.
        
        Parameters
        ----------
        method : str
            The method to call.
        *args : any
            Additional arguments to pass to the method.
        from_step : Optional[int], default=None
            If not ``None`` and model is a pipeline, which estimator within that pipeline should the method be called from?
        **kwargs : Any
            Additional keyword arguments to pass to the method.
        
        Returns
        -------
        out : np.ndarray | torch.Tensor
            Stacked outputs from method call of shape ``(cv_n_[, ...])``.
        """
        
        # check fit
        if len(self.model_) == 0:
            raise ValueError(f'Validator has not been fitted yet.')
        
        # check type of model
        if isinstance(self.model_[0], Pipeline):
            # in a pipeline, we enable the from_step argument
            if from_step is not None:
                # check method in step
                if not hasattr(self.model_[0][from_step], method):
                    raise ValueError(f'Estimator at position {from_step} in Pipeline does not expose {method}().')

                # collect
                out = [
                    getattr(self.model_[i][from_step], method)(*args, **kwargs)
                    for i in range(len(self.model_))
                ]
            else:
                # check method in pipeline
                if not hasattr(self.model_[0], method):
                    raise ValueError(f'Pipeline does not expose {method}().')
                
                # collect
                out = [
                    getattr(self.model_[i], method)(*args, **kwargs)
                    for i in range(len(self.model_))
                ]
        else:
            # check method in estimator
            if not hasattr(self.model_[0], method):
                raise ValueError(f'Model does not expose {method}().')
            
            # collect
            out = [
                getattr(self.model_[i], method)(*args, **kwargs)
                for i in range(len(self.model_))
            ]
        
        # stack data
        if isinstance(out[0], torch.Tensor):
            out = torch.stack(out, dim = 0)
        elif isinstance(out[0], np.ndarray):
            out = np.array(out)
        
        return out

    def collect(self, attr: str, from_step: Optional[int] = None) -> Union[List, np.ndarray, torch.Tensor]:
        """Collect ``attr`` from all fitted estimators, pipelines or specific estimators within a pipeline.
        
        Parameters
        ----------
        attr : str
            The attribute to collect.
        from_step : Optional[int], default=None
            If not ``None`` and model is a pipeline, which estimator within that pipeline should the method be called from?
        
        Returns
        -------
        out : np.ndarray | torch.Tensor
            Stacked attributes of shape ``(cv_n_[, ...])``.
        """
        
        # check fit
        if len(self.model_) == 0:
            raise ValueError(f'Validator has not been fitted yet.')
        
        # check type of model
        if isinstance(self.model_[0], Pipeline):
            # in a pipeline, we enable the from_step argument
            if from_step is not None:
                # check attribute in step
                if not hasattr(self.model_[0][from_step], attr):
                    raise ValueError(f'Estimator at position {from_step} in Pipeline does not expose attribute {attr}.')

                # collect
                out = [
                    getattr(self.model_[i][from_step], attr)
                    for i in range(len(self.model_))
                ]
            else:
                # check attribute in pipeline
                if not hasattr(self.model_[0], attr):
                    raise ValueError(f'Pipeline does not expose attribute {attr}.')
                
                # collect
                out = [
                    getattr(self.model_[i], attr)
                    for i in range(len(self.model_))
                ]
        else:
            # check attribute in estimator
            if not hasattr(self.model_[0], attr):
                raise ValueError(f'Model does not expose attribute {attr}.')
            
            # collect
            out = [
                getattr(self.model_[i], attr)
                for i in range(len(self.model_))
            ]
        
        # stack data
        if isinstance(out[0], torch.Tensor):
            out = torch.stack(out, dim = 0)
        elif isinstance(out[0], np.ndarray):
            out = np.array(out)
        
        return out
            
    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Call ``transform`` in all models.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The input data of arbitrary shape.
        
        Returns
        -------
        Z : np.ndarray | torch.Tensor
            The transformed data of shape ``(cv_n_[, ...])``.
        
        See also
        --------
        mvpy.crossvalidation.Validator.call : The underlying call function.
        """
        
        return self.call('transform', X)
    
    def decision_function(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Call ``decision_function`` in all models.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The input data of arbitrary shape.
        
        Returns
        -------
        df : np.ndarray | torch.Tensor
            The decision values of shape ``(cv_n_[, ...])``.
        
        See also
        --------
        mvpy.crossvalidation.Validator.call : The underlying call function.
        """
        
        return self.call('decision_function', X)
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Call ``predict`` in all models.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The input data of arbitrary shape.
        
        Returns
        -------
        y_h : np.ndarray | torch.Tensor
            The predicted output data of shape ``(cv_n_[, ...])``.
        
        See also
        --------
        mvpy.crossvalidation.Validator.call : The underlying call function.
        """
        
        return self.call('predict', X)
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Call ``predict_proba`` in all models.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The input data of arbitrary shape.
        
        Returns
        -------
        p : np.ndarray | torch.Tensor
            The predicted probabilities of shape ``(cv_n_[, ...])``.
        
        See also
        --------
        mvpy.crossvalidation.Validator.call : The underlying call function.
        """
        
        return self.call('predict_proba', X)

    def score(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
        """Score new data in all models according to :py:attr:`~mvpy.crossvalidation.Validator.metric`.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The input data of arbitrary shape.
        y : np.ndarray | torch.Tensor
            The output data of arbitrary shape.
        
        Returns
        -------
        score : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
            The scores of all models in new data where individual entries are now of shape ``(cv_n_[, ...])``.
        """
        
        # setup args
        score_args = (X, y) if y is not None else (X,)
        
        # check if we have a metric
        if self.metric is None:
            return self.call('score', *score_args)

        # otherwise, call score ourselves
        scores_ = [
            score(self.model_[i], self.metric, *score_args)
            for i in range(len(self.model_))
        ]
        
        # check for single metric
        if len(self.metric) == 1:
            # check type
            if isinstance(X, torch.Tensor):
                return torch.stack(scores_, dim = 0)
            else:
                return np.array(scores_)
        
        # otherwise, we have multiple metrics
        out = {metric.name: [] for metric in self.metric}
        
        # sort data
        for score_ in scores_:
            # loop over metrics
            for metric in self.metric:
                out[metric.name].append(score_[metric])
        
        # stack within metric
        for metric in self.metrics:
            # check type
            if isinstance(X, torch.Tensor):
                out[metric.name] = torch.stack(out[metric.name], dim = 0)
            else:
                out[metric.name] = np.array(out[metric.name])
        
        return out