'''
A collection of estimators for decoding features using ridge classifiers.
'''

import numpy as np
import torch
import sklearn
import scipy

from ..preprocessing.labelbinariser import _LabelBinariser_numpy, _LabelBinariser_torch
from .. import metrics
from ..utilities import compile

from typing import Union, Any, Dict, List, Tuple, Optional

def _compute_Qr_numpy(Q: np.ndarray, r: np.ndarray, p_i: np.ndarray, p_k: np.ndarray, j_i: np.ndarray, k_i: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Q and r for current iteration.
    
    Parameters
    ----------
    Q : np.ndarray
        Nulled tensor of shape (n_samples, n_classes, n_classes).
    r : np.ndarray
        Nulled tensor of shape (n_samples, n_classes)
    p_i : np.ndarray
        Raw probability estimates of shape (n_samples, n_classes, n_classes).
    p_k : np.ndarray
        Current probability estimates of shape (n_samples, n_classes, n_classes).
    j_i : np.ndarray
        Lower triangle indices of classifiers.
    k_i : np.ndarray
        Lower triangle indices of classifiers.
    eps : float
        Epsilon to use for computations.
    
    Returns
    -------
    Q : np.ndarray
        Updated tensor of shape (n_samples, n_classes, n_classes).
    r : np.ndarray
        Updated tensor of shape (n_samples, n_classes).
    """
    
    # get dims
    N, K = p_i.shape[0], p_i.shape[1]
    batch = np.arange(N)[:,None]
    
    # setup data
    pij = p_i[:,j_i,k_i]
    pji = p_i[:,k_i,j_i]
    w = 1.0 / np.clip(pij * pji, a_min = eps, a_max = None)
    denom = p_k[:,j_i] + p_k[:,k_i] + eps
    denom2 = denom ** 2
    
    # compute contributions
    v_j = w * pji / denom2
    v_k = w * pij / denom2
    rr_j = w * (pij - p_k[:, j_i] / denom)
    rr_k = w * (pji - p_k[:, k_i] / denom)
    
    # accumulate contributions relative to j
    np.add.at(Q, (batch, j_i[None,:], j_i[None,:]), v_j)
    np.add.at(Q, (batch, j_i[None,:], k_i[None,:]), -v_j)
    np.add.at(r, (batch, j_i[None,:]), rr_j)
    
    # accumulate contributions relative to k
    np.add.at(Q, (batch, k_i[None,:], k_i[None,:]), v_k)
    np.add.at(Q, (batch, k_i[None,:], j_i[None,:]), -v_k)
    np.add.at(r, (batch, k_i[None,:]), rr_k)
    
    return Q, r

def _pairwise_coupling_numpy(p_i: np.ndarray, j_i: np.ndarray, k_i: np.ndarray, max_iter: int = 100, eps: float = 1e-12, tol: float = 1e-10) -> np.ndarray:
    """Compute multiclass probabilities in OvO classifiers using Wu-Lin coupling.
    
    Parameters
    ----------
    p_i : np.ndarray
        Raw probability estimates of shape (n_samples, n_classes, n_classes).
    j_i : np.ndarray
        Lower triangle indices of classifiers.
    k_i : np.ndarray
        Lower triangle indices of classifiers.
    max_iter : int
        Maximum number of iterations to perform.
    eps : float
        Epsilon to use for computations.
    tol : float
        Tolerance of the stopping criterion.
    
    Returns
    -------
    p : np.ndarray
        Probability estimates from Wu-Lin coupling of shape (n_samples, n_classes).
    """
    
    # check n_samples and n_classes
    N, K = p_i.shape[0], p_i.shape[1]
    
    # setup outputs
    p_k = np.ones((N, K,)) / K
    
    # setup intermediates
    Q = np.zeros((N, K, K))
    r = np.zeros((N, K,))
    
    # setup constants
    I = np.broadcast_to(np.eye(K)[None,:,:], Q.shape)
    ones = np.ones((N, K, 1))
    zeros = np.zeros((N, 1, 1))
    chance = np.ones_like(p_k) / K

    # loop over iters
    for _ in range(max_iter):
        # reset Q, r
        Q[:] = 0.0
        r[:] = 0.0
        
        # build Q, r
        Q, r = _compute_Qr_numpy(Q, r, p_i, p_k, j_i, k_i, eps = eps)
        
        # place small ridge on Q for stability
        A = Q + 1e-9 * I
        
        # setup equation
        top = np.concatenate([A, ones], axis = 2)
        bottom = np.concatenate([ones.swapaxes(1, 2), zeros], axis = 2)
        KKT = np.concatenate([top, bottom], axis = 1)
        rhs = np.concatenate([r, zeros[:,:,0]], axis = 1)
        
        # solve equation
        delta = np.linalg.solve(KKT, rhs[:,:,None]).squeeze()[:,:K]
        
        # update our probability estimates (clamped to [0, ...])
        p_n = np.clip(p_k + delta, a_min = 0, a_max = None)
        
        # clamp upper end
        s = p_n.sum(axis = 1, keepdims = True)
        p_n = np.where(s > eps, p_n / s, chance)
        
        # check convergence
        if np.max(np.sum(np.abs(p_n - p_k), axis = 1)) < tol:
            # if converged, update outputs and break
            p_k = p_n
            break
        
        # if not converged, setup next iteration
        p_k = p_n
    
    return p_k

class _Classifier_numpy(sklearn.base.BaseEstimator):
    r"""Implements a wrapper for classifiers with numpy backend.
    
    This class is public facing, but should not generally be used. In essence, it provides a convenient wrapper for classifiers that solves OvR/OvO problems.
    
    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : str, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[Any, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    
    Attributes
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : str, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[Any, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    estimators_ : Union[sklearn.base.BaseEstimator, List[sklearn.base.BaseEstimator]]
        All instances of the estimator class (only of type list if OvO).
    binariser_ : LabelBinariser
        Label binariser used internally.
    coef_ : np.ndarray
        If available, coefficients from all classifiers ([n_classifiers,], n_channels, n_classes).
    intercept_ : np.ndarray
        If available, intercepts from all classifiers ([n_classifiers,], n_classes).
    pattern_ : np.ndarray
        If available, patterns from all classifiers ([n_classifiers,], n_channels, n_classes).
    offsets_ : np.ndarray
        Numerical offsets for each feature in outputs, used internally.
    metric_ : mvpy.metrics.Metric
        The default metric to use.
    """
    
    def __init__(self, estimator: sklearn.base.BaseEstimator, method: str = 'OvR', arguments: List[Any] = [], kwarguments: Dict[Any, Any] = dict()):
        """Obtain a classifier wrapper.
        
        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            The estimator type wrapped by this class.
        method : str, default='OvR'
            For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
        arguments : List[Any], default=[]
            Arguments to pass to the estimator at initialisation.
        kwarguments : Dict[Any, Any], default=dict()
            Keyword arguments to pass to the estimator at initialisation.
        """
        
        # setup args
        self.estimator = estimator
        self.method = method
        self.arguments = arguments
        self.kwarguments = kwarguments
        
        # check method
        if method not in ['OvR', 'OvO']:
            raise ValueError(f'Method `{method}` unknown. Must be [\'OvR\', \'OvO\'].')

        # setup internals
        self.estimators_ = None
        self.binariser_ = _LabelBinariser_numpy()
        self.coef_ = None
        self.intercept_ = None
        self.pattern_ = None
        self.offsets_ = None
        self.metric_ = metrics.accuracy
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Classifier_numpy":
        """Fit the estimator.

        Parameters
        ----------
        X : np.ndarray
            The features of shape (n_samples, n_channels).
        y : np.ndarray
            The targets of shape (n_samples[, n_features]).
        
        Returns
        -------
        clf : _Classifier_numpy
            The classifier.
        """
        
        # check shape
        if len(y.shape) == 1:
            y = y[:, None]
        
        # check method
        if self.method == 'OvR':
            # create single estimator for OvR
            self.estimators_ = self.estimator(*self.arguments, **self.kwarguments)
            self.estimators_.fit(X, y)
            
            # collect
            if hasattr(self.estimators_, 'coef_'):
                self.coef_ = self.estimators_.coef_
            
            if hasattr(self.estimators_, 'intercept_'):
                self.intercept_ = self.estimators_.intercept_
            
            if hasattr(self.estimators_, 'pattern_'):
                self.pattern_ = self.estimators_.pattern_
        else:
            # fit labels
            L = self.binariser_.fit_transform(y)
            
            # create estimator list
            self.estimators_ = []
            self.coef_ = []
            self.intercept_ = []
            self.pattern_ = []
            self.offsets_ = []
            offset = 0
            
            # loop over features
            for i in range(self.binariser_.n_features_):
                # add offset
                self.offsets_.append(offset)
                
                # loop over pairs
                for j in range(self.binariser_.n_classes_[i]):
                    for k in range(self.binariser_.n_classes_[i]):
                        # skip if unnecessary
                        if j <= k: continue
                        
                        # find samples
                        indc_jk = (y[:,i] == self.binariser_.labels_[i][j]) \
                                    | (y[:,i] == self.binariser_.labels_[i][k])
                        indc_jk = indc_jk.squeeze()
                        
                        # fit estimator
                        est_ijk = self.estimator(*self.arguments, **self.kwarguments)
                        est_ijk.fit(X[indc_jk], y[indc_jk,i])
                        self.estimators_.append(est_ijk)
                        
                        # collect
                        if hasattr(est_ijk, 'coef_'):
                            self.coef_.append(est_ijk.coef_)
                        
                        if hasattr(est_ijk, 'intercept_'):
                            self.intercept_.append(est_ijk.intercept_)
                        
                        if hasattr(est_ijk, 'pattern_'):
                            self.pattern_.append(est_ijk.pattern_)

            # stack
            self.coef_ = np.array(self.coef_)
            self.intercept_ = np.array(self.intercept_)
            self.pattern_ = np.array(self.pattern_)
            self.offsets_ = np.array(self.offsets_)
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Predict from the estimator.

        Parameters
        ----------
        X : np.ndarray
            The features (n_samples, n_channels).

        Returns
        -------
        df : np.ndarray
            The predictions of shape (n_samples, n_classes).
        """
        
        # check model fit
        if self.estimators_ is None:
            raise ValueError(f'Classifier must be fit before calling decision function.')
        
        if self.method == 'OvR':
            # compute simple decision function
            return self.estimators_.decision_function(X)
        else:
            df = []
            
            for i in range(len(self.estimators_)):
                df.append(self.estimators_[i].decision_function(X))
            
            return np.array(df)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : np.ndarray
            The features (n_samples, n_channels).
        
        Returns
        -------
        y_h : np.ndarray
            The predictions of shape (n_samples, n_features).
        """
        
        # check model fit
        if self.estimators_ is None:
            raise ValueError(f'Classifier must be fit before calling predict.')
        
        if self.method == 'OvR':
            # for OvR, we can just use the estimator prediction
            return self.estimators_.predict(X)
        else:
            ijk = 0
            y_h = []
            
            # loop over features
            for i in range(self.binariser_.n_features_):
                # setup voting
                votes = []
                
                for j in range(self.binariser_.n_classes_[i]):
                    for k in range(self.binariser_.n_classes_[i]):
                        # skip if unnecessary
                        if j <= k: continue
                        
                        # get prediction
                        votes.append(self.estimators_[ijk].predict(X).squeeze())
                        
                        # move tally
                        ijk += 1
                
                # convert votes
                votes = np.array(votes)
                
                # count votes
                labels = self.binariser_.labels_[i]
                n_counts = np.zeros((votes.shape[1], self.binariser_.n_classes_[i]))
                
                for j, label in enumerate(labels):
                    n_counts[:,j] = (votes == label).sum(axis = 0)
                
                # decide winner
                top = np.argmax(n_counts, axis = 1)
                y_h.append(labels[top])
            
            # convert
            return np.array(y_h).T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute probabilities assigned to each class.

        Parameters
        ----------
        X : np.ndarray
            The features of shape ``(n_samples, n_channels)``.

        Returns
        -------
        p : np.ndarray
            The predictions of shape ``(n_samples, n_classes)``.
        
        .. warning::
            Probabilities are computed from `expit()` over outputs of
            :py:meth:`~mvpy.estimators.Classifier.decision_function` 
            where, for :py:attr:`~mvpy.estimators.Classifier.method` ``OvR``,
            we use Wu-Lin coupling. Consequently, probability estimates
            returned by this class are not calibrated.
        """
        
        # check model fit
        if self.estimators_ is None:
            raise ValueError(f'Classifier must be fit before calling predict.')
        
        # check method
        if self.method == 'OvR':
            # we can simply use the internal function
            return self.estimators_.predict_proba(X)
        
        # otherwise, setup probabilities per sample
        p_ind = 1 / (1 + np.exp(-self.decision_function(X)))
        
        # setup loop
        ijk = 0
        p = []
        
        # loop over features
        for i in range(self.binariser_.n_features_):
            # setup probability matrix of feature
            p_i = np.zeros(
                (X.shape[0], self.binariser_.n_classes_[i], self.binariser_.n_classes_[i])
            )
            
            # OvO estimators are in lower triangle order
            j_i, k_i = np.tril_indices(
                self.binariser_.n_classes_[i],
                k = -1,
            )
            
            # OvO span
            ijk_range = np.arange(ijk, ijk + j_i.shape[0])
            
            # populate matrix efficiently
            p_i[:,k_i,j_i] = p_ind[ijk_range,:,0].T
            p_i[:,j_i,k_i] = p_ind[ijk_range,:,1].T
            
            # move tally on ijk for next feature
            ijk += j_i.shape[0]
            
            # compute pairwise coupling
            eps = 5e-3 / self.binariser_.n_classes_[i]
            max_iter = max(100, self.binariser_.n_classes_[i])
            
            p_k = _pairwise_coupling_numpy(
                p_i, j_i, k_i, 
                max_iter = max_iter, eps = eps, tol = eps
            )
            
            # put probabilities for this feature on stack
            p.append(p_k)
        
        # cat the data such that we have (n_samples, n_features * n_classes)
        p = np.concatenate(p, axis = -1)
        
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
            Metric or tuple of metrics to compute. If ``None``, defaults to :py:attr:`~mvpy.estimators.Classifier.metric_`.
        
        Returns
        -------
        score : np.ndarray
            Scores of shape ``(n_features,)``.
        
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
    
    def clone(self) -> "_Classifier_numpy":
        """Clone this class.
        
        Returns
        -------
        clf : _Classifier_numpy
            The cloned object.
        """
        
        return _Classifier_numpy(
            estimator = self.estimator,
            method = self.method,
            arguments = self.arguments,
            kwarguments = self.kwarguments
        )

@compile.torch()
def _compute_Qr_torch(Q: torch.Tensor, r: torch.Tensor, p_i: torch.Tensor, p_k: torch.Tensor, j_i: torch.Tensor, k_i: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Q and r for current iteration.
    
    Parameters
    ----------
    Q : torch.Tensor
        Nulled tensor of shape (n_samples, n_classes, n_classes).
    r : torch.Tensor
        Nulled tensor of shape (n_samples, n_classes)
    p_i : torch.Tensor
        Raw probability estimates of shape (n_samples, n_classes, n_classes).
    p_k : torch.Tensor
        Current probability estimates of shape (n_samples, n_classes, n_classes).
    j_i : torch.Tensor
        Lower triangle indices of classifiers.
    k_i : torch.Tensor
        Lower triangle indices of classifiers.
    eps : float
        Epsilon to use for computations.
    
    Returns
    -------
    Q : torch.Tensor
        Updated tensor of shape (n_samples, n_classes, n_classes).
    r : torch.Tensor
        Updated tensor of shape (n_samples, n_classes).
    """
    
    # setup data
    pij = p_i[:,j_i,k_i]
    pji = p_i[:,k_i,j_i]
    w = 1.0 / torch.clamp(pij * pji, min = eps)
    denom = p_k[:,j_i] + p_k[:,k_i] + eps
    denom2 = denom ** 2
    
    # compute Q and r updates relative to j
    Q[:,j_i,j_i] += w * pji / denom2
    Q[:,j_i,k_i] -= w * pji / denom2
    r[:,j_i] += w * (pij - p_k[:,j_i] / denom)
    
    # compute Q and r updates relative to k
    Q[:,k_i,k_i] += w * pij / denom2
    Q[:,k_i,j_i] -= w * pij / denom2
    r[:,k_i] += w * (pji - p_k[:,k_i] / denom)
    
    return Q, r

def _pairwise_coupling_torch(p_i: torch.Tensor, j_i: torch.Tensor, k_i: torch.Tensor, max_iter: int = 100, eps: float = 1e-12, tol: float = 1e-10) -> torch.Tensor:
    """Compute multiclass probabilities in OvO classifiers using Wu-Lin coupling.
    
    Parameters
    ----------
    p_i : torch.Tensor
        Raw probability estimates of shape (n_samples, n_classes, n_classes).
    j_i : torch.Tensor
        Lower triangle indices of classifiers.
    k_i : torch.Tensor
        Lower triangle indices of classifiers.
    max_iter : int
        Maximum number of iterations to perform.
    eps : float
        Epsilon to use for computations.
    tol : float
        Tolerance of the stopping criterion.
    
    Returns
    -------
    p : torch.Tensor
        Probability estimates from Wu-Lin coupling of shape (n_samples, n_classes).
    """
    
    # check type and device
    dtype = p_i.dtype
    device = p_i.device
    
    # check n_samples and n_classes
    N, K = p_i.shape[0], p_i.shape[1]
    
    # setup outputs
    p_k = torch.ones((N, K,), dtype = dtype, device = device) / K
    
    # setup intermediates
    Q = torch.zeros((N, K, K), dtype = dtype, device = device)
    r = torch.zeros((N, K,), dtype = dtype, device = device)
    
    # setup constants
    I = torch.eye(K, dtype = dtype, device = device)[None,:,:].expand(*Q.shape)
    ones = torch.ones((N, K, 1), dtype = dtype, device = device)
    zeros = torch.zeros((N, 1, 1), dtype = dtype, device = device)
    chance = torch.full_like(p_k, 1.0 / K)

    # loop over iters
    for _ in range(max_iter):
        # reset Q, r
        Q[:] = 0.0
        r[:] = 0.0
        
        # build Q, r
        Q, r = _compute_Qr_torch(Q, r, p_i, p_k, j_i, k_i, eps = eps)
        
        # place small ridge on Q for stability
        A = Q + 1e-9 * I
        
        # setup equation
        top = torch.cat([A, ones], dim = 2)
        bottom = torch.cat([ones.transpose(1, 2), zeros], dim = 2)
        KKT = torch.cat([top, bottom], dim = 1)
        rhs = torch.cat([r, zeros[:,:,0]], dim = 1)
        
        # solve equation
        delta = torch.linalg.solve(KKT, rhs.unsqueeze(2)).squeeze(2)[:,:K]
        
        # update our probability estimates (clamped to [0, ...])
        p_n = torch.clamp(p_k + delta, min = 0)
        
        # clamp upper end
        s = p_n.sum(dim = 1, keepdim = True)
        p_n = torch.where(s > eps, p_n / s, chance)
        
        # check convergence
        if torch.max(torch.sum(torch.abs(p_n - p_k), dim = 1)) < tol:
            # if converged, update outputs and break
            p_k = p_n
            break
        
        # if not converged, setup next iteration
        p_k = p_n
    
    return p_k

class _Classifier_torch(sklearn.base.BaseEstimator):
    """Implements a wrapper for classifiers with torch backend.
    
    This class is public facing, but should not generally be used. In essence, it provides a convenient wrapper for classifiers that solves OvR/OvO problems.
    
    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : str, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[Any, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    
    Attributes
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : str, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[Any, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    estimators_ : Union[sklearn.base.BaseEstimator, List[sklearn.base.BaseEstimator]]
        All instances of the estimator class (only of type list if OvO).
    binariser_ : LabelBinariser
        Label binariser used internally.
    coef_ : torch.Tensor
        If available, coefficients from all classifiers ([n_classifiers,], n_channels, n_classes).
    intercept_ : torch.Tensor
        If available, intercepts from all classifiers ([n_classifiers,], n_classes).
    pattern_ : torch.Tensor
        If available, patterns from all classifiers ([n_classifiers,], n_channels, n_classes).
    offsets_ : torch.Tensor
        Numerical offsets for each feature in outputs, used internally.
    metric_ : mvpy.metrics.Metric
        The default metric to use.
    """
    
    def __init__(self, estimator: sklearn.base.BaseEstimator, method: str = 'OvR', arguments: List[Any] = [], kwarguments: Dict[Any, Any] = dict()):
        """Obtain a classifier wrapper.
        
        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            The estimator type wrapped by this class.
        method : str, default='OvR'
            For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
        arguments : List[Any], default=[]
            Arguments to pass to the estimator at initialisation.
        kwarguments : Dict[Any, Any], default=dict()
            Keyword arguments to pass to the estimator at initialisation.
        """
        
        # setup args
        self.estimator = estimator
        self.method = method
        self.arguments = arguments
        self.kwarguments = kwarguments
        
        # check method
        if method not in ['OvR', 'OvO']:
            raise ValueError(f'Method `{method}` unknown. Must be [\'OvR\', \'OvO\'].')

        # setup internals
        self.estimators_ = None
        self.binariser_ = _LabelBinariser_torch()
        self.coef_ = None
        self.intercept_ = None
        self.pattern_ = None
        self.offsets_ = None
        self.metric_ = metrics.accuracy
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "_Classifier_torch":
        """Fit the estimator.

        Parameters
        ----------
        X : torch.Tensor
            The features of shape (n_samples, n_channels).
        y : torch.Tensor
            The targets of shape (n_samples[, n_features]).
        
        Returns
        -------
        clf : _Classifier_torch
            The classifier.
        """
        
        # check shape
        if len(y.shape) == 1:
            y = y[:, None]
        
        # check method
        if self.method == 'OvR':
            # create single estimator for OvR
            self.estimators_ = self.estimator(*self.arguments, **self.kwarguments)
            self.estimators_.fit(X, y)
            
            # collect
            if hasattr(self.estimators_, 'coef_'):
                self.coef_ = self.estimators_.coef_
            
            if hasattr(self.estimators_, 'intercept_'):
                self.intercept_ = self.estimators_.intercept_
            
            if hasattr(self.estimators_, 'pattern_'):
                self.pattern_ = self.estimators_.pattern_
        else:
            # fit labels
            L = self.binariser_.fit_transform(y)
            
            # create estimator list
            self.estimators_ = []
            self.coef_ = []
            self.intercept_ = []
            self.pattern_ = []
            self.offsets_ = []
            offset = 0
            
            # loop over features
            for i in range(self.binariser_.n_features_):
                # add offset
                self.offsets_.append(offset)
                
                # loop over pairs
                for j in range(self.binariser_.n_classes_[i]):
                    for k in range(self.binariser_.n_classes_[i]):
                        # skip if unnecessary
                        if j <= k: continue
                        
                        # find samples
                        indc_jk = (y[:,i] == self.binariser_.labels_[i][j]) \
                                    | (y[:,i] == self.binariser_.labels_[i][k])
                        indc_jk = indc_jk.squeeze()
                        
                        # fit estimator
                        est_ijk = self.estimator(*self.arguments, **self.kwarguments)
                        est_ijk.fit(X[indc_jk], y[indc_jk,i])
                        self.estimators_.append(est_ijk)
                        
                        # collect
                        if hasattr(est_ijk, 'coef_'):
                            self.coef_.append(est_ijk.coef_)
                        
                        if hasattr(est_ijk, 'intercept_'):
                            self.intercept_.append(est_ijk.intercept_)
                        
                        if hasattr(est_ijk, 'pattern_'):
                            self.pattern_.append(est_ijk.pattern_)

            # stack
            if len(self.coef_) > 0: 
                self.coef_ = torch.stack(self.coef_)
            else:
                self.coef_ = torch.tensor(self.coef_, dtype = X.dtype, device = X.device)
            
            self.intercept_ = torch.stack(self.intercept_)
            
            if len(self.pattern_) > 0: 
                self.pattern_ = torch.stack(self.pattern_)
            else:
                self.pattern = torch.tensor(self.pattern_, dtype = X.dtype, device = X.device)
            
            self.offsets_ = torch.tensor(self.offsets_)
        
        return self
    
    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Predict from the estimator.

        Parameters
        ----------
        X : torch.Tensor
            The features (n_samples, n_channels).

        Returns
        -------
        df : torch.Tensor
            The predictions of shape (n_samples, n_classes).
        """
        
        # check model fit
        if self.estimators_ is None:
            raise ValueError(f'Classifier must be fit before calling decision function.')
        
        if self.method == 'OvR':
            # compute simple decision function
            return self.estimators_.decision_function(X)
        else:
            df = []
            
            for i in range(len(self.estimators_)):
                df.append(self.estimators_[i].decision_function(X))
            
            return torch.stack(df)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            The features (n_samples, n_channels).
        
        Returns
        -------
        y_h : torch.Tensor
            The predictions of shape (n_samples, n_features).
        """
        
        # check model fit
        if self.estimators_ is None:
            raise ValueError(f'Classifier must be fit before calling predict.')
        
        if self.method == 'OvR':
            # for OvR, we can just use the estimator prediction
            return self.estimators_.predict(X)
        else:
            ijk = 0
            y_h = []
            
            # loop over features
            for i in range(self.binariser_.n_features_):
                # setup voting
                votes = []
                
                for j in range(self.binariser_.n_classes_[i]):
                    for k in range(self.binariser_.n_classes_[i]):
                        # skip if unnecessary
                        if j <= k: continue
                        
                        # get prediction
                        votes.append(self.estimators_[ijk].predict(X).squeeze())
                        
                        # move tally
                        ijk += 1
                
                # convert votes
                votes = torch.stack(votes)
                
                # count votes
                labels = self.binariser_.labels_[i]
                n_counts = torch.zeros((votes.shape[1], self.binariser_.n_classes_[i]), dtype = X.dtype, device = X.device)
                
                for j, label in enumerate(labels):
                    n_counts[:,j] = (votes == label).float().sum(0)
                
                # decide winner
                top = torch.argmax(n_counts, dim = 1)
                y_h.append(labels[top])
            
            # convert
            return torch.stack(y_h).T

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Compute probabilities assigned to each class.

        Parameters
        ----------
        X : torch.Tensor
            The features of shape ``(n_samples, n_channels)``.

        Returns
        -------
        p : torch.Tensor
            The predictions of shape ``(n_samples, n_classes)``.
        
        .. warning::
            Probabilities are computed from `expit()` over outputs of
            :py:meth:`~mvpy.estimators.Classifier.decision_function` 
            where, for :py:attr:`~mvpy.estimators.Classifier.method` ``OvR``,
            we use Wu-Lin coupling. Consequently, probability estimates
            returned by this class are not calibrated.
        """
        
        # check model fit
        if self.estimators_ is None:
            raise ValueError(f'Classifier must be fit before calling predict.')
        
        # check method
        if self.method == 'OvR':
            # we can simply use the internal function
            return self.estimators_.predict_proba(X)
        
        # otherwise, setup probabilities per sample
        p_ind = 1 / (1 + torch.exp(-self.decision_function(X)))
        
        # setup loop
        ijk = 0
        p = []
        
        # loop over features
        for i in range(self.binariser_.n_features_):
            # setup probability matrix of feature
            p_i = torch.zeros(
                (X.shape[0], self.binariser_.n_classes_[i], self.binariser_.n_classes_[i]), 
                dtype = X.dtype, device = X.device
            )
            
            # OvO estimators are in lower triangle order
            j_i, k_i = torch.tril_indices(
                self.binariser_.n_classes_[i],
                self.binariser_.n_classes_[i],
                offset = -1,
                device = X.device
            )
            
            # OvO span
            ijk_range = torch.arange(ijk, ijk + j_i.shape[0], device = X.device)
            
            # populate matrix efficiently
            p_i[:,k_i,j_i] = p_ind[ijk_range,:,0].t()
            p_i[:,j_i,k_i] = p_ind[ijk_range,:,1].t()
            
            # move tally on ijk for next feature
            ijk += j_i.shape[0]
            
            # compute pairwise coupling
            eps = 5e-3 / self.binariser_.n_classes_[i]
            max_iter = max(100, self.binariser_.n_classes_[i])
            
            p_k = _pairwise_coupling_torch(
                p_i, j_i, k_i, 
                max_iter = max_iter, eps = eps, tol = eps
            )
            
            # put probabilities for this feature on stack
            p.append(p_k)
        
        # cat the data such that we have (n_samples, n_features * n_classes)
        p = torch.cat(p, dim = -1)
        
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
            Metric or tuple of metrics to compute. If ``None``, defaults to :py:attr:`~mvpy.estimators.Classifier.metric_`.
        
        Returns
        -------
        score : torch.Tensor
            Scores of shape ``(n_features,)``.
        
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
    
    def clone(self) -> "_Classifier_torch":
        """Clone this class.
        
        Returns
        -------
        clf : _Classifier_torch
            The cloned object.
        """
        
        return _Classifier_torch(
            estimator = self.estimator,
            method = self.method,
            arguments = self.arguments,
            kwarguments = self.kwarguments
        )

class Classifier(sklearn.base.BaseEstimator):
    """Implements a wrapper for classifiers that handle one-versus-one (OvO) and one-versus-rest (OvR) classification schemes.
    
    While this class is exposed publically, there are few (if any) direct use 
    cases for this class. In principle, it exists for other classifiers that
    want to handle multi-class cases as OvO or OvR as a wrapper function that
    can either be inherited or created as a super class, specifying the desired
    estimator (recommended option).
    
    One-versus-rest (``OvR``) classification computes the decision functions over
    inputs :math:`X` and then takes the maximum value across decision values
    to predict the most likely classes :math:`\\hat{y}`.
    
    One-versus-one (``OvO``) classification computes all decision functions from
    binary classifiers (e.g., :math:`c_0` vs :math:`c_1`, :math:`c_0` vs :math:`c_2`,
    :math:`c_1` vs :math:`c_2`, ...). For each individual classification problem,
    the maximum value is recorded as one vote for the winning class. Votes are
    then aggregated across all classifiers and the maximum number of votes decides
    the most likely classes :math:`\\hat{y}`.
    
    .. warning::
        When calling :py:meth:`~mvpy.estimators.Classifier.predict_proba`, probabilities
        are computed from ``expit()`` over outputs of :py:meth:`~mvpy.estimators.Classifier.decision_function`.
        While this outputs valid probabilities, they are consequently based on decision
        values that are on arbitrary scales. This may lead to ill-callibrated probability
        estimates. If accurate probability estimates are desired, please consider using 
        :py:class:`~mvpy.estimators.CalibratedClassifier` (to be implemented).
    
    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : {'OvR', 'OvO'}, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[str, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    
    Attributes
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : {'OvR', 'OvO'}, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[str, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    estimators_ : sklearn.base.BaseEstimator | List[sklearn.base.BaseEstimator]
        All instances of the estimator class (only of type list if OvO).
    binariser_ : mvpy.estimators.LabelBinariser
        Label binariser used internally.
    coef_ : np.ndarray | torch.Tensor
        If available, coefficients from all classifiers ``([n_classifiers,] n_channels, n_classes)``.
    intercept_ : np.ndarray | torch.Tensor
        If available, intercepts from all classifiers ``([n_classifiers,] n_classes)``.
    pattern_ : np.ndarray | torch.Tensor
        If available, patterns from all classifiers ``([n_classifiers,] n_channels, n_classes)``.
    offsets_ : np.ndarray | torch.Tensor
        Numerical offsets for each feature in outputs, used internally.
    metric_ : mvpy.metrics.Accuracy
        The default metric to use.
    
    See also
    --------
    mvpy.estimators.RidgeClassifier, mvpy.estimators.SVC : Classifiers that use this class as a wrapper.
    mvpy.preprocessing.LabelBinariser : Label binariser used internally to generated one-hot encodings.
    """
    
    def __init__(self, estimator: sklearn.base.BaseEstimator, method: str = 'OvR', arguments: List[Any] = [], kwarguments: Dict[Any, Any] = dict()):
        """Obtain a classifier wrapper.
        
        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            The estimator type wrapped by this class.
        method : {'OvR', 'OvO}, default='OvR'
            For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
        arguments : List[Any], default=[]
            Arguments to pass to the estimator at initialisation.
        kwarguments : Dict[str, Any], default=dict()
            Keyword arguments to pass to the estimator at initialisation.
        """
        
        # setup args
        self.estimator = estimator
        self.method = method
        self.arguments = arguments
        self.kwarguments = kwarguments
    
    def _get_estimator(self, X: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> sklearn.base.BaseEstimator:
        """Obtain the wrapper and estimator for this SVC.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        y : np.ndarray | torch.Tensor
            Input labels of shape ``(n_samples[, n_features])``.
        
        Returns
        -------
        clf : mvpy.estimators.Classifier
            The classifier.
        """
        
        if isinstance(X, torch.Tensor) & isinstance(y, torch.Tensor):
            return _Classifier_torch(
                self.estimator,
                method = self.method,
                arguments = self.arguments,
                kwarguments = self.kwarguments
            )
        elif isinstance(X, np.ndarray) & isinstance(y, np.ndarray):
            return _Classifier_numpy(
                self.estimator,
                method = self.method,
                arguments = self.arguments,
                kwarguments = self.kwarguments
            )
        
        raise TypeError(f'`X` and `y` must be either torch.Tensor or np.ndarray, but got {type(X)} and {type(y)}.')
    
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
        clf : mvpy.estimators.Classifier
            The classifier.
        """
        
        return self._get_estimator(X, y).fit(X, y)
    
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
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The features ``(n_samples, n_channels)``.
        
        Returns
        -------
        y_h : np.ndarray | torch.Tensor
            The predictions of shape ``(n_samples, n_features)``.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Compute probabilities assigned to each class.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The features of shape ``(n_samples, n_channels)``.

        Returns
        -------
        p : np.ndarray | torch.Tensor
            The predictions of shape ``(n_samples, n_classes)``.
        
        .. warning::
            Probabilities are computed from `expit()` over outputs of
            :py:meth:`~mvpy.estimators.Classifier.decision_function` 
            where, for :py:attr:`~mvpy.estimators.Classifier.method` ``OvR``,
            we use Wu-Lin coupling. Consequently, probability estimates
            returned by this class are not calibrated.
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
            Metric or tuple of metrics to compute. If ``None``, defaults to :py:attr:`~mvpy.estimators.Classifier.metric_`.
        
        Returns
        -------
        score : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
            Scores of shape ``(n_features,)``.
        
        .. warning::
            If multiple values are supplied for ``metric``, this function will
            output a dictionary of ``{Metric.name: score, ...}`` rather than
            a stacked array. This is to provide consistency across cases where
            metrics may or may not differ in their output shapes.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def to_torch(self) -> sklearn.base.BaseEstimator:
        """Obtain the estimator with torch as backend.
        
        Returns
        -------
        clf : mvpy.estimators.classifier._Classifier_torch
            The estimator.
        """
        
        return self._get_estimator(torch.tensor([1.0]), torch.tensor([1.0]))
    
    def to_numpy(self) -> sklearn.base.BaseEstimator:
        """Obtain the estimator with numpy as backend.
        
        Returns
        -------
        clf : mvpy.estimators.classifier._Classifier_numpy
            The estimator.
        """
        
        return self._get_estimator(np.array([1.0]), np.array([1.0]))
    
    def clone(self) -> "Classifier":
        """Clone this class.
        
        Returns
        -------
        clf : Classifier
            The cloned object.
        """
        
        return Classifier(
            self.estimator,
            method = self.method,
            arguments = self.arguments,
            kwarguments = self.kwarguments
        )
    
    def copy(self) -> "Classifier":
        """Clone this class.
        
        Returns
        -------
        clf : Classifier
            The cloned object.
        """
        
        return self.clone()