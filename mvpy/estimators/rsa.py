'''
A collection of estimators for computing
representational similarities.
'''

import sklearn
import numpy as np
import torch

from joblib import Parallel, delayed

from typing import Union, Callable, Any

from ..utilities import Progressbar
from ..math import euclidean

class _RSA_numpy(sklearn.base.BaseEstimator):
    """Implements a representational similarity estimator using numpy backend.
    """
    
    def __init__(self, estimator_: Callable = euclidean, n_jobs_: Union[int, None] = None, verbose_: bool = False):
        """Obtain a representational similarity estimator.
        
        Parameters
        ----------
        estimator_ : Callable, optional
            The estimator to use for computing the RDM (default = mv.math.euclidean).
        n_jobs_ : Union[int, None], optional
            The number of parallel jobs to use (default = None).
        verbose_ : bool, default=False
            Whether to report progress.
        """
        
        self.estimator_ = estimator_
        self.n_jobs_ = n_jobs_
        self.verbose_ = verbose_
        
        self.dims_ = None
        self.cx_ = None
        self.cy_ = None
        self.rdm_ = None

    def fit(self, X: np.ndarray, *args: Any):
        """Fit the estimator (vacant).
        """
        
        # call transform to obtain rdm
        self.transform(X, *args)
        
        return self
    
    def transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        """Transforms the data to representational similarity space.
        
        Parameters
        ----------
        X : np.ndarray
            The data to compute the RDM for (samples x [...] x features [x time])
        args : Any
            Additional arguments to pass to the estimator
        
        Returns
        -------
        rdm : np.ndarray
            The RDM (trials x [...] [x time])
        """
        
        # setup dimensions
        dims = X.shape
        
        # check dimensions
        if len(dims) < 2:
            raise ValueError('`X` must be at least 2-dimensional.')
        
        # check dimensions
        if len(dims) < 3:
            X = X[...,np.newaxis]
        
        # setup dimensions
        self.dims_ = dims = X.shape
        N, F, T = dims[0], dims[-2], dims[-1]
        
        # setup indices
        n = np.arange(N)
        nx, ny = np.meshgrid(n, n)
        self.cx_, self.cy_ = np.triu_indices(N, k = 1)
        i, j = nx.flatten()[self.cx_], ny.T.flatten()[self.cy_]
        
        # compute RDM
        context = Progressbar(enabled = self.verbose_, desc = "Computing RDM...", total = T)
        
        with context as progress_bar:
            self.rdm_ = np.stack(Parallel(n_jobs = self.n_jobs_)
                                    (delayed(self.estimator_)
                                        (X.swapaxes(-2, -1)[i,...,k,:], 
                                        X.swapaxes(-2, -1)[j,...,k,:],
                                        *args)
                                    for k in range(T)),
                                axis = -1)
            
        return self.rdm_
    
    def fit_transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        """Fits the estimator and transforms data to representational similarity space.
        
        Parameters
        ----------
        X : np.ndarray
            The data to compute the RDM for (samples x [...] x features [x time])
        args : any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        rdm : np.ndarray
            The RDM (trials x [...] [x time])
        """
        
        return self.fit(X, *args).transform(X, *args)
    
    def full_rdm(self) -> np.ndarray:
        """Obtain the full RDM for visualisation purposes.
        
        Returns
        -------
        rdm : np.ndarray
            The full RDM (trials x trials [x ... x time])
        """
        
        # check transform
        if (self.dims_ is None) | (self.cx_ is None) | (self.cy_ is None) | (self.rdm_ is None):
            raise RuntimeError('`transform` must be called before `full_rdm`.')
        
        # setup rdm
        rdm = np.zeros((self.dims_[0], self.dims_[0], *self.dims_[1:-2], self.dims_[-1])) * np.nan
        rdm[self.cx_, self.cy_] = rdm[self.cy_, self.cx_] = self.rdm_
        
        return rdm
    
    def clone(self):
        """Obtain a clone of this class.
        
        Returns
        -------
        _RSA_numpy
            The cloned estimator.
        """
        
        return _RSA_numpy(estimator_ = self.estimator_, n_jobs_ = self.n_jobs_, verbose_ = self.verbose_)

class _RSA_torch(sklearn.base.BaseEstimator):
    """Implements representational similarity estimation using torch as our backend.
    """
    
    def __init__(self, estimator_: Callable = euclidean, n_jobs_: Union[int, None] = None, verbose_: bool = False):
        """Obtain a RSA estimator.
        
        Parameters
        ----------
        estimator_ : Callable, optional
            The estimator to use for computing the RDM (default = mv.math.euclidean).
        n_jobs_ : Union[int, None], optional
            The number of parallel jobs to use (default = None).
        verbose_ : bool, default=False
            Whether to print progress.
        """
        
        self.estimator_ = estimator_
        self.n_jobs_ = n_jobs_
        self.verbose_ = verbose_
        
        self.dims_ = None
        self.cx_ = None
        self.cy_ = None
        self.rdm_ = None

    def fit(self, X: torch.Tensor, *args: Any):
        """Fit the estimator. (vacant)
        """
        
        # call transform to obtain rdm
        self.transform(X, *args)
        
        return self
    
    def transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
        """Transform the data to representational similarity space.
        
        Parameters
        ----------
        X : torch.Tensor
            The data to compute the RDM for (samples x [...] x features [x time])
        args : Any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        rdm : torch.Tensor
            The RDM (trials x [...] [x time])
        """
        
        # setup dimensions
        dims = X.shape
        
        # check dimensions
        if len(dims) < 2:
            raise ValueError('`X` must be at least 2-dimensional.')
        
        # check dimensions
        if len(dims) < 3:
            X = X[...,None]
        
        # setup dimensions
        self.dims_ = dims = X.shape
        N, F, T = dims[0], dims[-2], dims[-1]
        
        # setup indices
        n = torch.arange(N)
        nx, ny = torch.meshgrid(n, n, indexing = 'ij')
        self.cx_, self.cy_ = torch.triu_indices(N, N, offset = 1)
        i, j = nx.T.flatten()[self.cx_], ny.flatten()[self.cy_]
        
        # compute RDM
        context = Progressbar(enabled = self.verbose_, desc = "Computing RDM...", total = T)
        
        with context as progress_bar:
            self.rdm_ = torch.stack(Parallel(n_jobs = self.n_jobs_)
                                    (delayed(self.estimator_)
                                        (X.swapaxes(-2, -1)[i,...,k,:], 
                                        X.swapaxes(-2, -1)[j,...,k,:],
                                        *args)
                                    for k in range(T)),
                                    -1)
        
        return self.rdm_
    
    def fit_transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
        """Fit the estimator and transform data to representational similarity space.
        
        Parameters
        ----------
        X : torch.Tensor
            The data to compute the RDM for (samples x [...] x features [x time]).
        args : Any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        torch.Tensor
            The RDM (trials x [...] [x time]).
        """
        
        return self.fit(X, *args).transform(X, *args)
    
    def full_rdm(self) -> torch.Tensor:
        """Obtain the full RDM.

        Returns
        -------
        torch.Tensor
            The full RDM (trials x trials [x ... x time]).
        """

        # check transform
        if (self.dims_ is None) | (self.cx_ is None) | (self.cy_ is None) | (self.rdm_ is None):
            raise RuntimeError('`transform` must be called before `full_rdm`.')
        
        # setup rdm
        rdm = torch.zeros((self.dims_[0], self.dims_[0], *self.dims_[1:-2], self.dims_[-1]), dtype = self.rdm_.dtype, device = self.rdm_.device) * torch.nan
        rdm[self.cx_, self.cy_] = rdm[self.cy_, self.cx_] = self.rdm_
        
        return rdm
    
    def clone(self):
        """Obtain a clone of this class.
        
        Returns
        -------
        _RSA_torch
            The cloned estimator.
        """
        
        return _RSA_torch(estimator_ = self.estimator_, n_jobs_ = self.n_jobs_, verbose_ = self.verbose_)

class _GroupedRSA_numpy(sklearn.base.BaseEstimator):
    """Implements a grouped RSA estimator using numpy as our backend.
    """
    
    def __init__(self, estimator_: Callable = euclidean, n_jobs_: Union[int, None] = None, verbose_: bool = False):
        """Obtain a grouped RSA estimator.
        
        Parameters
        ----------
        estimator_ : Callable, optional
            The estimator to use for computing the RDM (default = mv.math.euclidean).
        n_jobs_ : int, optional
            The number of parallel jobs to use (default = None).
        verbose_ : bool, default=False
            Whether to print progress.
        """

        self.estimator_ = estimator_
        self.n_jobs_ = n_jobs_
        self.verbose_ = verbose_
        
        self.dims_ = None
        self.cx_ = None
        self.cy_ = None
        self.rdm_ = None
    
    def fit(self, *args):
        """Fit the estimator (vacant).
        """
        
        # call transform to obtain rdm
        self.transform(X, *args)
        
        return self

    def transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        """Transform the data to representational similarity space.
        
        Parameters
        ----------
        X : np.ndarray
            The data to compute the RDM for (samples x groups x [...] x features [x time])
        args : any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        rdm : np.ndarray
            The RDM (groupings x [...] [x time])
        """
        
        # setup dimensions
        dims = X.shape
        
        # check dimensions
        if len(dims) < 3:
            raise ValueError('`X` must be at least 3-dimensional.')
        
        # check dimensions
        if len(dims) < 4:
            X = X[...,np.newaxis]
        
        # setup dimensions
        self.dims_ = dims = X.shape
        N, G, F, T = dims[0], dims[1], dims[-2], dims[-1]
        
        # setup indices
        n = np.arange(G)
        nx, ny = np.meshgrid(n, n)
        self.cx_, self.cy_ = np.triu_indices(G, k = 1)
        i, j = nx.flatten()[self.cx_], ny.T.flatten()[self.cy_]
        
        # compute RDM
        context = Progressbar(enabled = self.verbose_, desc = "Computing RDM...", total = T)
        
        with context as progress_bar:
            self.rdm_ = np.stack(Parallel(n_jobs = self.n_jobs_)
                                    (delayed(self.estimator_)
                                        (X.swapaxes(-2, -1)[:,i,...,k,:].swapaxes(0, 1), 
                                        X.swapaxes(-2, -1)[:,j,...,k,:].swapaxes(0, 1),
                                        *args)
                                    for k in range(T)),
                                axis = -1)
        
        return self.rdm_
    
    def fit_transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        """Fit the estimator and transform data to representational similarity.
        
        Parameters
        ----------
        X : np.ndarray
            The data to compute the RDM for (samples x groups x [...] x features [x time]).
        args : Any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        np.ndarray
            The RDM (groupings x [...] [x time]).
        """
        
        return self.fit(X, *args).transform(X, *args)
    
    def full_rdm(self) -> np.ndarray:
        """Obtain the full RDM for visualisation purposes.
        
        Returns
        -------
        np.ndarray
            The full RDM (grouping x grouping [x ... x time]).
        """
        
        if (self.dims_ is None) | (self.cx_ is None) | (self.cy_ is None) | (self.rdm_ is None):
            raise RuntimeError('`transform` must be called before `full_rdm`.')
        
        # setup rdm
        shape = (self.dims_[1], self.dims_[1], self.dims_[-1]) if len(self.dims_) == 4 else (self.dims_[1], self.dims_[1], *self.dims_[2:-2], *self.dims_[-1])
        rdm = np.zeros(shape) * np.nan
        rdm[self.cx_, self.cy_] = rdm[self.cy_, self.cx_] = self.rdm_
        
        return rdm
        
    
    def clone(self):
        """Obtain a clone of this class.
        
        Returns
        -------
        _GroupedRSA_numpy
            The cloned estimator.
        """
        
        return _GroupedRSA_numpy(estimator_ = self.estimator_, n_jobs_ = self.n_jobs_, verbose_ = self.verbose_)

class _GroupedRSA_torch(sklearn.base.BaseEstimator):
    """Implements a group-wise RSA estimator using torch as our backend.
    """
    
    def __init__(self, estimator_: Callable = euclidean, n_jobs_: Union[int, None] = None, verbose_: bool = False):
        """Obtain a grouped RSA estimator.
        
        Parameters
        ----------
        estimator_ : Callable, optional
            The estimator to use for computing the RDM (default = mv.math.euclidean).
        n_jobs_ : Union[int, None], optional
            The number of parallel jobs to use. (default=None)
        verbose_ : bool, default=False
            Whether to print progress information.
        """
        
        # set options
        self.estimator_ = estimator_
        self.n_jobs_ = n_jobs_
        self.verbose_ = verbose_
        
        # set attributes
        self.cx_ = None
        self.cy_ = None
        self.dims_ = None
        self.rdm_ = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor = None, *args):
        """Fit the estimator. (vacant)
        """
        
        # call transform to obtain rdm
        self.transform(X, *args)
        
        return self
    
    def transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
        """Computes the RDM for a given grouped dataset.
        
        Parameters
        ----------
        X : torch.Tensor
            The data to compute the RDM for (samples x groups x [...] x features [x time])
        args : Any
            Additional arguments
        
        Returns
        -------
        rdm : torch.Tensor
            The RDM (groupings x [...] [x time])
        """
        
        # setup dimensions
        dims = X.shape
        
        # check dimensions
        if len(dims) < 3:
            raise ValueError('`X` must be at least 3-dimensional.')
        
        # check dimensions
        if len(dims) < 4:
            X = X[...,None]
        
        # setup dimensions
        self.dims_ = dims = X.shape
        N, G, F, T = dims[0], dims[1], dims[-2], dims[-1]
        
        # setup indices
        n = torch.arange(G)
        nx, ny = torch.meshgrid(n, n, indexing = 'ij')
        self.cx_, self.cy_ = torch.triu_indices(G, G, offset = 1)
        i, j = nx.T.flatten()[self.cx_], ny.flatten()[self.cy_]
        
        # compute RDM
        context = Progressbar(enabled = self.verbose_, desc = "Computing RDM...", total = T)
        
        with context as progress_bar:
            self.rdm_ = torch.stack(Parallel(n_jobs = self.n_jobs_)
                                    (delayed(self.estimator_)
                                        (X.swapaxes(-2, -1)[:,i,...,k,:], 
                                        X.swapaxes(-2, -1)[:,j,...,k,:],
                                        *args)
                                    for k in range(T)),
                                    -1)
            
        return self.rdm_
    
    def fit_transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
        """Fit the estimaotor and transform data to representational similarity.
        
        Parameters
        ----------
        X : torch.Tensor
            The data to compute the RDM for (samples x groups x [...] x features [x time]).
        args : Any
            Additional arguments to pass to the estimator.
        
        Returns
        -------
        rdm : torch.Tensor
            The RDM (groupings x [...] [x time]).
        """
        
        return self.fit(X, *args).transform(X, *args)
    
    def full_rdm(self) -> torch.Tensor:
        """Obtain the full RDM for visusalisation purposes.
        
        Returns
        -------
        rdm : torch.Tensor
            The RDM (groupings x groupings [x .... x time]).
        """
        
        # check transform
        if (self.dims_ is None) | (self.cx_ is None) | (self.cy_ is None) | (self.rdm_ is None):
            raise RuntimeError('`transform` must be called before `full_rdm`.')
        
        # setup place holder
        shape = (self.dims_[1], self.dims_[1], self.dims_[-1]) if len(self.dims_) == 4 else (self.dims_[1], self.dims_[1], *self.dims_[2:-2], *self.dims_[-1])
        rdm = torch.zeros(shape, dtype = self.rdm_.dtype, device = self.rdm_.device) * torch.nan
        rdm[self.cx_, self.cy_] = rdm[self.cy_, self.cx_] = self.rdm_
        
        return rdm
    
    def clone(self):
        """Obtain a clone of this class.
        
        Returns
        -------
        _GroupedRSA_torch
            The cloned estimator.
        """
        
        return _GroupedRSA_torch(estimator_ = self.estimator_, n_jobs_ = self.n_jobs_, verbose_ = self.verbose_)

class RSA(sklearn.base.BaseEstimator):
    """Implements representational similarity analysis.
    
    Representational similarity analysis computes the geometry of input
    data :math:`X` in their feature space. For example, given input
    data :math:`X` of shape ``(n_trials, n_channels, n_timepoints)``, 
    it would compute representational (dis-)similarity matrices of shape 
    ``(n_trials, n_trials, n_timepoints)`` through some (dis-)similarity
    function :math:`f`.
    
    Generally, performing this over representations of different systems
    allows drawing second-order comparisons about shared properties of
    those systems. This is typically done by computing  multiple (dis-)similarity 
    matrices from neural and simulated data before comparing the two using, 
    for example, :py:func:`~mvpy.math.spearmanr` to obtain a measure of 
    how similar some hypothetical simulated system is to the observed
    neural geometry.
    
    For more information on representational similarity analysis,
    please see [1]_ [2]_.
    
    Parameters
    ----------
    grouped : bool, default=False
        Whether to use a grouped RSA (this is required for cross-validated metrics to make sense, irrelevant otherwise).
    estimator : Callable, default=mvpy.math.euclidean
        The estimator/metric to use for RDM computation.
    n_jobs : int, default=None
        Number of jobs to run in parallel (default = None).
    verbose : bool, default=False
        Whether to print progress information.
    
    Attributes
    ----------
    rdm_ : np.ndarray | torch.Tensor
        The upper triangle of the representational (dis-)similarity matrix of shape ``(n_triu_indices[, ...])``.
    cx_ : np.ndarray | torch.Tensor
        The upper triangular indices of the RDM.
    cy_ : np.ndarray | torch.Tensor
        The upper triangular indices of the RDM.
    grouped : bool
        Whether the RSA is grouped.
    estimator : Callable
        The estimator/metric to use for RDM computation.
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : bool, default=False
        Whether to print progress information.
    
    See also
    --------
    mvpy.math.cosine, mvpy.math.cosine_d, mvpy.math.cv_euclidean, mvpy.math.cv_mahalanobis, mvpy.math.euclidean, mvpy.math.mahalanobis, mvpy.math.pearsonr, mvpy.math.pearsonr_d, mvpy.math.spearmanr, mvpy.math.spearmanr_d : Available (dis-)similarity functions.
    
    Notes
    -----
    Computing (dis-)similarity across input data :math:`X` may be inherently
    biassed. For example, distance metrics like :py:func:`~mvpy.math.euclidean` 
    or :py:func:`~mvpy.math.mahalanobis` may never truly be zero given the 
    noise inherent to neural responses. Consequently, cross-validation can
    be applied to compute unbiassed estimators through :py:func:`mvpy.math.cv_euclidean` 
    or :py:func:`mvpy.math.cv_mahalanobis`. To do this, make sure to collect
    many trials per condition and structure your data :math:`X` as 
    ``(n_trials, n_groups, n_channels, n_timepoints)`` while setting
    :py:attr:`~mvpy.estimators.RSA.grouped` ``True``.
    
    For more information on cross-validation, please see [3]_.
    
    References
    ----------
    .. [1] Kriegeskorte, N. (2008). Representational similarity analaysis - connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience. 10.3389/neuro.06.004.2008
    .. [2] Diedrichsen, J., & Kriegeskorte, N. (2017). Representational models: A common framework for understanding encoding, pattern-component, and representational similarity analysis. PLOS Computational Biology, 13, e1005508. 10.1371/journal.pcbi.1005508
    .. [3] Diedrichsen, J., Provost, S., & Zareamoghaddam, H. (2016). On the distribution of cross-validated mahalanobis distances. arXiv. 10.48550/arXiv.1607.01371
    
    Examples
    --------
    Let's assume we have some data with 100 trials and 5 groups, recording 10 channels over 50 time points:
    
    >>> import torch
    >>> from mvpy.math import euclidean, cv_euclidean
    >>> from mvpy.estimators import RSA
    >>> X = torch.normal(0, 1, (100, 5, 10, 50))
    >>> rsa = RSA(estimator = euclidean)
    >>> rsa.transform(X).shape
    torch.Size([4950, 5, 50])
    
    If we want to compute a cross-validated RSA over the groups instead, we can use:
    
    >>> rsa = RSA(grouped = True, estimator = cv_euclidean)
    >>> rsa.transform(X).shape
    torch.Size([10, 50])
    
    Finally, if we want to plot the full RDM, we can do:
    
    >>> rdm = torch.zeros((5, 5, 50))
    >>> rdm[rsa.cx_, rsa.cy_] = rsa.rdm_
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(rdm[...,0], cmap = 'RdBu_r')
    """
    
    def __init__(self, grouped: bool = False, estimator: Callable = euclidean, n_jobs: Union[int, None] = None, verbose: bool = False):
        """Obtain a representational similarity estimator.
        
        Parameters
        ----------
        grouped : bool, default=False
            Whether to use a grouped RSA (this is required for cross-validated metrics to make sense, irrelevant otherwise).
        estimator : callable, default=mv.math.euclidean
            The estimator to use.
        n_jobs : int, default=None
            Number of jobs to run in parallel (default = None).
        """
        
        # set attributes
        self.grouped = grouped
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # check estimator
        if not callable(estimator):
            raise TypeError(f'Estimator must be a callable type, but got {type(estimator)}.')
        
    def _get_estimator(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> sklearn.base.BaseEstimator:
        """Given grouping and data, determine which estimator to use.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The data to compute the RDM for of shape ``(n_trials[, n_groups], n_channels, n_timepoints)``.
        args : Any
            Additional arguments
        
        Returns
        -------
        estimator : sklearn.base.BaseEstimator
            The estimator to use.
        """
        
        if self.grouped & isinstance(X, torch.Tensor):
            return _GroupedRSA_torch
        elif self.grouped & isinstance(X, np.ndarray):
            return _GroupedRSA_numpy
        elif (not self.grouped) & isinstance(X, torch.Tensor):
            return _RSA_torch
        elif (not self.grouped) & isinstance(X, np.ndarray):
            return _RSA_numpy
        
        raise ValueError(f'Got an unexpected combination of grouped=`{self.grouped}` and type=`{type(X)}`.')

    def fit(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> "RSA":
        """Fit the estimator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The data to compute the RDM for of shape ``(n_trials[, n_groups], n_channels, n_timepoints)``.
        args : Any
            Additional arguments
        
        Returns
        -------
        rsa : mvpy.estimators.RSA
            Fitted RSA estimator.
        """
        
        return self._get_estimator(X, *args)(estimator_ = self.estimator, n_jobs_ = self.n_jobs, verbose_ = self.verbose).fit(X, *args)
    
    def transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Transform the data into representational similarity.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The data to compute the RDM for of shape ``(n_trials[, n_groups], n_channels, n_timepoints)``.
        args : Any
            Additional arguments
        
        Returns
        -------
        rdm : np.ndarray | torch.Tensor
            The representational similarity matrix of shape ``(n_trials, n_trials, n_timepoints)`` or ``(n_groups, n_groups, n_timepoints)``.
        """
        
        return self._get_estimator(X, *args)(estimator_ = self.estimator, n_jobs_ = self.n_jobs, verbose_ = self.verbose).transform(X, *args)
    
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Fit the estimator and transform data into representational similarity.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The data to compute the RDM for of shape ``(n_trials[, n_groups], n_channels, n_timepoints)``.
        args : Any
            Additional arguments
        
        Returns
        -------
        rdm : np.ndarray | torch.Tensor
            The representational similarity matrix of shape ``(n_trials, n_trials, n_timepoints)`` or ``(n_groups, n_groups, n_timepoints)``.
        """
        
        return self._get_estimator(X, *args)(estimator_ = self.estimator, n_jobs_ = self.n_job, verbose_ = self.verbose).fit_transform(X, *args)
    
    def to_torch(self):
        """Make this estimator use the torch backend. Note that this method does not support conversion between types.
        
        Returns
        -------
        rsa : sklearn.base.BaseEstimator
            The estimator.
        """
        
        return self._get_estimator(torch.tensor([1]))(estimator_ = self.estimator, n_jobs_ = self.n_jobs, verbose_ = self.verbose)
    
    def to_numpy(self):
        """Make this estimator use the numpy backend. Note that this method does not support conversion between types.
        
        Returns
        -------
        rsa : sklearn.base.BaseEstimator
            The estimator.
        """

        return self._get_estimator(np.array([1]))(estimator_ = self.estimator, n_jobs_ = self.n_jobs, verbose_ = self.verbose)
    
    def full_rdm(self) -> Union[np.ndarray, torch.Tensor]:
        """Obtain the full representational similartiy matrix.
        
        Returns
        -------
        rdm : np.ndarray | torch.Tensor
            The representational similarity matrix of shape ``(n_trials, n_trials, n_timepoints)`` or ``(n_groups, n_groups, n_timepoints)``.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
        
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        rsa : mvpy.estimators.RSA
            A clone of this class.
        """
        
        return RSA(grouped = self.grouped, estimator = self.estimator, n_jobs = self.n_jobs, verbose = self.verbose)