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
        if None in [self.dims_, self.cx_, self.cy_, self.rdm_]:
            raise RuntimeError('`transform` must be called before `full_rdm`.')
        
        # setup rdm
        rdm = np.zeros((self.dims_[0], self.dims_[0], *self.dims_[1:-2], self.dims[-1])) * np.nan
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
        if None in [self.dims_, self.cx_, self.cy_, self.rdm_]:
            raise RuntimeError('`transform` must be called before `full_rdm`.')
        
        # setup rdm
        rdm = torch.zeros((self.dims_[0], self.dims_[0], *self.dims_[1:-2], self.dims_[-1])) * torch.nan
        rdm[self.cx_, self.cy_] = rdm[self.cy_, self.cx_] = self.rdm_
        
        return self.rdm_
    
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
        
        if None in [self.dims_, self.rdm_, self.cx_, self.cy_]:
            raise RuntimeError('`transform` must be called before `full_rdm`.')
        
        # setup rdm
        rdm = np.zeros((self.dims_[1], self.dims_[1], *self.dims_[2:-2], *self.dims_[-1])) * np.nan
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
        if None in [self.dims_, self.rdm_, self.cx_, self.cy_]:
            raise RuntimeError('`transform` must be called before `full_rdm`.')
        
        # setup place holder
        rdm = torch.zeros((self.dims_[1], self.dims_[1], *self.dims_[2:-2], self.dims_[-1]), dtype = self.rdm_.dtype, device = self.rdm_.device) * torch.nan
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
    """Implements representational similarity analysis as an estimator. Note that this class expects features to be the second to last dimension.
    
    Parameters
    ----------
    grouped : bool, default=False
        Whether to use a grouped RSA (this is required for cross-validated metrics to make sense, irrelevant otherwise).
    estimator : callable, default=mv.math.euclidean
        The estimator/metric to use for RDM computation.
    n_jobs : int, default=None
        Number of jobs to run in parallel (default = None).
    verbose : bool, default=False
        Whether to print progress information.
    
    Attributes
    ----------
    rdm_ : Union[np.ndarray, torch.Tensor]
        The representational (dis)similarity matrix.
    cx_ : Union[np.ndarray, torch.Tensor]
        The upper triangular indices of the RDM.
    cy_ : Union[np.ndarray, torch.Tensor]
        The upper triangular indices of the RDM.
    grouped : bool
        Whether the RSA is grouped.
    estimator : Callable
        The estimator/metric to use for RDM computation.
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : bool
        Whether to print progress information.
    
    Notes
    -----
    If you would like to perform, for example, a cross-validated RSA using :func:`mvpy.math.cv_euclidean`, you should make sure that the first dimension in your data is trials, whereas the second dimension groups them meaningfully. The resulting RDM will then be computed over groups, with cross-validation over trials.
    
    For more information on representational similarity, please see [2]_.
    
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
    
    Note that if you would like to perform a decoding RSA, you can use a OvR classifier instead. For example, let's assume we have data from 100 trials, 60 channels and 50 time points. Data are from 5 distinct classes.
    
    >>> from mvpy.estimators import Classifier
    >>> X = torch.normal(0, 1, (100, 10, 50))
    >>> y = torch.randint(0, 5, (100, 50))
    >>> 
    
    References
    ----------
    .. [2] Kriegeskorte, N. (2008). Representational similarity analaysis - connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience. 10.3389/neuro.06.004.2008
    
    See also
    --------
    :func:`mvpy.math.euclidean`
    :func:`mvpy.math.cv_euclidean`
    :func:`mvpy.math.cosine`
    :func:`mvpy.math.cosine_d`
    :func:`mvpy.math.pearsonr`
    :func:`mvpy.math.pearsonr_d`
    :func:`mvpy.math.spearmanr`
    :func:`mvpy.math.spearmanr_d`
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
        
    def _get_estimator(self, X: np.ndarray, *args: Any) -> sklearn.base.BaseEstimator:
        """Given grouping and data, determine which estimator to use.
        
        Parameters
        ----------
        X : np.ndarray
            The data to compute the RDM for.
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

    def fit(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Any:
        """Fit the estimator.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The data to compute the RDM for.
        args : Any
            Additional arguments
        """
        
        return self._get_estimator(X, *args)(estimator_ = self.estimator, n_jobs_ = self.n_jobs, verbose_ = self.verbose).fit(X, *args)
    
    def transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Transform the data into representational similarity.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The data to compute the RDM for.
        args : Any
            Additional arguments
        
        Returns
        -------
        rdm : Union[np.ndarray, torch.Tensor]
            The representational similarity.
        """
        
        return self._get_estimator(X, *args)(estimator_ = self.estimator, n_jobs_ = self.n_jobs, verbose_ = self.verbose).transform(X, *args)
    
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Fit the estimator and transform data into representational similarity.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The data to compute the RDM for.
        args : Any
            Additional arguments
        
        Returns
        -------
        rdm : Union[np.ndarray, torch.Tensor]
            The representational similarity.
        """
        
        return self._get_estimator(X, *args)(estimator_ = self.estimator, n_jobs_ = self.n_job, verbose_ = self.verbose).fit_transform(X, *args)
    
    def to_torch(self):
        """Make this estimator use the torch backend. Note that this method does not support conversion between types.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The estimator.
        """
        
        return self._get_estimator(torch.tensor([1]))(estimator_ = self.estimator, n_jobs_ = self.n_jobs, verbose_ = self.verbose)
    
    def to_numpy(self):
        """Make this estimator use the numpy backend. Note that this method does not support conversion between types.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The estimator.
        """

        return self._get_estimator(np.array([1]))(estimator = self.estimator, n_jobs = self.n_jobs, verbose = self.verbose)
    
    def full_rdm(self) -> Union[np.ndarray, torch.Tensor]:
        """Obtain the full representational similartiy matrix.
        
        Returns
        -------
        rdm : Union[np.ndarray, torch.Tensor]
            The representational similarity matrix in full.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
        
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        RSA
            A clone of this class.
        """
        
        return RSA(grouped = self.grouped, estimator = self.estimator, n_jobs = self.n_jobs, verbose = self.verbose)