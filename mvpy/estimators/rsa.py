'''
A collection of estimators for computing
representational similarities.
'''

import sklearn
import numpy as np
import torch

from joblib import Parallel, delayed

from typing import Union, Callable, Any

from ..math import euclidean

class _RSA_numpy(sklearn.base.BaseEstimator):
    """Implements a representational similarity estimator using numpy backend.
    """
    
    def __init__(self, estimator: Callable = euclidean, n_jobs: Union[int, None] = None):
        """Obtain a representational similarity estimator.
        
        Parameters
        ----------
        estimator : Callable, optional
            The estimator to use for computing the RDM (default = mv.math.euclidean).
        n_jobs : Union[int, None], optional
            The number of parallel jobs to use (default = None).
        """
        
        self.estimator_ = estimator
        self.n_jobs_ = n_jobs
        
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
        dims = X.shape
        N, F, T = dims[0], dims[-2], dims[-1]
        
        # setup indices
        n = np.arange(N)
        nx, ny = np.meshgrid(n, n)
        self.cx_, self.cy_ = np.triu_indices(N, k = 1)
        i, j = nx.flatten()[self.cx_], ny.T.flatten()[self.cy_]
        
        # compute RDM
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

class _RSA_torch(sklearn.base.BaseEstimator):
    """Implements representational similarity estimation using torch as our backend.
    """
    
    def __init__(self, estimator: Callable = euclidean, n_jobs: Union[int, None] = None):
        """Obtain a RSA estimator.
        
        Parameters
        ----------
        estimator : Callable, optional
            The estimator to use for computing the RDM (default = mv.math.euclidean).
        n_jobs : Union[int, None], optional
            The number of parallel jobs to use (default = None).
        """
        
        self.estimator_ = estimator
        self.n_jobs_ = n_jobs
        
        self.cx_ = None
        self.cy_ = None
        self.rdm_ = None

    def fit(self, X: torch.Tensor, *args: Any):
        """Fit the estimator (vacant).
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
        dims = X.shape
        N, F, T = dims[0], dims[-2], dims[-1]
        
        # setup indices
        n = torch.arange(N)
        nx, ny = torch.meshgrid(n, n, indexing = 'ij')
        self.cx_, self.cy_ = torch.triu_indices(N, N, offset = 1)
        i, j = nx.T.flatten()[self.cx_], ny.flatten()[self.cy_]
        
        # compute RDM
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

class _GroupedRSA_numpy(sklearn.base.BaseEstimator):
    """Implements a grouped RSA estimator using numpy as our backend.
    """
    
    def __init__(self, estimator: Callable = euclidean, n_jobs: Union[int, None] = None):
        """Obtain a grouped RSA estimator.
        
        Parameters
        ----------
        estimator : Callable, optional
            The estimator to use for computing the RDM (default = mv.math.euclidean).
        n_jobs : int, optional
            The number of parallel jobs to use (default = None).
        """

        self.estimator_ = estimator
        self.n_jobs_ = n_jobs
        
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
        dims = X.shape
        N, G, F, T = dims[0], dims[1], dims[-2], dims[-1]
        
        # setup indices
        n = np.arange(G)
        nx, ny = np.meshgrid(n, n)
        self.cx_, self.cy_ = np.triu_indices(G, k = 1)
        i, j = nx.flatten()[self.cx_], ny.T.flatten()[self.cy_]
        
        # compute RDM
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

class _GroupedRSA_torch(sklearn.base.BaseEstimator):
    """Implements a group-wise RSA estimator using torch as our backend.
    """
    
    def __init__(self, estimator: Callable = euclidean, n_jobs: Union[int, None] = None):
        """Obtain a grouped RSA estimator.
        
        Parameters
        ----------
        estimator : Callable, optional
            The estimator to use for computing the RDM (default = mv.math.euclidean).
        n_jobs : Union[int, None], optional
            The number of parallel jobs to use. (default=None)
        """
        
        self.estimator_ = estimator
        self.n_jobs_ = n_jobs
        
        self.cx_ = None
        self.cy_ = None
        self.rdm_ = None
    
    def fit(self, X: torch.Tensor, *args):
        """Fit the estimator (vaccant)."""
        
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
        dims = X.shape
        N, G, F, T = dims[0], dims[1], dims[-2], dims[-1]
        
        # setup indices
        n = torch.arange(G)
        nx, ny = torch.meshgrid(n, n, indexing = 'ij')
        self.cx_, self.cy_ = torch.triu_indices(G, G, offset = 1)
        i, j = nx.T.flatten()[self.cx_], ny.flatten()[self.cy_]
        
        # compute RDM
        self.rdm_ = torch.stack(Parallel(n_jobs = self.n_jobs_)
                                (delayed(self.estimator_)
                                    (X.swapaxes(-2, -1)[:,i,...,k,:], 
                                     X.swapaxes(-2, -1)[:,j,...,k,:],
                                     *args)
                                 for k in range(T)),
                                -1)
        
        return self.rdm_
    
    def fit_transform(self, X: torch.Tensor, *args: Any) -> np.ndarray:
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
    
    Attributes
    ----------
    rdm_ : Union[np.ndarray, torch.Tensor]
        The representational (dis)similarity matrix.
    cx_ : Union[np.ndarray, torch.Tensor]
        The upper triangular indices of the RDM.
    cy_ : Union[np.ndarray, torch.Tensor]
        The upper triangular indices of the RDM.
    grouped_ : bool
        Whether the RSA is grouped.
    estimator_ : Callable
        The estimator/metric to use for RDM computation.
    n_jobs_ : int
        Number of jobs to run in parallel.
    
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
    
    def __init__(self, grouped: bool = False, estimator: Callable = euclidean, n_jobs: Union[int, None] = None):
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
        
        self.grouped_ = grouped
        self.estimator_ = estimator
        self.n_jobs_ = n_jobs
    
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
        
        if self.grouped_ & isinstance(X, torch.Tensor):
            return _GroupedRSA_torch
        elif self.grouped_ & isinstance(X, np.ndarray):
            return _GroupedRSA_numpy
        elif (not self.grouped_) & isinstance(X, torch.Tensor):
            return _RSA_torch
        elif (not self.grouped_) & isinstance(X, np.ndarray):
            return _RSA_numpy
        
        raise ValueError(f'Got an unexpected combination of grouped=`{self.grouped_}` and type=`{type(X)}`.')

    def fit(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Any:
        """Fit the estimator.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The data to compute the RDM for.
        args : Any
            Additional arguments
        """
        
        return self._get_estimator(X, *args)(estimator = self.estimator_, n_jobs = self.n_jobs_).fit(X, *args)
    
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
        
        return self._get_estimator(X, *args)(estimator = self.estimator_, n_jobs = self.n_jobs_).transform(X, *args)
    
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
        
        return self._get_estimator(X, *args)(estimator = self.estimator_, n_jobs = self.n_jobs_).fit_transform(X, *args)