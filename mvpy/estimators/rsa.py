'''
A collection of estimators for computing
representational similarities. For more
information, see:

    Kriegeskorte, N. (2008). Representational similarity analaysis - connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience. 10.3389/neuro.06.004.2008
'''

import sklearn
import numpy as np
import torch

from joblib import Parallel, delayed

from typing import Union, Callable, Any

from ..math import euclidean

class _RSA_numpy(sklearn.base.BaseEstimator):
    '''
    Implements a trial-wise RSA estimator using numpy
    as our backend. This is the default estimator.
    '''
    
    def __init__(self, estimator: Callable = euclidean, n_jobs: Union[int, None] = None):
        '''
        Constructor.
        
        INPUTS:
            estimator   -   The estimator to use for computing the RDM (default = mv.math.euclidean).
            n_jobs      -   The number of parallel jobs to use (default = None).
        '''
        
        self.estimator_ = estimator
        self.n_jobs_ = n_jobs
        
        self.cx_ = None
        self.cy_ = None
        self.rdm_ = None

    def fit(self, X: np.ndarray, *args: Any):
        '''
        Fit method (vacant).
        '''
        
        return self
    
    def transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        '''
        Computes the RDM for a given trial-wise dataset.
        
        INPUTS:
            X       -   The data to compute the RDM for (samples x [...] x features [x time])
            args    -   Additional arguments to pass to the estimator.
        
        OUTPUTS:
            rdm     -   The RDM (trials x [...] [x time])
        '''
        
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
        '''
        Computes the RDM for a given trial-wise dataset.
        
        INPUTS:
            X       -   The data to compute the RDM for (samples x [...] x features [x time])
            args    -   Additional arguments to pass to the estimator.
        
        OUTPUTS:
            rdm     -   The RDM (trials x [...] [x time])
        '''
        
        return self.fit(X, *args).transform(X, *args)

class _RSA_torch(sklearn.base.BaseEstimator):
    '''
    Implements a trial-wise RSA estimator using torch
    as our backend. This is the default estimator.
    '''
    
    def __init__(self, estimator: Callable = euclidean, n_jobs: Union[int, None] = None):
        '''
        Constructor.
        
        INPUTS:
            estimator   -   The estimator to use for computing the RDM (default = mv.math.euclidean).
            n_jobs      -   The number of parallel jobs to use (default = None).
        '''
        
        self.estimator_ = estimator
        self.n_jobs_ = n_jobs
        
        self.cx_ = None
        self.cy_ = None
        self.rdm_ = None

    def fit(self, X: torch.Tensor, *args: Any):
        '''
        Fit method (vacant).
        '''
        
        return self
    
    def transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
        '''
        Computes the RDM for a given trial-wise dataset.
        
        INPUTS:
            X       -   The data to compute the RDM for (samples x [...] x features [x time])
            args    -   Additional arguments to pass to the estimator.
        
        OUTPUTS:
            rdm     -   The RDM (trials x [...] [x time])
        '''
        
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
        '''
        Computes the RDM for a given trial-wise dataset.
        
        INPUTS:
            X       -   The data to compute the RDM for (samples x [...] x features [x time])
            args    -   Additional arguments to pass to the estimator.
        
        OUTPUTS:
            rdm     -   The RDM (trials x [...] [x time])
        '''
        
        return self.fit(X, *args).transform(X, *args)

class _GroupedRSA_numpy(sklearn.base.BaseEstimator):
    '''
    Implements a group-wise RSA estimator using numpy
    as our backend. Note that grouped estimators are
    required only for cross-validated metrics.
    '''
    
    def __init__(self, estimator: Callable = euclidean, n_jobs: Union[int, None] = None):
        '''
        Constructor.
        
        INPUTS:
            estimator   -   The estimator to use for computing the RDM (default = mv.math.euclidean).
            n_jobs      -   The number of parallel jobs to use (default = None).
        '''
        
        self.estimator_ = estimator
        self.n_jobs_ = n_jobs
        
        self.cx_ = None
        self.cy_ = None
        self.rdm_ = None
    
    def fit(self, *args):
        '''
        Fit method (vacant).
        '''
        
        return self

    def transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        '''
        Computes the RDM for a given grouped dataset.
        
        INPUTS:
            X       -   The data to compute the RDM for (samples x groups x [...] x features [x time])
            args    -   Additional arguments to pass to the estimator.
        
        OUTPUTS:
            rdm     -   The RDM (groupings x [...] [x time])
        '''
        
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
        '''
        Computes the RDM for a given grouped dataset.
        
        INPUTS:
            X       -   The data to compute the RDM for (samples x groups x [...] x features [x time])
            args    -   Additional arguments to pass to the estimator.
        
        OUTPUTS:
            rdm     -   The RDM (groupings x [...] [x time])
        '''
        
        return self.fit(X, *args).transform(X, *args)

class _GroupedRSA_torch(sklearn.base.BaseEstimator):
    '''
    Implements a group-wise RSA estimator using torch
    as our backend. Note that grouped estimators are
    required only for cross-validated metrics.
    '''
    
    def __init__(self, estimator: Callable = euclidean, n_jobs: Union[int, None] = None):
        '''
        Constructor.
        
        INPUTS:
            estimator   -   The estimator to use for computing the RDM (default = mv.math.euclidean).
            n_jobs      -   The number of parallel jobs to use (default = None).
        '''
        
        self.estimator_ = estimator
        self.n_jobs_ = n_jobs
        
        self.cx_ = None
        self.cy_ = None
        self.rdm_ = None
    
    def fit(self, *args):
        '''
        Fit method (vacant).
        '''
        
        return self

    def transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
        '''
        Computes the RDM for a given grouped dataset.
        
        INPUTS:
            X       -   The data to compute the RDM for (samples x groups x [...] x features [x time])
            args    -   Additional arguments to pass to the estimator.
        
        OUTPUTS:
            rdm     -   The RDM (groupings x [...] [x time])
        '''
        
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
        '''
        Computes the RDM for a given grouped dataset.
        
        INPUTS:
            X       -   The data to compute the RDM for (samples x groups x [...] x features [x time])
            args    -   Additional arguments to pass to the estimator.
        
        OUTPUTS:
            rdm     -   The RDM (groupings x [...] [x time])
        '''
        
        return self.fit(X, *args).transform(X, *args)

class RSA(sklearn.base.BaseEstimator):
    '''
    Class managing the RSA estimators.
    '''
    
    def __init__(self, grouped: bool = False, estimator: Callable = euclidean, n_jobs: Union[int, None] = None):
        '''
        Constructor.
        
        INPUTS:
            grouped     -   Do we need a grouped RSA (this is required for cross-validated metrics to make sense, irrelevant otherwise)?
            estimator   -   The estimator/metric to use (default = mv.math.euclidean).
            n_jobs      -   Number of jobs to run in parallel (default = None).
        '''
        
        self.grouped_ = grouped
        self.estimator_ = estimator
        self.n_jobs_ = n_jobs
    
    def _get_estimator(self, X: np.ndarray, *args: Any) -> sklearn.base.BaseEstimator:
        '''
        Given grouping and data, decide which RSA estimator to supply.
        
        INPUTS:
            X       -   The data.
            args    -   Additional arguments.
        
        OUTPUTS:
            RSA     -   The estimator to use.
        '''
        
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
        '''
        Call fit method on the estimator.
        
        INPUTS:
            X       -   The data (either [[samples] x ][samples] x features x time or [samples] x grouping [x samples [x samples]] x features x time).
            args    -   Additional arguments to pass to estimator.
        '''
        
        return self._get_estimator(X, *args)(estimator = self.estimator_, n_jobs = self.n_jobs_).fit(X, *args)
    
    def transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        '''
        Call transform method on the estimator.
        
        INPUTS:
            X       -   The data (either [[samples] x ][samples] x features x time or [samples] x grouping [x samples [x samples]] x features x time).
            args    -   Additional arguments to pass to estimator.
        '''
        
        return self._get_estimator(X, *args)(estimator = self.estimator_, n_jobs = self.n_jobs_).transform(X, *args)
    
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        '''
        Call fit_transform methond on the estimator.
        
        INPUTS:
            X       -   The data (either [[samples] x ][samples] x features x time or [samples] x grouping [x samples [x samples]] x features x time).
            args    -   Additional arguments to pass to estimator.
        '''
        
        return self._get_estimator(X, *args)(estimator = self.estimator_, n_jobs = self.n_jobs_).fit_transform(X, *args)