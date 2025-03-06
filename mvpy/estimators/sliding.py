'''
A collection of estimators that allow for sliding other estimators over a dimension of the data.
'''

import numpy as np
import torch
import sklearn

from joblib import Parallel, delayed

from ..utilities import Progressbar

from typing import Union, Any, Callable

class _Sliding_numpy(sklearn.base.BaseEstimator):
    """Implements a sliding estimator using numpy as our backend.
    
    Parameters
    ----------
    estimator : Union[Callable, sklearn.base.BaseEstimator]
        The estimator to slide across the data.
    dims : tuple, default=(-1,)
        Over which dimensions to slide the estimator.
    n_jobs : Union[int, None], default=None
        The number of parallel jobs to use.
    top : bool, default=True
        Is this the top-level slider?
    verbose : bool, default=True
        Whether to print progress.
    
    Attributes
    ----------
    estimators_ : list
        The estimators that were fit.
    """
    
    def __init__(self, estimator: Union[Callable, sklearn.base.BaseEstimator], dims: tuple = (-1,), n_jobs: Union[int, None] = None, top: bool = True, verbose: bool = True):
        """Obtain a new sliding estimator.
        
        Parameters
        ----------
        estimator : Union[Callable, sklearn.base.BaseEstimator]
            The estimator to slide across the data.
        dims : tuple, default=(-1,)
            Over which dimensions to slide the estimator.
        n_jobs : Union[int, None], default=None
            The number of parallel jobs to use.
        top : bool, default=True
            Is this the top-level slider?
        verbose : bool, default=True
            Whether to print progress.
        
        Attributes
        ----------
        estimators_ : list
            The estimators that were fit.
        Any : Any
            Attributes that will be collected from estimators.
        """

        # setup opts
        self.estimator = estimator
        self.dims = dims
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.top = top
        
        # setup attributes
        self.estimators_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, *args: Any):
        """Fit all estimators.
        
        Parameters
        ----------
        X : np.ndarray
            The data to slide the estimator over.
        y : np.ndarray
            The data to slide the estimator over.
        *args : Any
            Additional arguments to pass to the estimator.
        """
        
        # check types
        if (isinstance(X, np.ndarray) == False) | (isinstance(y, np.ndarray) == False):
            raise ValueError(f'X and y must be of type np.ndarray but got {type(X)} and {type(y)}.')
        
        # check dimensions
        if len(X.shape) != len(y.shape):
            raise ValueError(f'X and y must have the same number of dimensions for alignment, but got {X.shape} and {y.shape}.')
        
        # check estimator
        if isinstance(self.estimator, sklearn.base.BaseEstimator):
            # move sliding axis to final dimension
            X = np.moveaxis(X, self.dims[0], -1)
            y = np.moveaxis(y, self.dims[0], -1)
            
            # setup our estimator depending on dims
            effective_dims = tuple((np.arange(len(X.shape))[list(self.dims)] - 1).tolist())
            estimator_ = _Sliding_numpy(estimator = self.estimator, dims = effective_dims[1:], n_jobs = None, top = False) if len(self.dims) > 1 else self.estimator.clone()
            
            # fit estimators
            context = Progressbar(enabled = self.verbose, desc = "Fitting estimators...", total = X.shape[-1])
            
            X_i = np.arange(X.shape[-1]).astype(int) if X.shape[-1] > 1 else np.zeros((y.shape[-1],)).astype(int)
            y_i = np.arange(y.shape[-1]).astype(int) if y.shape[-1] > 1 else np.zeros((X.shape[-1],)).astype(int)
            
            with context as progress_bar:
                self.estimators_ = [Parallel(n_jobs = self.n_jobs)
                                        (delayed(estimator_.clone().fit)
                                            (X[...,i], 
                                            y[...,j], 
                                            *args)
                                        for i, j in zip(X_i, y_i))][0]
        else:
            self.estimators_ = []
        
        return self

    def transform(self, X: np.ndarray, y: Union[np.ndarray, None] = None, *args) -> np.ndarray:
        """Transform data in all estimators.
        
        Parameters
        ----------
        X : np.ndarray
            The data to transform.
        y : Union[np.ndarray, None], default=None
            The targets to transform.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        np.ndarray
            The transformed data.
        """
        
        # check types
        if (isinstance(X, np.ndarray) == False):
            raise ValueError(f'X must be of type np.ndarray but got {type(X)}.')
        
        # check estimator
        if (self.estimators_ is None) & (callable(self.estimator) == False):
            raise ValueError('Estimators not fitted yet.')

        # check estimator type
        if isinstance(self.estimator, sklearn.base.BaseEstimator):
            # move sliding axis to final dimension
            X = np.moveaxis(X, self.dims[0], -1)

            # make predictions
            return np.stack(Parallel(n_jobs = self.n_jobs)
                                    (delayed(self.estimators_[i].transform)
                                        (X[...,i], 
                                         *args)
                                     for i in range(X.shape[-1])), self.dims[0])
        else:
            # move sliding axis to final dimension
            X = np.moveaxis(X, self.dims[0], -1)
            y = np.moveaxis(y, self.dims[0], -1)

            # setup our estimator depending on dims
            effective_dims = tuple((np.arange(len(X.shape))[list(self.dims)] - 1).tolist())
            estimator_ = _Sliding_numpy(estimator = self.estimator, dims = effective_dims[1:], n_jobs = None, top = False).transform if len(self.dims) > 1 else self.estimator
            
            # fit estimators
            X_i = np.arange(X.shape[-1]).astype(int) if X.shape[-1] > 1 else np.zeros((y.shape[-1],)).astype(int)
            y_i = np.arange(y.shape[-1]).astype(int) if y.shape[-1] > 1 else np.zeros((X.shape[-1],)).astype(int)
            
            Z = np.stack(Parallel(n_jobs = self.n_jobs)
                                (delayed(estimator_)
                                    (X[...,i], 
                                     y[...,j], 
                                     *args)
                                 for i, j in zip(X_i, y_i)), self.dims[0] - 1)
            
            # if we're the top level, make sure trials remain first dimension
            if self.top:
                return Z.swapaxes(0, 1)
            else:
                return Z
    
    def fit_transform(self, X: np.ndarray, y: Union[np.ndarray, None] = None, *args) -> np.ndarray:
        """Fit and transform data from all estimators.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray, default=None
            Target data.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        np.ndarray
            Transformed data.
        """
        
        return self.fit(X, y = y, *args).transform(X, y = y, *args)
    
    def predict(self, X: np.ndarray, y: Union[np.ndarray, None] = None, *args) -> np.ndarray:
        """Make predictions from all estimators.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray, default=None
            Target data.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        np.ndarray
            Predictions.
        """
        
        # check types
        if (isinstance(X, np.ndarray) == False):
            raise ValueError(f'X must be of type np.ndarray but got {type(X)}.')
        
        # check estimator
        if (self.estimators_ is None) & (callable(self.estimator) == False):
            raise ValueError('Estimators not fitted yet.')

        # check estimator type
        if isinstance(self.estimator, sklearn.base.BaseEstimator):
            # move sliding axis to final dimension
            X = np.moveaxis(X, self.dims[0], -1)

            # make predictions
            return np.stack(Parallel(n_jobs = self.n_jobs)
                                    (delayed(self.estimators_[i].predict)
                                        (X[...,i], 
                                         *args)
                                     for i in range(X.shape[-1])), self.dims[0])
        else:
            # move sliding axis to final dimension
            X = np.moveaxis(X, self.dims[0], -1)
            y = np.moveaxis(y, self.dims[0], -1)

            # setup our estimator depending on dims
            effective_dims = tuple((np.arange(len(X.shape))[list(self.dims)] - 1).tolist())
            estimator_ = _Sliding_numpy(estimator = self.estimator, dims = effective_dims[1:], n_jobs = None, top = False).predict if len(self.dims) > 1 else self.estimator
            
            # fit estimators
            X_i = np.arange(X.shape[-1]).astype(int) if X.shape[-1] > 1 else np.zeros((y.shape[-1],)).astype(int)
            y_i = np.arange(y.shape[-1]).astype(int) if y.shape[-1] > 1 else np.zeros((X.shape[-1],)).astype(int)

            Z = np.stack(Parallel(n_jobs = self.n_jobs)
                                (delayed(estimator_)
                                    (X[...,i], 
                                     y[...,j], 
                                     *args)
                                 for i, j in zip(X_i, y_i)), self.dims[0] - 1)
            
            # if we're the top level, make sure trials remain first dimension
            if self.top:
                return Z.swapaxes(0, 1)
            else:
                return Z
    
    def predict(self, X: np.ndarray, y: Union[np.ndarray, None] = None, *args) -> np.ndarray:
        """Make predictions from all estimators.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray, default=None
            Target data.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        np.ndarray
            Predictions.
        """
        
        # check types
        if (isinstance(X, np.ndarray) == False):
            raise ValueError(f'X must be of type np.ndarray but got {type(X)}.')
        
        # check estimator
        if (self.estimators_ is None) & (callable(self.estimator) == False):
            raise ValueError('Estimators not fitted yet.')

        # check estimator type
        if isinstance(self.estimator, sklearn.base.BaseEstimator):
            # move sliding axis to final dimension
            X = np.moveaxis(X, self.dims[0], -1)

            # make predictions
            return np.stack(Parallel(n_jobs = self.n_jobs)
                                    (delayed(self.estimators_[i].predict_proba)
                                        (X[...,i], 
                                         *args)
                                     for i in range(X.shape[-1])), self.dims[0])
        else:
            # move sliding axis to final dimension
            X = np.moveaxis(X, self.dims[0], -1)
            y = np.moveaxis(y, self.dims[0], -1)

            # setup our estimator depending on dims
            effective_dims = tuple((np.arange(len(X.shape))[list(self.dims)] - 1).tolist())
            estimator_ = _Sliding_numpy(estimator = self.estimator, dims = effective_dims[1:], n_jobs = None, top = False).predict_proba if len(self.dims) > 1 else self.estimator
            
            # fit estimators
            X_i = np.arange(X.shape[-1]).astype(int) if X.shape[-1] > 1 else np.zeros((y.shape[-1],)).astype(int)
            y_i = np.arange(y.shape[-1]).astype(int) if y.shape[-1] > 1 else np.zeros((X.shape[-1],)).astype(int)

            Z = np.stack(Parallel(n_jobs = self.n_jobs)
                                (delayed(estimator_)
                                    (X[...,i], 
                                     y[...,j], 
                                     *args)
                                 for i, j in zip(X_i, y_i)), self.dims[0] - 1)
            
            # if we're the top level, make sure trials remain first dimension
            if self.top:
                return Z.swapaxes(0, 1)
            else:
                return Z
    
    def collect(self, attr: str) -> np.ndarray:
        """Collect an attribute from all fitted estimators.
        
        Parameters
        ----------
        attr : str
            Attribute to collect.
        
        Returns
        -------
        np.ndarray
            Tensor of shape (n_estimators, ...)
        """
        
        if isinstance(self.estimators_[0], _Sliding_numpy):
            return np.stack([self.estimators_[i].collect(attr) for i in range(len(self.estimators_))])
        else:
            if hasattr(self.estimators_[0], attr) == False:
                raise AttributeError(f'Estimator of type `{type(self.estimators_[0])}` does not have attribute `{attr}`.')
            
            return np.stack([getattr(self.estimators_[i], attr) for i in range(len(self.estimators_))])
    
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        _Sliding_numpy
            Cloned class.
        """
        
        return _Sliding_numpy(estimator = self.estimator, dims = self.dims, n_jobs = self.n_jobs, top = self.top, verbose = self.verbose)

class _Sliding_torch(sklearn.base.BaseEstimator):
    """Implements a sliding estimator using torch as our backend.
    
    Parameters
    ----------
    estimator : Union[Callable, sklearn.base.BaseEstimator]
        The estimator to slide across the data.
    dims : tuple, default=(-1,)
        Over which dimensions to slide the estimator.
    n_jobs : Union[int, None], default=None
        The number of parallel jobs to use.
    top : bool, default=True
        Is this the top-level slider?
    verbose : bool, default=True
        Whether to print progress.
    
    Attributes
    ----------
    estimators_ : list
        The estimators that were fit.
    """
    
    def __init__(self, estimator: Union[Callable, sklearn.base.BaseEstimator], dims: tuple = (-1,), n_jobs: Union[int, None] = None, top: bool = True, verbose: bool = True):
        """Obtain a new sliding estimator.
        
        Parameters
        ----------
        estimator : Union[Callable, sklearn.base.BaseEstimator]
            The estimator to slide across the data.
        dims : tuple, default=(-1,)
            Over which dimensions to slide the estimator.
        n_jobs : Union[int, None], default=None
            The number of parallel jobs to use.
        top : bool, default=True
            Is this the top-level slider?
        verbose : bool, default=True
            Whether to print progress.
        
        Attributes
        ----------
        estimators_ : list
            The estimators that were fit.
        Any : Any
            Attributes that will be collected from estimators.
        """

        # setup opts
        self.estimator = estimator
        self.dims = dims
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.top = top
        
        # setup attributes
        self.estimators_ = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, *args: Any):
        """Fit all estimators.
        
        Parameters
        ----------
        X : torch.Tensor
            The data to slide the estimator over.
        y : torch.Tensor
            The data to slide the estimator over.
        *args : Any
            Additional arguments to pass to the estimator.
        """
        
        # check types
        if (isinstance(X, torch.Tensor) == False) | (isinstance(y, torch.Tensor) == False):
            raise ValueError(f'X and y must be of type torch.Tensor but got {type(X)} and {type(y)}.')
        
        # check dimensions
        if len(X.shape) != len(y.shape):
            raise ValueError(f'X and y must have the same number of dimensions for alignment, but got {X.shape} and {y.shape}.')
        
        # check estimator
        if isinstance(self.estimator, sklearn.base.BaseEstimator):
            # move sliding axis to final dimension
            X = torch.moveaxis(X, self.dims[0], -1)
            y = torch.moveaxis(y, self.dims[0], -1)
            
            # setup our estimator depending on dims
            effective_dims = tuple((torch.arange(len(X.shape))[list(self.dims)] - 1).tolist())
            estimator_ = _Sliding_torch(estimator = self.estimator, dims = effective_dims[1:], n_jobs = None, top = False) if len(self.dims) > 1 else self.estimator.clone()
            
            # fit estimators
            context = Progressbar(enabled = self.verbose, desc = "Fitting estimators...", total = X.shape[-1])
            
            X_i = torch.arange(X.shape[-1]).to(torch.int32) if X.shape[-1] > 1 else torch.zeros((y.shape[-1],)).to(torch.int32)
            y_i = torch.arange(y.shape[-1]).to(torch.int32) if y.shape[-1] > 1 else torch.zeros((X.shape[-1],)).to(torch.int32)
            
            with context as progress_bar:
                self.estimators_ = [Parallel(n_jobs = self.n_jobs)
                                        (delayed(estimator_.clone().fit)
                                            (X[...,i], 
                                             y[...,j], 
                                             *args)
                                        for i, j in zip(X_i, y_i))][0]
        else:
            self.estimators_ = []
        
        return self

    def transform(self, X: torch.Tensor, y: Union[torch.Tensor, None] = None, *args) -> torch.Tensor:
        """Transform data in all estimators.
        
        Parameters
        ----------
        X : torch.Tensor
            The data to transform.
        y : Union[torch.Tensor, None], default=None
            The targets to transform.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        torch.Tensor
            The transformed data.
        """
        
        # check types
        if (isinstance(X, torch.Tensor) == False):
            raise ValueError(f'X must be of type torch.Tensor but got {type(X)}.')
        
        # check estimator
        if (self.estimators_ is None) & (callable(self.estimator) == False):
            raise ValueError('Estimators not fitted yet.')

        # check estimator type
        if isinstance(self.estimator, sklearn.base.BaseEstimator):
            # move sliding axis to final dimension
            X = torch.moveaxis(X, self.dims[0], -1)

            # make predictions
            return torch.stack(Parallel(n_jobs = self.n_jobs)
                                    (delayed(self.estimators_[i].transform)
                                        (X[...,i], 
                                         *args)
                                     for i in range(X.shape[-1])), self.dims[0])
        else:
            # move sliding axis to final dimension
            X = torch.moveaxis(X, self.dims[0], -1)
            y = torch.moveaxis(y, self.dims[0], -1)

            # setup our estimator depending on dims
            effective_dims = tuple((torch.arange(len(X.shape))[list(self.dims)] - 1).tolist())
            estimator_ = _Sliding_torch(estimator = self.estimator, dims = effective_dims[1:], n_jobs = None, top = False).transform if len(self.dims) > 1 else self.estimator
            
            # fit estimators
            X_i = torch.arange(X.shape[-1]).to(torch.int32) if X.shape[-1] > 1 else torch.zeros((y.shape[-1],)).to(torch.int32)
            y_i = torch.arange(y.shape[-1]).to(torch.int32) if y.shape[-1] > 1 else torch.zeros((X.shape[-1],)).to(torch.int32)
            
            Z = torch.stack(Parallel(n_jobs = self.n_jobs)
                                (delayed(estimator_)
                                    (X[...,i], 
                                     y[...,j], 
                                     *args)
                                 for i, j in zip(X_i, y_i)), self.dims[0] - 1)
            
            # if we're the top level, make sure trials remain first dimension
            if self.top:
                return Z.swapaxes(0, 1)
            else:
                return Z
    
    def fit_transform(self, X: torch.Tensor, y: Union[torch.Tensor, None] = None, *args) -> torch.Tensor:
        """Fit and transform data from all estimators.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data.
        y : torch.Tensor, default=None
            Target data.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        torch.Tensor
            Transformed data.
        """
        
        return self.fit(X, y = y, *args).transform(X, y = y, *args)
    
    def predict(self, X: torch.Tensor, y: Union[torch.Tensor, None] = None, *args) -> torch.Tensor:
        """Make predictions from all estimators.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data.
        y : torch.Tensor, default=None
            Target data.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        torch.Tensor
            Predictions.
        """
        
        # check types
        if (isinstance(X, torch.Tensor) == False):
            raise ValueError(f'X must be of type torch.Tensor but got {type(X)}.')
        
        # check estimator
        if (self.estimators_ is None) & (callable(self.estimator) == False):
            raise ValueError('Estimators not fitted yet.')

        # check estimator type
        if isinstance(self.estimator, sklearn.base.BaseEstimator):
            # move sliding axis to final dimension
            X = torch.moveaxis(X, self.dims[0], -1)

            # make predictions
            return torch.stack(Parallel(n_jobs = self.n_jobs)
                                    (delayed(self.estimators_[i].predict)
                                        (X[...,i], 
                                         *args)
                                     for i in range(X.shape[-1])), self.dims[0])
        else:
            # move sliding axis to final dimension
            X = torch.moveaxis(X, self.dims[0], -1)
            y = torch.moveaxis(y, self.dims[0], -1)

            # setup our estimator depending on dims
            effective_dims = tuple((torch.arange(len(X.shape))[list(self.dims)] - 1).tolist())
            estimator_ = _Sliding_torch(estimator = self.estimator, dims = effective_dims[1:], n_jobs = None, top = False).predict if len(self.dims) > 1 else self.estimator
            
            # fit estimators
            X_i = torch.arange(X.shape[-1]).to(torch.int32) if X.shape[-1] > 1 else torch.zeros((y.shape[-1],)).to(torch.int32)
            y_i = torch.arange(y.shape[-1]).to(torch.int32) if y.shape[-1] > 1 else torch.zeros((X.shape[-1],)).to(torch.int32)

            Z = torch.stack(Parallel(n_jobs = self.n_jobs)
                                (delayed(estimator_)
                                    (X[...,i], 
                                     y[...,j], 
                                     *args)
                                 for i, j in zip(X_i, y_i)), self.dims[0] - 1)
            
            # if we're the top level, make sure trials remain first dimension
            if self.top:
                return Z.swapaxes(0, 1)
            else:
                return Z
    
    def predict_proba(self, X: torch.Tensor, y: Union[torch.Tensor, None] = None, *args) -> torch.Tensor:
        """Make predictions from all estimators.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data.
        y : torch.Tensor, default=None
            Target data.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        torch.Tensor
            Predictions.
        """
        
        # check types
        if (isinstance(X, torch.Tensor) == False):
            raise ValueError(f'X must be of type torch.Tensor but got {type(X)}.')
        
        # check estimator
        if (self.estimators_ is None) & (callable(self.estimator) == False):
            raise ValueError('Estimators not fitted yet.')

        # check estimator type
        if isinstance(self.estimator, sklearn.base.BaseEstimator):
            # move sliding axis to final dimension
            X = torch.moveaxis(X, self.dims[0], -1)

            # make predictions
            return torch.stack(Parallel(n_jobs = self.n_jobs)
                                    (delayed(self.estimators_[i].predict_proba)
                                        (X[...,i], 
                                         *args)
                                     for i in range(X.shape[-1])), self.dims[0])
        else:
            # move sliding axis to final dimension
            X = torch.moveaxis(X, self.dims[0], -1)
            y = torch.moveaxis(y, self.dims[0], -1)

            # setup our estimator depending on dims
            effective_dims = tuple((torch.arange(len(X.shape))[list(self.dims)] - 1).tolist())
            estimator_ = _Sliding_torch(estimator = self.estimator, dims = effective_dims[1:], n_jobs = None, top = False).predict_proba if len(self.dims) > 1 else self.estimator
            
            # fit estimators
            X_i = torch.arange(X.shape[-1]).to(torch.int32) if X.shape[-1] > 1 else torch.zeros((y.shape[-1],)).to(torch.int32)
            y_i = torch.arange(y.shape[-1]).to(torch.int32) if y.shape[-1] > 1 else torch.zeros((X.shape[-1],)).to(torch.int32)

            Z = torch.stack(Parallel(n_jobs = self.n_jobs)
                                (delayed(estimator_)
                                    (X[...,i], 
                                     y[...,j], 
                                     *args)
                                 for i, j in zip(X_i, y_i)), self.dims[0] - 1)
            
            # if we're the top level, make sure trials remain first dimension
            if self.top:
                return Z.swapaxes(0, 1)
            else:
                return Z
    
    def collect(self, attr: str) -> torch.Tensor:
        """Collect an attribute from all fitted estimators.
        
        Parameters
        ----------
        attr : str
            Attribute to collect.
        
        Returns
        -------
        torch.Tensor
            Tensor of shape (n_estimators, ...)
        """
        
        if isinstance(self.estimators_[0], _Sliding_torch):
            return torch.stack([self.estimators_[i].collect(attr) for i in range(len(self.estimators_))])
        else:
            if hasattr(self.estimators_[0], attr) == False:
                raise AttributeError(f'Estimator of type `{type(self.estimators_[0])}` does not have attribute `{attr}`.')
            
            return torch.stack([getattr(self.estimators_[i], attr) for i in range(len(self.estimators_))])
    
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        _Sliding_torch
            Cloned class.
        """
        
        return _Sliding_torch(estimator = self.estimator, dims = self.dims, n_jobs = self.n_jobs, top = self.top, verbose = self.verbose)

class Sliding(sklearn.base.BaseEstimator):
    """Implements a sliding estimator that allows you to fit estimators iteratively over a set of dimensions.
    
    Parameters
    ----------
    estimator : Callable, sklearn.base.BaseEstimator
        Estimator to use.
    dims : Union[int, tuple, list, np.ndarray, torch.Tensor], default=-1
        Dimensions to slide over.
    n_jobs : Union[int, None], default=None
        Number of jobs to run in parallel.
    top : bool, default=True
        Is this a top-level estimator?
    verbose : bool, default=False
        Whether to print progress.
    
    Attributes
    ----------
    estimators_ : list
        List of fitted estimators.
    
    Notes
    -----
    This class generally expects that your input data is of shape (n_trials, [...], n_channels, [...]). Make sure that your data and dimension selection is appropriate for the estimator you wish to fit.
    Note also that, when fitting estimators, you _must_ have an equal number of dimensions in X and y. If you do not, please simply pad to the same dimension length.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import Sliding, Decoder
    >>> X = torch.normal(0, 1, (240, 50, 4, 100)) # trials x searchlights x channels x time
    >>> y = torch.normal(0, 1, (240, 1, 5, 100)) # trials x searchlights x outcomes x time
    >>> decoder = Decoder(alphas = torch.logspace(-5, 10, 20))
    >>> sliding = Sliding(estimator = decoder, dims = (1, 3), n_jobs = 4) # slide over searchlights and time
    >>> sliding.fit(X, y)
    >>> patterns = sliding.collect('pattern_')
    >>> patterns.shape
    torch.Size([50, 100, 4, 5])
    """
    
    def __new__(self, estimator: Union[Callable, sklearn.base.BaseEstimator], dims: Union[int, tuple, list, np.ndarray, torch.Tensor] = -1, n_jobs: Union[int, None] = None, top: bool = True, verbose: bool = False):
        """Obtain a new sliding estimator.
        
        Parameters
        ----------
        estimator : Callable, sklearn.base.BaseEstimator
            Estimator to use.
        dims : Union[int, tuple, list, np.ndarray, torch.Tensor], default=-1
            Dimensions to slide over.
        n_jobs : Union[int, None], default=None
            Number of jobs to run in parallel.
        top : bool, default=True
            Is this a top-level estimator?
        verbose : bool, default=False
            Whether to print progress.
        """
        
        return_type = _Sliding_torch
        
        # check dims
        if isinstance(dims, int):
            dims = (dims,)
        elif isinstance(dims, list):
            dims = tuple(dims)
        elif isinstance(dims, np.ndarray):
            return_type = _Sliding_numpy
            dims = tuple(dims.flatten())
        elif isinstance(dims, torch.Tensor):
            dims = tuple(dims.flatten().tolist())
        elif isinstance(dims, tuple):
            dims = dims
        else:
            raise ValueError(f'`dims` must be an integer, tuple, list, numpy array or torch tensor, but got {type(dims)}.')
        
        # check estimator
        if (isinstance(estimator, sklearn.base.BaseEstimator) == False) & (callable(estimator) == False):
            raise ValueError(f'`estimator` must be sklearn.base.BaseEstimator or Callable, but got {type(estimator)}.')
        
        self.estimator = estimator
        self.dims = dims
        self.top = top
        self.verbose = verbose
        
        return return_type(estimator, dims = dims, n_jobs = n_jobs, top = top, verbose = verbose)
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], *args):
        """Fit the estimator.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Input data.
        y : Union[np.ndarray, torch.Tensor]
            Target data.
        *args
            Additional arguments.
        """

        raise NotImplementedError('This method is not implemented for the base class.')
    
    def transform(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor, None] = None, *args) -> Union[np.ndarray, torch.Tensor]:
        """Transform the data.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Input data.
        y : Union[np.ndarray, torch.Tensor, None], default=None
            Target data.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Transformed data.
        """
        
        raise NotImplementedError('This method is not implemented for the base class.')
    
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], *args) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform the data.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Input data.
        y : Union[np.ndarray, torch.Tensor]
            Target data.
        *args : Any
            Additional arguments.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Transformed data.
        """

        raise NotImplementedError('This method is not implemented for the base class.')

    def predict(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor, None] = None, *args) -> Union[np.ndarray, torch.Tensor]:
        """Predict the targets.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Input data.
        y : Union[np.ndarray, torch.Tensor, None], default=None
            Target data.
        *args :
            Additional arguments.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Predicted targets.
        """

        raise NotImplementedError('This method is not implemented for the base class.')
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor, None] = None, *args) -> Union[np.ndarray, torch.Tensor]:
        """Predict the probabilities.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Input data.
        y : Union[np.ndarray, torch.Tensor, None], default=None
            Target data.
        *args :
            Additional arguments.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Predicted probabilities.
        """
        
        raise NotImplementedError('This method is not implemented for the base class.')
        
    def collect(self, attr: str) -> Union[np.ndarray, torch.Tensor]:
        """Collect the attribute of the estimators.
        
        Parameters
        ----------
        attr : str
            Attribute to collect.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Collected attribute.
        """
        
        raise NotImplementedError('This method is not implemented for the base class.')
    
    def clone(self):
        """Clone this class.

        Returns
        -------
        Sliding
            Cloned class.
        """

        return Sliding(estimator = self.estimator, dims = self.dims, n_jobs = self.n_jobs, top = self.top, verbose = self.verbose)