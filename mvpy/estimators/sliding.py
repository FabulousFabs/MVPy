'''
A collection of estimators that allow for sliding other estimators over a dimension of the data.
'''

import numpy as np
import torch
import sklearn

from joblib import Parallel, delayed

from ..utilities import Progressbar
from .. import metrics

from typing import Union, Any, Callable, Optional, Dict, Tuple

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
    
    def decision_function(self, X: np.ndarray, y: Union[np.ndarray, None] = None, *args) -> np.ndarray:
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
                                    (delayed(self.estimators_[i].decision_function)
                                        (X[...,i], 
                                         *args)
                                     for i in range(X.shape[-1])), self.dims[0])
        else:
            # move sliding axis to final dimension
            X = np.moveaxis(X, self.dims[0], -1)
            y = np.moveaxis(y, self.dims[0], -1)

            # setup our estimator depending on dims
            effective_dims = tuple((np.arange(len(X.shape))[list(self.dims)] - 1).tolist())
            estimator_ = _Sliding_numpy(estimator = self.estimator, dims = effective_dims[1:], n_jobs = None, top = False).decision_function if len(self.dims) > 1 else self.estimator
            
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
    
    def predict_proba(self, X: np.ndarray, y: Union[np.ndarray, None] = None, *args) -> np.ndarray:
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
    
    def score(self, X: np.ndarray, y: np.ndarray, metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None) -> np.ndarray:
        """Score the estimator.
        
        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The output data.
        metric: Optional[Metric | Tuple[Metric]], default=None
            The metric to use. If ``None``, default to the underlying estimator's default metric.
        
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
            metric = self.estimators_[0].metric_
        
        return metrics.score(self, metric, X, y)
    
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
    
    def decision_function(self, X: torch.Tensor, y: Union[torch.Tensor, None] = None, *args) -> torch.Tensor:
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
                                    (delayed(self.estimators_[i].decision_function)
                                        (X[...,i], 
                                         *args)
                                     for i in range(X.shape[-1])), self.dims[0])
        else:
            # move sliding axis to final dimension
            X = torch.moveaxis(X, self.dims[0], -1)
            y = torch.moveaxis(y, self.dims[0], -1)

            # setup our estimator depending on dims
            effective_dims = tuple((torch.arange(len(X.shape))[list(self.dims)] - 1).tolist())
            estimator_ = _Sliding_torch(estimator = self.estimator, dims = effective_dims[1:], n_jobs = None, top = False).decision_function if len(self.dims) > 1 else self.estimator
            
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
    
    def score(self, X: torch.Tensor, y: torch.Tensor, metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None) -> torch.Tensor:
        """Score the estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            The input data.
        y : torch.Tensor
            The output data.
        metric: Optional[Metric | Tuple[Metric]], default=None
            The metric to use. If ``None``, default to the underlying estimator's default metric.
        
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
            metric = self.estimators_[0].metric_
        
        return metrics.score(self, metric, X, y)
    
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
    
    This is particularly useful when we have, for example, a temporal dimension in our data
    such that, for example, we have neural data :math:`X` ``(n_trials, n_channels, n_timepoints)``
    and class labels :math:`y` ``(n_trials, n_features, n_timepoints)`` and want to fit a separate
    classifier at each time step. In this case, we can wrap our classifier object in 
    :py:class:`~mvpy.estimators.Sliding` with ``dims=(-1,)`` to automatically fit our classifiers
    across all timepoints.
    
    Parameters
    ----------
    estimator : Callable | sklearn.base.BaseEstimator
        Estimator to use. Note that this must expose a ``clone()`` method.
    dims : int | Tuple[int] | List[int] | np.ndarray | torch.Tensor, default=-1
        Dimensions to slide over. Note that types are inferred here, defaulting to torch. If you are fitting a numpy estimator, please specify ``dims`` as ``np.ndarray``.
    n_jobs : Optional[int], default=None
        Number of jobs to run in parallel.
    top : bool, default=True
        Is this a top-level estimator? If multiple ``dims`` are specified, this will be ``False`` in recursive :py:class:`~mvpy.estimators.Sliding` objects.
    verbose : bool, default=False
        Should progress be reported verbosely?
    
    Attributes
    ----------
    estimator : Callable | sklearn.base.BaseEstimator
        Estimator to use. Note that this must expose a ``clone()`` method.
    dims : int | Tuple[int] | List[int] | np.ndarray | torch.Tensor, default=-1
        Dimensions to slide over. Note that types are inferred here, defaulting to torch. If you are fitting a numpy estimator, please specify ``dims`` as ``np.ndarray``.
    n_jobs : Optional[int], default=None
        Number of jobs to run in parallel.
    top : bool, default=True
        Is this a top-level estimator? If multiple ``dims`` are specified, this will be ``False`` in recursive :py:class:`~mvpy.estimators.Sliding` objects.
    verbose : bool, default=False
        Should progress be reported verbosely?
    estimators_ : List[Callable, sklearn.base.BaseEstimator]
        List of fitted estimators.
    
    Notes
    -----
    When fitting estimators using :py:meth:`~mvpy.estimators.Sliding.fit`, ``X`` and ``y`` must have 
    the same number of dimensions. If this is not the case, please pad or expand your data appropriately.
    
    Examples
    --------
    If, for example, we have :math:`X` ``(n_trials, n_frequencies, n_channels, n_timepoints)`` 
    and :math:`y` ``(n_trials, n_frequencies, n_features, n_timepoints)`` and we want to slide
    a :py:class:`~mvpy.estimators.RidgeDecoder` over ``(n_frequencies, n_timepoints)``, we can
    do:
    
    >>> import torch
    >>> from mvpy.estimators import Sliding, RidgeDecoder
    >>> X = torch.normal(0, 1, (240, 5, 64, 100))
    >>> y = torch.normal(0, 1, (240, 1, 5, 100))
    >>> decoder = RidgeDecoder(
    >>>     alphas = torch.logspace(-5, 10, 20)
    >>> )
    >>> sliding = Sliding(
    >>>     estimator = decoder, 
    >>>     dims = (1, 3), 
    >>>     n_jobs = 4
    >>> ).fit(X, y)
    >>> patterns = sliding.collect('pattern_')
    >>> patterns.shape
    torch.Size([5, 100, 64, 5])
    """
    
    def __new__(self, estimator: Union[Callable, sklearn.base.BaseEstimator], dims: Union[int, tuple, list, np.ndarray, torch.Tensor] = -1, n_jobs: Union[int, None] = None, top: bool = True, verbose: bool = False):
        """Obtain a new sliding estimator.
        
        Parameters
        ----------
        estimator : Callable | sklearn.base.BaseEstimator
            Estimator to use. Note that this must expose a ``clone()`` method.
        dims : int | Tuple[int] | List[int] | np.ndarray | torch.Tensor, default=-1
            Dimensions to slide over. Note that types are inferred here, defaulting to torch. If you are fitting a numpy estimator, please specify ``dims`` as ``np.ndarray``.
        n_jobs : Optional[int], default=None
            Number of jobs to run in parallel.
        top : bool, default=True
            Is this a top-level estimator? If multiple ``dims`` are specified, this will be ``False`` in recursive :py:class:`~mvpy.estimators.Sliding` objects.
        verbose : bool, default=False
            Should progress be reported verbosely?
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
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], *args) -> "Sliding":
        """Fit the sliding estimators.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of arbitrary shape.
        y : np.ndarray | torch.Tensor
            Target data of arbitrary shape.
        *args
            Additional arguments to pass to estimators.
        
        Returns
        -------
        sliding : mvpy.estimators.Sliding
            The fitted sliding estimator.
        """

        raise NotImplementedError('This method is not implemented for the base class.')
    
    def transform(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor, None] = None, *args) -> Union[np.ndarray, torch.Tensor]:
        """Transform the data.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of arbitrary shape.
        y : Optional[np.ndarray | torch.Tensor], default=None
            Target data of arbitrary shape.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        Z : np.ndarray | torch.Tensor
            Transformed data of arbitrary shape.
        """
        
        raise NotImplementedError('This method is not implemented for the base class.')
    
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], *args) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform the data.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of arbitrary shape.
        y : Optional[np.ndarray | torch.Tensor], default=None
            Target data of arbitrary shape.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        Z : np.ndarray | torch.Tensor
            Transformed data of arbitrary shape.
        """

        raise NotImplementedError('This method is not implemented for the base class.')

    def predict(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor, None] = None, *args) -> Union[np.ndarray, torch.Tensor]:
        """Predict the targets.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of arbitrary shape.
        y : Optional[np.ndarray | torch.Tensor], default=None
            Target data of arbitrary shape.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        y_h : np.ndarray | torch.Tensor
            Predicted data of arbitrary shape.
        """

        raise NotImplementedError('This method is not implemented for the base class.')
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor, None] = None, *args) -> Union[np.ndarray, torch.Tensor]:
        """Predict the probabilities.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of arbitrary shape.
        y : Optional[np.ndarray | torch.Tensor], default=None
            Target data of arbitrary shape.
        *args : Any
            Additional arguments.
        
        Returns
        -------
        p : np.ndarray | torch.Tensor
            Probabilities of arbitrary shape.
        """
        
        raise NotImplementedError('This method is not implemented for the base class.')
    
    def score(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None) -> Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
        """Make predictions from :math:`X` and score against :math:`y`.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of arbitrary shape.
        y : np.ndarray | torch.Tensor
            Output data of arbitrary shape.
        metric : Optional[Metric | Tuple[Metric]], default=None
            Metric or tuple of metrics to compute. If ``None``, defaults to the metric specified for the underlying estimator.
        
        Returns
        -------
        score : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
            Scores of shape arbitrary shape.
        
        .. warning::
            If multiple values are supplied for ``metric``, this function will
            output a dictionary of ``{Metric.name: score, ...}`` rather than
            a stacked array. This is to provide consistency across cases where
            metrics may or may not differ in their output shapes.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
        
    def collect(self, attr: str) -> Union[np.ndarray, torch.Tensor]:
        """Collect an attribute from all estimators.
        
        Parameters
        ----------
        attr : str
            Attribute to collect from all fitted estimators.
        
        Returns
        -------
        attr : np.ndarray | torch.Tensor
            Collected attribute of shape ``(*dims[, ...])``.
        """
        
        raise NotImplementedError('This method is not implemented for the base class.')
    
    def clone(self):
        """Clone this class.

        Returns
        -------
        sliding : mvpy.estimators.Sliding
            Cloned class.
        """

        return Sliding(
            estimator = self.estimator, 
            dims = self.dims, 
            n_jobs = self.n_jobs, 
            top = self.top, 
            verbose = self.verbose
        )