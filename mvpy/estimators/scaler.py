'''
A collection of estimators for scaling data.
'''

import numpy as np
import torch
import sklearn

from typing import Union, Any

class _Scaler_numpy(sklearn.base.BaseEstimator):
    r"""A standard scaler for numpy arrays. Note that this class is not exported and should not be used directly.
    
    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    
    with_std : bool, default=True
        If True, scale the data to unit variance.
    
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    
    copy : bool, default=False
        If True, the data will be copied.
    
    Attributes
    ----------
    shape_ : tuple
        The shape of the data.
    
    mean_ : Union[np.ndarray, torch.Tensor]
        The mean of the data.
    
    var_ : Union[np.ndarray, torch.Tensor]
        The variance of the data.
    
    scale_ : Union[np.ndarray, torch.Tensor]
        The scale of the data.
    
    Notes
    -----
    This is a scaler analogous to sklearn.preprocessing.StandardScaler, except that here we support n-dimensional arrays,
    and use a degrees-of-freedom correction for computing variances (without the step-wise fitting).
    
    By default, this scaler will compute:
    
    .. math::
        z = \frac{x - \mu}{\sigma}
    
    where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation of the data.
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True, dims: Union[list, tuple, int, None] = None):
        """Obtain a scaler.
        
        Parameters
        ----------
        with_mean : bool, default=True
            If True, center the data before scaling.
        with_std : bool, default=True
            If True, scale the data to unit variance.
        dims : int, list or tuple of ints, default=None
            The dimensions over which to scale (None for first dimension).
        copy : bool, default=False
            If True, the data will be copied.
        """

        # store options
        self.dims = self.dims_ = dims
        self.with_mean = self.with_mean_ = with_mean
        self.with_std = self.with_std_ = with_std
        
        self.shape_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray, *args: Any, sample_weight: Union[np.ndarray, None] = None):
        """Fit the scaler.
        
        Parameters
        ----------
        X : np.ndarray
            The data.
        args : Any
            Additional arguments.
        sample_weight : np.ndarray, default=None
            The sample weights.
        """
        
        # check type
        if not isinstance(X, np.ndarray):
            raise TypeError(f'`X` must be a numpy array, but got {type(X)}.')
        
        # check dimensions
        if self.dims_ is None:
            self.dims_ = np.array([0])
        
        # set type
        self.dims_ = np.array(self.dims_).astype(int)
        self.dims_ = tuple(self.dims_)
        
        # check sample weights
        if sample_weight is not None:
            if not isinstance(sample_weight, np.ndarray):
                raise TypeError(f'`sample_weight` must be a numpy array, but got {type(sample_weight)}.')
            
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f'`sample_weight` must match `X` in first dimension, but got {sample_weight.shape} and {X.shape}.')

            # setup dims
            sample_weight = sample_weight.reshape((X.shape[0],) + (1,) * (len(X.shape) - 1))
        
        # save expected dimensions
        self.shape_ = X.shape
        
        # reset if needed
        if (self.mean_ is not None) | (self.var_ is not None) | (self.scale_ is not None):
            self.mean_ = None
            self.var_ = None
            self.scale_ = None
        
        # fit scaler
        if sample_weight is not None:
            n = (sample_weight != 0.0).sum(axis = 0, keepdims = True)
            self.mean_ = (X * sample_weight).sum(axis = self.dims_, keepdims = True) / sample_weight.sum(axis = self.dims_, keepdims = True)
            self.var_ = ((X - self.mean_) ** 2 * sample_weight).sum(axis = self.dims_, keepdims = True) / ((n - 1) / n * sample_weight.sum(axis = self.dims_, keepdims = True))
            self.scale_ = np.sqrt(self.var_)
        else:
            self.mean_ = X.mean(axis = self.dims_, keepdims = True)
            self.var_ = X.var(axis = self.dims_, keepdims = True, ddof = 1)
            self.scale_ = np.sqrt(self.var_)
        
        # ensure no NaNs will be produced
        self.scale_[(self.scale_ == 0.0) | (np.isnan(self.scale_))] = 1.0
        
        return self

    def transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        """Transform the data using scaler.
        
        Parameters
        ----------
        X : np.ndarray
            The data.
        args : Any
            Additional arguments.

        Returns
        -------
        np.ndarray
            The transformed data.
        """

        # check fit
        if (self.with_mean_ & (self.mean_ is None)) | (self.with_std_ & (self.scale_ is None)):
            raise ValueError('The scaler has not been fitted yet.')
        
        # demean
        if self.with_mean_:
            X = X - self.mean_
        
        # scale
        if self.with_std_:
            X = X / self.scale_
        
        return X

    def inverse_transform(self, X: np.ndarray, *args: any) -> np.ndarray:
        """Invert the transform of the data.
        
        Parameters
        ----------
        X : np.ndarray
            The data.
        args : Any
            Additional arguments.
        
        Returns
        -------
        np.ndarray
            The inverse transformed data.
        """
        
        # check fit
        if (self.with_mean_ & (self.mean_ is None)) | (self.with_std_ & (self.scale_ is None)):
            raise ValueError('The scaler has not been fitted yet.')
        
        # re-scale
        if self.with_std_:
            X = X * self.scale_
        
        # re-mean
        if self.with_mean_:
            X = X + self.mean_
        
        return X
    
    def fit_transform(self, X: np.ndarray, *args: Any, sample_weight: Union[np.ndarray, None] = None) -> np.ndarray:
        """Fit and transform the data in one step.
        
        Parameters
        ----------
        X : np.ndarray
            The data.
        args : Any
            Additional arguments.
        sample_weight : np.ndarray, default=None
            The sample weights.
        
        Returns
        -------
        np.ndarray
            The transformed data.
        """
        
        return self.fit(X, *args, sample_weight = sample_weight).transform(X, *args)
    
    def clone(self):
        """Obtain a clone of this class.
        
        Returns
        -------
        _Scaler_numpy
            The clone.
        """
        
        return _Scaler_numpy(with_mean = self.with_mean_, with_std = self.with_std_, dims = self.dims_)

class _Scaler_torch(sklearn.base.BaseEstimator):
    r"""A standard scaler for torch tensors. Note that this class is not exported and should not be called directly.
    
    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    
    with_std : bool, default=True
        If True, scale the data to unit variance.
    
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    
    copy : bool, default=False
        If True, the data will be copied.
    
    Attributes
    ----------
    shape_ : tuple
        The shape of the data.
    
    mean_ : Union[np.ndarray, torch.Tensor]
        The mean of the data.
    
    var_ : Union[np.ndarray, torch.Tensor]
        The variance of the data.
    
    scale_ : Union[np.ndarray, torch.Tensor]
        The scale of the data.
    
    Notes
    -----
    This is a scaler analogous to sklearn.preprocessing.StandardScaler, except that here we support n-dimensional arrays,
    and use a degrees-of-freedom correction for computing variances (without the step-wise fitting).
    
    By default, this scaler will compute:
    
    .. math::
        z = \frac{x - \mu}{\sigma}
    
    where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation of the data.
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True, dims: Union[list, tuple, int, None] = None):
        """Obtain a scaler.
        
        Parameters
        ----------
        with_mean : bool, default=True
            If True, center the data before scaling.
        with_std : bool, default=True
            If True, scale the data to unit variance.
        dims : int, list or tuple of ints, default=None
            The dimensions over which to scale (None for first dimension).
        copy : bool, default=False
            If True, the data will be copied.
        """

        # store options
        self.dims = self.dims_ = dims
        self.with_mean = self.with_mean_ = with_mean
        self.with_std = self.with_std_ = with_std
                
        self.shape_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
    
    def fit(self, X: torch.Tensor, *args: Any, sample_weight: Union[torch.Tensor, None] = None):
        """Fit the scaler.
        
        Parameters
        ----------
        X : torch.Tensor
            The data.
        args : Any
            Additional arguments.
        sample_weight : torch.Tensor, default=None
            The sample weights.
        """
        
        # check type
        if not isinstance(X, torch.Tensor):
            raise TypeError(f'`X` must be a torch tensor, but got {type(X)}.')
        
        # check dimensions
        if self.dims_ is None:
            self.dims_ = torch.tensor([0])
        
        # set type
        if isinstance(self.dims_, torch.Tensor) == False:
            self.dims_ = torch.tensor(self.dims_)
        self.dims_ = self.dims_.to(torch.int32)
        self.dims_ = tuple(self.dims_.cpu().numpy())
        
        # check sample weights
        if sample_weight is not None:
            if not isinstance(sample_weight, torch.Tensor):
                raise TypeError(f'`sample_weight` must be a torch tensor, but got {type(sample_weight)}.')
            
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f'`sample_weight` must match `X` in first dimension, but got {sample_weight.shape} and {X.shape}.')

            # setup dims
            sample_weight = sample_weight.reshape((X.shape[0],) + (1,) * (len(X.shape) - 1))
        
        # save expected dimensions
        self.shape_ = X.shape
        
        # reset if needed
        if (self.mean_ is not None) | (self.var_ is not None) | (self.scale_ is not None):
            self.mean_ = None
            self.var_ = None
            self.scale_ = None
        
        # fit scaler
        if sample_weight is not None:
            n = (sample_weight != 0.0).sum(dim = 0, keepdim = True)
            self.mean_ = (X * sample_weight).sum(dim = self.dims_, keepdim = True) / sample_weight.sum(dim = self.dims_, keepdim = True)
            self.var_ = ((X - self.mean_) ** 2 * sample_weight).sum(dim = self.dims_, keepdim = True) / ((n - 1) / n * sample_weight.sum(dim = self.dims_, keepdim = True))
            self.scale_ = torch.sqrt(self.var_)
        else:
            self.mean_ = X.mean(dim = self.dims_, keepdim = True)
            self.var_ = X.var(dim = self.dims_, keepdim = True)
            self.scale_ = torch.sqrt(self.var_)
        
        # ensure no NaNs will be produced
        self.scale_[(self.scale_ == 0.0) | (torch.isnan(self.scale_))] = 1.0
        
        return self

    def transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
        """Transform the data using scaler.
        
        Parameters
        ----------
        X : torch.Tensor
            The data.
        args : Any
            Additional arguments.

        Returns
        -------
        torch.Tensor
            The transformed data.
        """

        # check fit
        if (self.with_mean_ & (self.mean_ is None)) | (self.with_std_ & (self.scale_ is None)):
            raise ValueError('The scaler has not been fitted yet.')
        
        # make sure we don't in-place anything
        X = X.clone()
        
        # demean
        if self.with_mean_:
            X = X - self.mean_
        
        # scale
        if self.with_std_:
            X = X / self.scale_
        
        return X

    def inverse_transform(self, X: torch.Tensor, *args: any) -> torch.Tensor:
        """Invert the transform of the data.
        
        Parameters
        ----------
        X : torch.Tensor
            The data.
        args : Any
            Additional arguments.
        
        Returns
        -------
        torch.Tensor
            The inverse transformed data.
        """
        
        # check fit
        if (self.with_mean_ & (self.mean_ is None)) | (self.with_std_ & (self.scale_ is None)):
            raise ValueError('The scaler has not been fitted yet.')
        
        # make sure we don't in-place anything
        X = X.clone()
        
        # re-scale
        if self.with_std_:
            X = X * self.scale_
        
        # re-mean
        if self.with_mean_:
            X = X + self.mean_
        
        return X
    
    def fit_transform(self, X: torch.Tensor, *args: Any, sample_weight: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """Fit and transform the data in one step.
        
        Parameters
        ----------
        X : torch.Tensor
            The data.
        args : Any
            Additional arguments.
        sample_weight : torch.Tensor, default=None
            The sample weights.
        
        Returns
        -------
        torch.Tensor
            The transformed data.
        """
        
        return self.fit(X, *args, sample_weight = sample_weight).transform(X, *args)

    def clone(self):
        """Obtain a clone of this class.
        
        Returns
        -------
        _Scaler_torch
            The clone.
        """
        
        return _Scaler_torch(with_mean = self.with_mean, with_std = self.with_std_, dims = self.dims_)

class Scaler(sklearn.base.BaseEstimator):
    r"""A standard scaler akin to sklearn.preprocessing.StandardScaler. See notes for some differences.
    
    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    
    with_std : bool, default=True
        If True, scale the data to unit variance.
    
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    
    copy : bool, default=False
        If True, the data will be copied.
    
    Attributes
    ----------
    shape_ : tuple
        The shape of the data.
    
    mean_ : Union[np.ndarray, torch.Tensor]
        The mean of the data.
    
    var_ : Union[np.ndarray, torch.Tensor]
        The variance of the data.
    
    scale_ : Union[np.ndarray, torch.Tensor]
        The scale of the data.
    
    Notes
    -----
    This is a scaler analogous to sklearn.preprocessing.StandardScaler, except that here we support n-dimensional arrays,
    and use a degrees-of-freedom correction for computing variances (without the step-wise fitting).
    
    By default, this scaler will compute:
    
    .. math::
        z = \frac{x - \mu}{\sigma}
    
    where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation of the data.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import Scaler
    >>> X = torch.normal(5, 10, (1000, 5))
    >>> print(X.std(0))
    tensor([ 9.7033, 10.2510, 10.2483, 10.1274, 10.2013])
    >>> scaler = Scaler().fit(X)
    >>> X_s = scaler.transform(X)
    >>> print(X_s.std(0))
    tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
    >>> X_i = scaler.inverse_transform(X_s)
    >>> print(X_i.std(0))
    tensor([ 9.7033, 10.2510, 10.2483, 10.1274, 10.2013])
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, dims: Union[list, tuple, int, None] = None):
        """Obtain a scaler.
        
        Parameters
        ----------
        with_mean : bool, default=True
            If True, center the data before scaling.
        with_std : bool, default=True
            If True, scale the data to unit variance.
        dims : int, list or tuple of ints, default=None
            The dimensions over which to scale (None for first dimension).
        """

        self.with_mean = with_mean
        self.with_std = with_std
        self.dims = dims
    
    def _get_estimator(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> sklearn.base.BaseEstimator:
        """Given the data, determine which scaler to use.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The data.
        args : Any
            Additional arguments.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The scaler.
        """
        
        if isinstance(X, torch.Tensor):
            return _Scaler_torch
        elif isinstance(X, np.ndarray):
            return _Scaler_numpy
        
        raise TypeError(f'`X` must be either torch.Tensor or np.ndarray, but got {type(X)}.')
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], *args: Any, sample_weight: Union[np.ndarray, torch.Tensor, None] = None) -> Any:
        """Fit the scaler.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The data.
        args : Any
            Additional arguments.
        sample_weight : Union[np.ndarray, torch.Tensor], default=None
            The sample weights.
        """
        
        return self._get_estimator(X, *args)(with_mean = self.with_mean, with_std = self.with_std, dims = self.dims).fit(X, *args, sample_weight = sample_weight)
    
    def transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Transform the data using scaler.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The data.
        args : Any
            Additional arguments.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The transformed data.
        """

        return self._get_estimator(X, *args)(with_mean = self.with_mean, with_std = self.with_std, dims = self.dims).transform(X, *args)
    
    def inverse_transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Invert the transform of the data.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The data.
        args : Any
            Additional arguments.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The inverse transformed data.
        """
        
        return self._get_estimator(X, *args)(with_mean = self.with_mean, with_std = self.with_std, dims = self.dims).inverse_transform(X, *args)
    
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any, sample_weight: Union[np.ndarray, torch.Tensor, None] = None) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform the data in one step.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The data.
        args : Any
            Additional arguments.
        sample_weight : Union[np.ndarray, torch.Tensor], default=None
            The sample weights.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The transformed data.
        """
        
        return self._get_estimator(X, *args)(with_mean = self.with_mean, with_std = self.with_std, dims = self.dims).fit_transform(X, *args, sample_weight = sample_weight)
    
    def to_torch(self):
        """Selet the torch scaler. Note that this cannot be called for conversion.
        
        Returns
        -------
        _Scaler_torch
            The torch scaler.
        """
        
        return self._get_estimator(torch.tensor([1]))(with_mean = self.with_mean, with_std = self.with_std, dims = self.dims)
    
    def to_numpy(self):
        """Selet the numpy scaler. Note that this cannot be called for conversion.

        Returns
        -------
        _Scaler_numpy
            The numpy scaler.
        """

        return self._get_estimator(np.array([1]))(with_mean = self.with_mean, with_std = self.with_std, dims = self.dims)
    
    def clone(self):
        """Obtain a clone of this class.
        
        Returns
        -------
        Scaler
            The clone.
        """
        
        return Scaler(with_mean = self.with_mean, with_std = self.with_std, dims = self.dims)
    
    def copy(self):
        """Obtain a copy of this class.

        Returns
        -------
        Scaler
            The copy.
        """
        
        return self.clone()