'''
A collection of estimators for robustly scaling data.
'''

import numpy as np
import torch
import sklearn

import math
import operator
from functools import reduce

from typing import Union, Any

class _RobustScaler_numpy(sklearn.base.BaseEstimator):
    """A robust scaler for torch tensors. Note that this class is not exported and should not be called directly.
    
    Parameters
    ----------
    with_centering : bool, default=True
        If True, center the data before scaling.
    with_scaling : bool, default=True
        If True, scale the data to unit variance.
    quantile_range : tuple[float, float], default=(25.0, 75.0)
        Tuple describing the quantiles.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    
    Attributes
    ----------
    with_centering : bool, default=True
        If True, center the data before scaling.
    with_scaling : bool, default=True
        If True, scale the data to unit variance.
    quantile_range : tuple[float, float], default=(25.0, 75.0)
        Tuple describing the quantiles.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    dims_ : tuple[int], default=None
        Tuple specifying the dimensions to scale over.
    centre_ : np.ndarray, default=None
        The centre of each feature of shape ``X``.
    scale_ : np.ndarray, default=None
        The scale of each feature of shape ``X`.
    """
    
    def __init__(self, with_centering: bool = True, with_scaling: bool = True, quantile_range: tuple[float] = (25.0, 75.0), dims: Union[list, tuple, int, None] = None):
        """Obtain a robust scaler.
        
        Parameters
        ----------
        with_centering : bool, default=True
            If True, center the data before scaling.
        with_scaling : bool, default=True
            If True, scale the data to unit variance.
        quantile_range : tuple[float, float], default=(25.0, 75.0)
            Tuple describing the quantiles.
        dims : int, list or tuple of ints, default=None
            The dimensions over which to scale (None for first dimension).
        """

        # store options
        self.dims = dims
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        
        # setup internals
        self.dims_ = dims
        self.centre_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray, *args: Any) -> "_RobustScaler_numpy":
        """Fit the scaler.
        
        Parameters
        ----------
        X : np.ndarray
            The data.
        args : Any
            Additional arguments.
        
        Returns
        -------
        scaler : _RobustScaler_numpy
            The fitted scaler.
        """
        
        # check type
        if not isinstance(X, np.ndarray):
            raise TypeError(f'`X` must be of type np.ndarray, but got {type(X)}.')
        
        # check dimensions
        if self.dims_ is None:
            self.dims_ = np.array([0])
        
        # set type
        self.dims_ = np.array(self.dims_).astype(int)
        self.dims_ = tuple(self.dims_)
        
        # save number of dims
        nd = X.ndim
        
        # normalise negatives (unique, sorted)
        red_dims = tuple(sorted({d % nd for d in self.dims_}))
        keep_dims = tuple([i for i in range(nd) if i not in red_dims])
        
        # permute so reduced dims come first, then kept dims
        perm = red_dims + keep_dims
        X_p = np.transpose(X, axes = perm)
        X_p = np.ascontiguousarray(X_p)
        
        # flatten reduced dims (leading axis R)
        R = 1 if len(red_dims) == 0 else reduce(operator.mul, (X.shape[d] for d in red_dims), 1)
        keep_shape = tuple(X.shape[d] for d in keep_dims)
        X_f = X_p.reshape(R, *keep_shape)
        
        # sort once along flattened reduction axis
        X_s = np.sort(X_f, axis = 0)
        
        # setup helper
        def q_interp(q: float) -> np.ndarray:
            """Interpolate quantile.
            
            Parameters
            ----------
            q : float
                Quantile in [0, 1].
            
            Returns
            -------
            q : np.ndarray
                The quantile of shape (n_features,).
            """
            
            # compute pos
            virtual_pos = q * (X_s.shape[0] - 1)
            pos = math.floor(virtual_pos)
            
            # handle uneven case (median easily defined)
            if (X_s.shape[0] % 2) == 1:
                return X_s[pos]
            
            # handle even by selecting both contenders
            next_pos = pos + 1
            
            if next_pos >= X_s.shape[0]:
                pos -= 1
                next_pos -= 1
            
            if pos < 0:
                pos = 0
                next_pos = 0

            # LERP between values
            gamma = virtual_pos - pos
            lv, hv = X_s[pos], X_s[next_pos]
            X_lerp = (1 - gamma) * lv + gamma * hv
            
            return X_lerp
        
        # determine quantiles with helper
        q_low = q_interp(self.quantile_range[0] / 100.0)
        q_med = q_interp(0.5)
        q_upp = q_interp(self.quantile_range[1] / 100.0)
        
        # prepare output shapes
        out_shape = [1] * nd
        for i, d in enumerate(keep_dims):
            out_shape[d] = keep_shape[i]
        
        # reshape data centre
        if self.with_centering:
            self.centre_ = q_med.reshape(out_shape)
        else:
            self.centre_ = np.zeros(out_shape)
        
        # reshape scale
        if self.with_scaling:
            self.scale_ = (q_upp - q_low).reshape(out_shape)
        else:
            self.scale_ = np.ones_like(self.centre_)
        
        # avoid NaNs
        self.scale_[
            (self.scale_ == 0.0) | (np.isnan(self.scale_))
        ] = 1.0
        
        return self

    def transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        """Transform the data using the robust scaler.
        
        Parameters
        ----------
        X : np.ndarray
            The data.
        args : Any
            Additional arguments.

        Returns
        -------
        Z : np.ndarray
            The transformed data.
        """

        # check fit
        if (self.with_centering & (self.centre_ is None)) | (self.with_scaling & (self.scale_ is None)):
            raise ValueError('The scaler has not been fitted yet.')
        
        # make sure we don't in-place anything
        X = X.copy()
        
        # demean
        if self.with_centering:
            X = X - self.centre_
        
        # scale
        if self.with_scaling:
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
        X : np.ndarray
            The inverse transformed data.
        """
        
        # check fit
        if (self.with_centering & (self.centre_ is None)) | (self.with_scaling & (self.scale_ is None)):
            raise ValueError('The scaler has not been fitted yet.')
        
        # make sure we don't in-place anything
        X = X.copy()
        
        # re-scale
        if self.with_scaling:
            X = X * self.scale_
        
        # re-mean
        if self.with_centering:
            X = X + self.centre_
        
        return X
    
    def fit_transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        """Fit and transform the data in one step.
        
        Parameters
        ----------
        X : np.ndarray
            The data.
        args : Any
            Additional arguments.
        
        Returns
        -------
        Z : np.ndarray
            The transformed data.
        """
        
        return self.fit(X, *args).transform(X, *args)

    def clone(self) -> "_RobustScaler_numpy":
        """Obtain a clone of this class.
        
        Returns
        -------
        scaler : _RobustScaler_numpy
            The clone.
        """
        
        return _RobustScaler_numpy(
            with_centering = self.with_centering, 
            with_scaling = self.with_scaling, 
            quantile_range = self.quantile_range, 
            dims = self.dims_
        )

class _RobustScaler_torch(sklearn.base.BaseEstimator):
    """A robust scaler for torch tensors. Note that this class is not exported and should not be called directly.
    
    Parameters
    ----------
    with_centering : bool, default=True
        If True, center the data before scaling.
    with_scaling : bool, default=True
        If True, scale the data to unit variance.
    quantile_range : tuple[float, float], default=(25.0, 75.0)
        Tuple describing the quantiles.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    
    Attributes
    ----------
    with_centering : bool, default=True
        If True, center the data before scaling.
    with_scaling : bool, default=True
        If True, scale the data to unit variance.
    quantile_range : tuple[float, float], default=(25.0, 75.0)
        Tuple describing the quantiles.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    dims_ : tuple[int], default=None
        Tuple specifying the dimensions to scale over.
    centre_ : torch.Tensor, default=None
        The centre of each feature of shape ``X``.
    scale_ : torch.Tensor, default=None
        The scale of each feature of shape ``X`.
    """
    
    def __init__(self, with_centering: bool = True, with_scaling: bool = True, quantile_range: tuple[float] = (25.0, 75.0), dims: Union[list, tuple, int, None] = None):
        """Obtain a robust scaler.
        
        Parameters
        ----------
        with_centering : bool, default=True
            If True, center the data before scaling.
        with_scaling : bool, default=True
            If True, scale the data to unit variance.
        quantile_range : tuple[float, float], default=(25.0, 75.0)
            Tuple describing the quantiles.
        dims : int, list or tuple of ints, default=None
            The dimensions over which to scale (None for first dimension).
        """

        # store options
        self.dims = dims
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        
        # setup internals
        self.dims_ = dims
        self.centre_ = None
        self.scale_ = None
    
    def fit(self, X: torch.Tensor, *args: Any) -> "_RobustScaler_torch":
        """Fit the scaler.
        
        Parameters
        ----------
        X : torch.Tensor
            The data.
        args : Any
            Additional arguments.
        
        Returns
        -------
        scaler : _RobustScaler_torch
            The fitted scaler.
        """
        
        # check type
        if not isinstance(X, torch.Tensor):
            raise TypeError(f'`X` must be of type torch.Tensor, but got {type(X)}.')
        
        # check dimensions
        if self.dims_ is None:
            self.dims_ = torch.tensor([0])
        
        # set type
        if isinstance(self.dims_, torch.Tensor) == False:
            self.dims_ = torch.tensor(self.dims_)
        self.dims_ = self.dims_.to(torch.int32)
        self.dims_ = tuple(self.dims_.cpu().numpy())
        
        # save number of dims
        nd = X.ndim
        
        # normalise negatives (unique, sorted)
        red_dims = tuple(sorted({d % nd for d in self.dims_}))
        keep_dims = tuple([i for i in range(nd) if i not in red_dims])
        
        # permute so reduced dims come first, then kept dims
        perm = red_dims + keep_dims
        X_p = X.permute(*perm).contiguous()
        
        # flatten reduced dims (leading axis R)
        R = 1 if len(red_dims) == 0 else reduce(operator.mul, (X.size(d) for d in red_dims), 1)
        keep_shape = tuple(X.size(d) for d in keep_dims)
        X_f = X_p.reshape(R, *keep_shape)
        
        # sort once along flattened reduction axis
        X_s, _ = torch.sort(X_f, dim = 0)
        
        # setup helper
        def q_interp(q: float) -> torch.Tensor:
            """Interpolate quantile.
            
            Parameters
            ----------
            q : float
                Quantile in [0, 1].
            
            Returns
            -------
            q : torch.Tensor
                The quantile of shape (n_features,).
            """
            
            # compute pos
            virtual_pos = q * (X_s.shape[0] - 1)
            pos = math.floor(virtual_pos)
            
            # handle uneven case (median easily defined)
            if (X_s.shape[0] % 2) == 1:
                return X_s[pos]
            
            # handle even by selecting both contenders
            next_pos = pos + 1
            
            if next_pos >= X_s.shape[0]:
                pos -= 1
                next_pos -= 1
            
            if pos < 0:
                pos = 0
                next_pos = 0

            # LERP between values
            gamma = virtual_pos - pos
            lv, hv = X_s[pos], X_s[next_pos]
            X_lerp = (1 - gamma) * lv + gamma * hv
            
            return X_lerp
        
        # determine quantiles with helper
        q_low = q_interp(self.quantile_range[0] / 100.0)
        q_med = q_interp(0.5)
        q_upp = q_interp(self.quantile_range[1] / 100.0)
        
        # prepare output shapes
        out_shape = [1] * nd
        for i, d in enumerate(keep_dims):
            out_shape[d] = keep_shape[i]
        
        # reshape data centre
        if self.with_centering:
            self.centre_ = q_med.reshape(out_shape)
        else:
            self.centre_ = torch.zeros(out_shape, dtype = X.dtype, device = X.device)
        
        # reshape scale
        if self.with_scaling:
            self.scale_ = (q_upp - q_low).reshape(out_shape)
        else:
            self.scale_ = torch.ones_like(self.centre_)
        
        # avoid NaNs
        self.scale_[
            (self.scale_ == 0.0) | (torch.isnan(self.scale_))
        ] = 1.0
        
        return self

    def transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
        """Transform the data using the robust scaler.
        
        Parameters
        ----------
        X : torch.Tensor
            The data.
        args : Any
            Additional arguments.

        Returns
        -------
        Z : torch.Tensor
            The transformed data.
        """

        # check fit
        if (self.with_centering & (self.centre_ is None)) | (self.with_scaling & (self.scale_ is None)):
            raise ValueError('The scaler has not been fitted yet.')
        
        # make sure we don't in-place anything
        X = X.clone()
        
        # demean
        if self.with_centering:
            X = X - self.centre_
        
        # scale
        if self.with_scaling:
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
        X : torch.Tensor
            The inverse transformed data.
        """
        
        # check fit
        if (self.with_centering & (self.centre_ is None)) | (self.with_scaling & (self.scale_ is None)):
            raise ValueError('The scaler has not been fitted yet.')
        
        # make sure we don't in-place anything
        X = X.clone()
        
        # re-scale
        if self.with_scaling:
            X = X * self.scale_
        
        # re-mean
        if self.with_centering:
            X = X + self.centre_
        
        return X
    
    def fit_transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
        """Fit and transform the data in one step.
        
        Parameters
        ----------
        X : torch.Tensor
            The data.
        args : Any
            Additional arguments.
        
        Returns
        -------
        Z : torch.Tensor
            The transformed data.
        """
        
        return self.fit(X, *args).transform(X, *args)

    def clone(self) -> "_RobustScaler_torch":
        """Obtain a clone of this class.
        
        Returns
        -------
        scaler : _RobustScaler_torch
            The clone.
        """
        
        return _RobustScaler_torch(
            with_centering = self.with_centering, 
            with_scaling = self.with_scaling, 
            quantile_range = self.quantile_range, 
            dims = self.dims_
        )

class RobustScaler(sklearn.base.BaseEstimator):
    """Implements a robust scaler that is invariant to outliers.
    
    By default, this scaler removes the median before scaling the data
    according to the interquartile range :math:`[0.25, 0.75]`. This is
    useful because, unlike :py:class:`~mvpy.preprocessing.Scaler`, it
    means that :py:class:`~mvpy.preprocessing.RobustScaler` is robust 
    to outliers that might affect a :py:class:`~mvpy.preprocessing.Scaler` 
    poorly.
    
    Both centering and scaling are optional and can be turned on or 
    off using :py:attr:`~mvpy.preprocessing.RobustScaler.with_centering`
    and :py:attr:`~mvpy.preprocessing.RobustScaler.with_scaling`.
    
    Parameters
    ----------
    with_centering : bool, default=True
        If True, center the data before scaling.
    with_scaling : bool, default=True
        If True, scale the data according to the quantiles.
    quantile_range : tuple[float, float], default=(25.0, 75.0)
        Tuple describing the quantiles.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (``None`` for first dimension).
    
    Attributes
    ----------
    with_centering : bool, default=True
        If True, center the data before scaling.
    with_scaling : bool, default=True
        If True, scale the data according to the quantiles.
    quantile_range : tuple[float, float], default=(25.0, 75.0)
        Tuple describing the quantiles.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (``None`` for first dimension).
    dims_ : tuple[int], default=None
        Tuple specifying the dimensions to scale over.
    centre_ : torch.Tensor, default=None
        The centre of each feature of shape ``X``.
    scale_ : torch.Tensor, default=None
        The scale of each feature of shape ``X`.
    
    See also
    --------
    mvpy.preprocessing.Scaler : An alternative scaler that normalises data to zero mean and unit variance.
    mvpy.preprocessing.Clamp : A complementary class that implements clamping data at specific values.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.preprocessing import RobustScaler
    >>> scaler = RobustScaler().to_torch()
    >>> X = torch.normal(5, 10, (1000, 5))
    >>> X[500,0] = 1e3
    >>> X.std(0)
    tensor([32.9122,  9.9615, 10.1481, 10.1058,  9.7468])
    >>> Z = scaler.fit_transform(X)
    >>> Z.std(0)
    tensor([2.7348, 0.7351, 0.7464, 0.7609, 0.7154])
    >>> H = scaler.inverse_transform(Z)
    >>> H.std(0)
    tensor([32.9122,  9.9615, 10.1481, 10.1058,  9.7468])
    """

    def __init__(self, with_centering: bool = True, with_scaling: bool = True, quantile_range: tuple[float, float] = (25.0, 75.0), dims: Union[list, tuple, int, None] = None):
        """Obtain a new RobustScaler.
        
        Parameters
        ----------
        with_centering : bool, default=True
            If True, center the data before scaling.
        with_scaling : bool, default=True
            If True, scale the data according to the quantiles.
        quantile_range : tuple[float, float], default=(25.0, 75.0)
            Tuple describing the quantiles.
        dims : int, list or tuple of ints, default=None
            The dimensions over which to scale (``None`` for first dimension).
        """

        # setup parameters
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.dims = dims
        
        # setup internal parameters
        self.dims_ = None
        self.centre_ = None
        self.scale_ = None
    
    def _get_estimator(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> sklearn.base.BaseEstimator:
        """Given the data, determine which robust scaler to use.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The data of arbitrary shape.
        args : Any
            Additional arguments.
        
        Returns
        -------
        scaler : sklearn.base.BaseEstimator
            The scaler.
        """
        
        if isinstance(X, torch.Tensor):
            return _RobustScaler_torch
        elif isinstance(X, np.ndarray):
            return _RobustScaler_numpy
        
        raise TypeError(f'`X` must be either torch.Tensor or np.ndarray, but got {type(X)}.')
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> "RobustScaler":
        """Fit the scaler.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The data of arbitrary shape.
        args : Any
            Additional arguments.
        
        Returns
        -------
        scaler : sklearn.base.BaseEstimator
            The fitted scaler.
        """
        
        return self._get_estimator(X, *args)(
            with_centering = self.with_centering, 
            with_scaling = self.with_scaling, 
            quantile_range = self.quantile_range,
            dims = self.dims
        ).fit(X, *args)
    
    def transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Transform the data using scaler.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The data of shape ``X``.
        args : Any
            Additional arguments.

        Returns
        -------
        Z : np.ndarray | torch.Tensor
            The transformed data of shape ``X``.
        """

        return self._get_estimator(X, *args)(
            with_centering = self.with_centering, 
            with_scaling = self.with_scaling, 
            quantile_range = self.quantile_range,
            dims = self.dims
        ).transform(X, *args)
    
    def inverse_transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Invert the transform of the data.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The data of shape ``X``.
        args : Any
            Additional arguments.
        
        Returns
        -------
        X : np.ndarray | torch.Tensor
            The inverse transformed data of shape ``X``.
        """
        
        return self._get_estimator(X, *args)(
            with_centering = self.with_centering, 
            with_scaling = self.with_scaling, 
            quantile_range = self.quantile_range,
            dims = self.dims
        ).inverse_transform(X, *args)
    
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform the data in one step.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The data of shape ``X``.
        args : Any
            Additional arguments.
        
        Returns
        -------
        Z : np.ndarray | torch.Tensor
            The transformed data of shape ``X``.
        """
        
        return self._get_estimator(X, *args)(
            with_centering = self.with_centering, 
            with_scaling = self.with_scaling, 
            quantile_range = self.quantile_range,
            dims = self.dims
        ).fit_transform(X, *args)
    
    def to_torch(self) -> "_RobustScaler_torch":
        """Select the torch backend. Note that this cannot be called for conversion.
        
        Returns
        -------
        scaler : _RobustScaler_torch
            The robust scaler using the torch backend.
        """
        
        return self._get_estimator(torch.tensor([1.0]))(
            with_centering = self.with_centering, 
            with_scaling = self.with_scaling, 
            quantile_range = self.quantile_range,
            dims = self.dims
        )
    
    def to_numpy(self) -> "_RobustScaler_numpy":
        """Select the numpy backend. Note that this cannot be called for conversion.

        Returns
        -------
        scaler : _RobustScaler_numpy
            The robust scaler using the numpy backend.
        """

        return self._get_estimator(np.array([1.0]))(
            with_centering = self.with_centering, 
            with_scaling = self.with_scaling, 
            quantile_range = self.quantile_range,
            dims = self.dims
        )
    
    def clone(self) -> "RobustScaler":
        """Obtain a clone of this class.
        
        Returns
        -------
        scaler : RobustScaler
            The cloned robust scaler.
        """
        
        return RobustScaler(
            with_centering = self.with_centering, 
            with_scaling = self.with_scaling, 
            quantile_range = self.quantile_range,
            dims = self.dims
        )
    
    def copy(self) -> "RobustScaler":
        """Obtain a copy of this class.

        Returns
        -------
        scaler : RobustScaler
            The copied robust scaler.
        """
        
        return self.clone()