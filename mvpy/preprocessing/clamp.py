'''
A collection of estimators for clamping data.
'''

import numpy as np
import torch
import sklearn

import operator
from functools import reduce

from typing import Union, Any, Tuple, Sequence, Optional

class _Clamp_numpy(sklearn.base.BaseEstimator):
    """A clamp for numpy backend to clamp extreme values.
    
    Parameters
    ----------
    lower : Optional[float], default=None
        Lower bound for clamping. If ``None``, no lower bound is applied.
    upper : Optional[float], default=None
        Upper bound for clamping, If ``None``, no upper bound is applied.
    method : {'iqr', 'quantile', 'mad'}, default='iqr'
        If both :py:attr:`~mvpy.preprocessing.Clamp.lower` and :py:attr:`~mvpy.preprocessing.Clamp.upper` are ``None``, what method to use for fitting bounds?
    k : Optional[float], default=None
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``iqr``, scale the :math:`[0.25, 0.75]` quantiles by :math:`k` (with ``default=1.5``). 
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``quantile``, clamp tails outside :math:`[k, 1 - k]` (with ``default = 0.05``).
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``mad``, scale the median absolute deviation by :math:`k` (with ``default=3.0``).
        Otherwise unused.
    eps : float, default=1e-9
        When checking span correctness, epsilon to apply as jitter.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    
    Attributes
    ----------
    lower : Optional[float], default=None
        Lower bound for clamping. If ``None``, no lower bound is applied.
    upper : Optional[float], default=None
        Upper bound for clamping, If ``None``, no upper bound is applied.
    method : {'iqr', 'quantile', 'mad'}, default='iqr'
        If both :py:attr:`~mvpy.preprocessing.Clamp.lower` and :py:attr:`~mvpy.preprocessing.Clamp.upper` are ``None``, what method to use for fitting bounds?
    k : Optional[float], default=None
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``iqr``, scale the :math:`[0.25, 0.75]` quantiles by :math:`k` (with ``default=1.5``). 
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``quantile``, clamp tails outside :math:`[k, 1 - k]` (with ``default = 0.05``).
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``mad``, scale the median absolute deviation by :math:`k` (with ``default=3.0``).
        Otherwise unused.
    eps : float, default=1e-9
        When checking span correctness, epsilon to apply as jitter.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    lower_ : float | np.ndarray, default=None
        Lower bound for clamping, either prespecified or fitted.
    upper_ : float | np.ndarray, default=None
        Upper bound for clamping, either prespecified or fitted.
    dims_ : tuple[int], default=None
        Tuple specifying the dimensions to scale over.
    """
    
    def __init__(self, lower: Optional[float] = None, upper: Optional[float] = None, method: str = 'iqr', k: Optional[float] = None, eps: float = 1e-9, dims: Union[list, tuple, int, None] = None):
        """Obtain a clamp.
        
        Parameters
        ----------
        lower : Optional[float], default=None
            Lower bound for clamping. If ``None``, no lower bound is applied.
        upper : Optional[float], default=None
            Upper bound for clamping, If ``None``, no upper bound is applied.
        method : {'iqr', 'quantile', 'mad'}, default='iqr'
            If both :py:attr:`~mvpy.preprocessing.Clamp.lower` and :py:attr:`~mvpy.preprocessing.Clamp.upper` are ``None``, what method to use for fitting bounds?
        k : Optional[float], default=None
            For :py:attr:`~mvpy.preprocessing.Clamp.method` ``iqr``, scale the :math:`[0.25, 0.75]` quantiles by :math:`k` (with ``default=1.5``). 
            For :py:attr:`~mvpy.preprocessing.Clamp.method` ``quantile``, clamp tails outside :math:`[k, 1 - k]` (with ``default = 0.05``).
            For :py:attr:`~mvpy.preprocessing.Clamp.method` ``mad``, scale the median absolute deviation by :math:`k` (with ``default=3.0``).
            Otherwise unused.
        eps : float, default=1e-9
            When checking span correctness, epsilon to apply as jitter.
        dims : int, list or tuple of ints, default=None
            The dimensions over which to scale (None for first dimension).
        """

        # store options
        self.lower = lower
        self.upper = upper
        self.method = method
        self.k = k
        self.dims = dims
        self.eps = eps
        
        # check setup of k
        if (self.method == 'iqr') and (k is None):
            # for iqr, set default to 1.5
            self.k = 1.5
        elif (self.method == 'quantile') and (k is None):
            # for quantile, set default to 0.05
            self.k = 0.05
        elif (self.method == 'mad') and (k is None):
            # for mad, set default to 3.0
            self.k = 3.0
        
        # setup internals
        self.lower_ = lower
        self.upper_ = upper
        self.dims_ = None
    
    @staticmethod
    def sorted_unique_dims_(ndim: int, dims: Sequence[int]) -> Tuple[int, ...]:
        """Return sorted unique dims.
        
        Parameters
        ----------
        ndim : int
            Number of dimensions.
        dims : Sequence[int]
            Sequence of dims.
        
        Returns
        -------
        dims : Tuple[int]
            Sorted unique dims.
        """
        
        # handle negatives (sort, unique)
        d = sorted({(ax if ax >= 0 else ndim + ax) for ax in dims})
        
        # check outputs
        for ax in d:
            if ax < 0 or ax >= ndim:
                raise IndexError('`dims` contains invalid axis.')
        
        return tuple(d)

    @staticmethod
    def reduce_over_dims_quantiles_(X: np.ndarray, reduce_dims: Tuple[int, ...], qs: np.ndarray) -> np.ndarray:
        """Compute quantiles over multiple dims.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape ``X`.
        reduce_dims : Tuple[int]
            Tuple specifying which dimensions to reduce.
        qs : np.ndarray
            Quantiles to compute.
        
        Returns
        -------
        Q : np.ndarray
            Output data of shape ``(n_quantiles, [...])``
        """
        
        # check dims
        if len(reduce_dims) == 0:
            return X[None,...] * np.ones((qs.shape[0], *X.shape))
        
        # check dims
        nd = X.ndim
        
        # check reduction
        reduce_dims = tuple(reduce_dims)
        keep_dims = tuple(i for i in range(nd) if i not in reduce_dims)
        
        # reshape X
        perm = (*reduce_dims, *keep_dims)
        X_p = np.transpose(X, axes = perm)
        
        # flatten reduce dims
        reduce_n = reduce(operator.mul, (X.shape[d] for d in reduce_dims), 1)
        rest_shape = X_p.shape[len(reduce_dims):]
        X_r = X_p.reshape(reduce_n, *rest_shape)
        
        # compute quantiles
        Q = np.quantile(X_r, qs, axis = 0, keepdims = True)
        
        # check shape
        if Q.shape[0] != qs.size:
            Q = Q[None,...]
        
        # insert singleton dims
        target_shape = [1] * nd
        for i, ax in enumerate(keep_dims):
            target_shape[ax] = rest_shape[i]
        Q_s = Q.reshape(qs.size, *target_shape)
        
        return Q_s
        
    def fit(self, X: np.ndarray, *args: Any) -> "_Clamp_numpy":
        """Fit the clamp.
        
        Parameters
        ----------
        X : np.ndarray
            The data.
        args : Any
            Additional arguments.
        
        Returns
        -------
        clamp : _Clamp_numpy
            The fitted clamp.
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
        
        # make dims sorted and unique
        self.dims_ = self.sorted_unique_dims_(X.ndim, self.dims_)
        
        # check if we have a prespecified range
        if (self.lower is None) and (self.upper is None):
            # if not, check fit method
            if self.method == 'iqr':
                # setup quantiles
                qs = np.array([0.25, 0.75])
                Q = self.reduce_over_dims_quantiles_(X, self.dims_, qs)
                
                # compute IQR
                Q1, Q3 = Q[0], Q[1]
                IQR = Q3 - Q1
                
                # set bounds
                self.lower_ = Q1 - self.k * IQR
                self.upper_ = Q3 + self.k * IQR
            elif self.method == 'quantile':
                # setup quantiles
                qs = np.array([self.k, 1.0 - self.k])
                Q = self.reduce_over_dims_quantiles_(X, self.dims_, qs)
                
                # setup bounds
                self.lower_ = Q[0]
                self.upper_ = Q[1]
            elif self.method == 'mad':
                # compute median
                qs = np.array([0.5])
                median = self.reduce_over_dims_quantiles_(X, self.dims_, qs)[0]
                
                # compute median absolute deviation
                mad = self.reduce_over_dims_quantiles_(np.abs(X - median), self.dims_, qs)[0]
                self.lower_ = median - self.k * mad
                self.upper_ = median + self.k * mad
            else:
                raise ValueError(f'Unknown method `{self.method}`. Expected one of [\'iqr\', \'quantile\', \'mad\'].')

            # ensure non-degenerate span
            span = np.maximum(
                self.upper_ - self.lower_,
                self.eps
            )
            
            centre = (self.upper_ + self.lower_) * 0.5
            self.lower_ = centre - 0.5 * span
            self.upper_ = centre + 0.5 * span
        else:
            # otherwise, simply use prespecified range
            self.lower_ = self.lower
            self.upper_ = self.upper
        
        return self
        
    def transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        """Transform the data using the clamp.
        
        Parameters
        ----------
        X : np.ndarray
            The data.
        args : Any
            Additional arguments.

        Returns
        -------
        Z : np.ndarray
            The clamped data.
        """
        
        # check if we had prespecified values
        if (self.lower is None) and (self.upper is None):
            # if not, check fit
            if (self.lower_ is None) or (self.upper_ is None):
                raise ValueError(f'Clamp has not been fitted yet.')
            
            # save bounds
            lower, upper = self.lower_, self.upper_
        else:
            # if so, save bounds
            lower, upper = self.lower_, self.upper_
        
        # apply bounds
        return np.clip(X, a_min = self.lower_, a_max = self.upper_)
    
    def inverse_transform(self, X: np.ndarray, *args: Any) -> np.ndarray:
        """Invert the transform of the data.
        
        Parameters
        ----------
        X : np.ndarray
            The input data of shape ``X``.
        args : Any
            Additional arguments.
        
        Returns
        -------
        X : np.ndarray
            The input data of shape ``X``.
        """
        
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
            The clamped data.
        """
        
        return self.fit(X, *args).transform(X, *args)

    def clone(self) -> "_Clamp_numpy":
        """Obtain a clone of this class.
        
        Returns
        -------
        clamp : _Clamp_numpy
            The clone.
        """
        
        return _Clamp_numpy(
            lower = self.lower,
            upper = self.upper,
            method = self.method,
            k = self.k,
            eps = self.eps,
            dims = self.dims_
        )

class _Clamp_torch(sklearn.base.BaseEstimator):
    """A clamp for torch backend to clamp extreme values.
    
    Parameters
    ----------
    lower : Optional[float], default=None
        Lower bound for clamping. If ``None``, no lower bound is applied.
    upper : Optional[float], default=None
        Upper bound for clamping, If ``None``, no upper bound is applied.
    method : {'iqr', 'quantile', 'mad'}, default='iqr'
        If both :py:attr:`~mvpy.preprocessing.Clamp.lower` and :py:attr:`~mvpy.preprocessing.Clamp.upper` are ``None``, what method to use for fitting bounds?
    k : Optional[float], default=None
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``iqr``, scale the :math:`[0.25, 0.75]` quantiles by :math:`k` (with ``default=1.5``). 
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``quantile``, clamp tails outside :math:`[k, 1 - k]` (with ``default = 0.05``).
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``mad``, scale the median absolute deviation by :math:`k` (with ``default=3.0``).
        Otherwise unused.
    eps : float, default=1e-9
        When checking span correctness, epsilon to apply as jitter.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    
    Attributes
    ----------
    lower : Optional[float], default=None
        Lower bound for clamping. If ``None``, no lower bound is applied.
    upper : Optional[float], default=None
        Upper bound for clamping, If ``None``, no upper bound is applied.
    method : {'iqr', 'quantile', 'mad'}, default='iqr'
        If both :py:attr:`~mvpy.preprocessing.Clamp.lower` and :py:attr:`~mvpy.preprocessing.Clamp.upper` are ``None``, what method to use for fitting bounds?
    k : Optional[float], default=None
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``iqr``, scale the :math:`[0.25, 0.75]` quantiles by :math:`k` (with ``default=1.5``). 
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``quantile``, clamp tails outside :math:`[k, 1 - k]` (with ``default = 0.05``).
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``mad``, scale the median absolute deviation by :math:`k` (with ``default=3.0``).
        Otherwise unused.
    eps : float, default=1e-9
        When checking span correctness, epsilon to apply as jitter.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    lower_ : float | torch.Tensor, default=None
        Lower bound for clamping, either prespecified or fitted.
    upper_ : float | torch.Tensor, default=None
        Upper bound for clamping, either prespecified or fitted.
    dims_ : tuple[int], default=None
        Tuple specifying the dimensions to scale over.
    """
    
    def __init__(self, lower: Optional[float] = None, upper: Optional[float] = None, method: str = 'iqr', k: Optional[float] = None, eps: float = 1e-9, dims: Union[list, tuple, int, None] = None):
        """Obtain a clamp.
        
        Parameters
        ----------
        lower : Optional[float], default=None
            Lower bound for clamping. If ``None``, no lower bound is applied.
        upper : Optional[float], default=None
            Upper bound for clamping, If ``None``, no upper bound is applied.
        method : {'iqr', 'quantile', 'mad'}, default='iqr'
            If both :py:attr:`~mvpy.preprocessing.Clamp.lower` and :py:attr:`~mvpy.preprocessing.Clamp.upper` are ``None``, what method to use for fitting bounds?
        k : Optional[float], default=None
            For :py:attr:`~mvpy.preprocessing.Clamp.method` ``iqr``, scale the :math:`[0.25, 0.75]` quantiles by :math:`k` (with ``default=1.5``). :py:attr:`~mvpy.preprocessing.Clamp.method` ``mad``, scale the median absolute deviation by :math:`k` (with ``default=3.0``). Otherwise unused.
        eps : float, default=1e-9
            When checking span correctness, epsilon to apply as jitter.
        dims : int, list or tuple of ints, default=None
            The dimensions over which to scale (None for first dimension).
        """

        # store options
        self.lower = lower
        self.upper = upper
        self.method = method
        self.k = k
        self.dims = dims
        self.eps = eps
        
        # check setup of k
        if (self.method == 'iqr') and (k is None):
            # for iqr, set default to 1.5
            self.k = 1.5
        elif (self.method == 'quantile') and (k is None):
            # for quantile, set default to 0.05
            self.k = 0.05
        elif (self.method == 'mad') and (k is None):
            # for mad, set default to 3.0
            self.k = 3.0
        
        # setup internals
        self.lower_ = lower
        self.upper_ = upper
        self.dims_ = None
    
    @staticmethod
    def sorted_unique_dims_(ndim: int, dims: Sequence[int]) -> Tuple[int, ...]:
        """Return sorted unique dims.
        
        Parameters
        ----------
        ndim : int
            Number of dimensions.
        dims : Sequence[int]
            Sequence of dims.
        
        Returns
        -------
        dims : Tuple[int]
            Sorted unique dims.
        """
        
        # handle negatives (sort, unique)
        d = sorted({(ax if ax >= 0 else ndim + ax) for ax in dims})
        
        # check outputs
        for ax in d:
            if ax < 0 or ax >= ndim:
                raise IndexError('`dims` contains invalid axis.')
        
        return tuple(d)

    @staticmethod
    def reduce_over_dims_quantiles_(X: torch.Tensor, reduce_dims: Tuple[int, ...], qs: torch.Tensor) -> torch.Tensor:
        """Compute quantiles over multiple dims.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape ``X`.
        reduce_dims : Tuple[int]
            Tuple specifying which dimensions to reduce.
        qs : torch.Tensor
            Quantiles to compute.
        
        Returns
        -------
        Q : torch.Tensor
            Output data of shape ``(n_quantiles, [...])``
        """
        
        # check dims
        if len(reduce_dims) == 0:
            return X.unsqueeze(0).expand(qs.shape[0], *X.shape)
        
        # check dims
        nd = X.ndim
        
        # check reduction
        reduce_dims = tuple(reduce_dims)
        keep_dims = tuple(i for i in range(nd) if i not in reduce_dims)
        
        # reshape X
        perm = (*reduce_dims, *keep_dims)
        X_p = X.permute(perm)
        
        # flatten reduce dims
        reduce_n = reduce(operator.mul, (X.size(d) for d in reduce_dims), 1)
        rest_shape = X_p.shape[len(reduce_dims):]
        X_r = X_p.reshape(reduce_n, *rest_shape)
        
        # compute quantiles
        Q = torch.quantile(X_r, qs, dim = 0, keepdim = True)
        
        # check shape
        if Q.shape[0] != qs.numel():
            Q = Q.unsqueeze(0)
        
        # insert singleton dims
        target_shape = [1] * nd
        for i, ax in enumerate(keep_dims):
            target_shape[ax] = rest_shape[i]
        Q_s = Q.reshape(qs.numel(), *target_shape)
        
        return Q_s
        
    def fit(self, X: torch.Tensor, *args: Any) -> "_Clamp_torch":
        """Fit the clamp.
        
        Parameters
        ----------
        X : torch.Tensor
            The data.
        args : Any
            Additional arguments.
        
        Returns
        -------
        clamp : _Clamp_torch
            The fitted clamp.
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
        
        # make dims sorted and unique
        self.dims_ = self.sorted_unique_dims_(X.ndim, self.dims_)
        
        # check if we have a prespecified range
        if (self.lower is None) and (self.upper is None):
            # if not, check fit method
            if self.method == 'iqr':
                # setup quantiles
                qs = torch.tensor([0.25, 0.75], dtype = X.dtype, device = X.device)
                Q = self.reduce_over_dims_quantiles_(X, self.dims_, qs)
                
                # compute IQR
                Q1, Q3 = Q[0], Q[1]
                IQR = Q3 - Q1
                
                # set bounds
                self.lower_ = Q1 - self.k * IQR
                self.upper_ = Q3 + self.k * IQR
            elif self.method == 'quantile':
                # setup quantiles
                qs = torch.tensor([self.k, 1.0 - self.k], dtype = X.dtype, device = X.device)
                Q = self.reduce_over_dims_quantiles_(X, self.dims_, qs)
                
                # setup bounds
                self.lower_ = Q[0]
                self.upper_ = Q[1]
            elif self.method == 'mad':
                # compute median
                qs = torch.tensor([0.5], dtype = X.dtype, device = X.device)
                median = self.reduce_over_dims_quantiles_(X, self.dims_, qs)[0]
                
                # compute median absolute deviation
                mad = self.reduce_over_dims_quantiles_((X - median).abs(), self.dims_, qs)[0]
                self.lower_ = median - self.k * mad
                self.upper_ = median + self.k * mad
            else:
                raise ValueError(f'Unknown method `{self.method}`. Expected one of [\'iqr\', \'quantile\', \'mad\'].')

            # ensure non-degenerate span
            span = torch.maximum(
                self.upper_ - self.lower_,
                torch.tensor(self.eps, dtype = X.dtype, device = X.device)
            )
            
            centre = (self.upper_ + self.lower_) * 0.5
            self.lower_ = centre - 0.5 * span
            self.upper_ = centre + 0.5 * span
        else:
            # otherwise, simply use prespecified range
            self.lower_ = self.lower
            self.upper_ = self.upper
        
        return self
        
    def transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
        """Transform the data using the clamp.
        
        Parameters
        ----------
        X : torch.Tensor
            The data.
        args : Any
            Additional arguments.

        Returns
        -------
        Z : torch.Tensor
            The clamped data.
        """
        
        # check if we had prespecified values
        if (self.lower is None) and (self.upper is None):
            # if not, check fit
            if (self.lower_ is None) or (self.upper_ is None):
                raise ValueError(f'Clamp has not been fitted yet.')
            
            # save bounds
            lower, upper = self.lower_, self.upper_
        else:
            # if so, save bounds
            lower, upper = self.lower_, self.upper_
        
        # apply bounds
        return torch.clamp(X, min = self.lower_, max = self.upper_)
    
    def inverse_transform(self, X: torch.Tensor, *args: Any) -> torch.Tensor:
        """Invert the transform of the data.
        
        Parameters
        ----------
        X : torch.Tensor
            The input data of shape ``X``.
        args : Any
            Additional arguments.
        
        Returns
        -------
        X : torch.Tensor
            The input data of shape ``X``.
        """
        
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
            The clamped data.
        """
        
        return self.fit(X, *args).transform(X, *args)

    def clone(self) -> "_Clamp_torch":
        """Obtain a clone of this class.
        
        Returns
        -------
        clamp : _Clamp_torch
            The clone.
        """
        
        return _Clamp_torch(
            lower = self.lower,
            upper = self.upper,
            method = self.method,
            k = self.k,
            eps = self.eps,
            dims = self.dims_
        )

class Clamp(sklearn.base.BaseEstimator):
    """Implements a clamp to handle extreme values.
    
    Generally, this will clamp data :math:`X` to lower and upper bounds
    defined by :py:attr:`~mvpy.preprocessing.Clamp.lower` and 
    :py:attr:`~mvpy.preprocessing.Clamp.upper` whenever they are exceeded.
    
    This can be useful for dealing with outliers: For example, in M-/EGG 
    data that was minimally preprocessed, this may be used to curb EOG 
    artifacts easily without removing time points or trials.
    
    By default, both :py:attr:`~mvpy.preprocessing.Clamp.lower` and 
    :py:attr:`~mvpy.preprocessing.Clamp.upper` will be ``None``. This
    constitutes a special case where the bounds will then be fit directly
    to the data. There are three different ways of fitting bounds, controlled
    by :py:attr:`~mvpy.preprocesing.Clamp.method`:
    
    1. ``iqr``: This will compute the inter-quartile range :math:`[0.25, 0.75]` and clamp data where :math:`X\\notin [\\textrm{median}(X) - k L, \\textrm{median}(X) + k U]`.
    2. ``quantile``: This will clamp data outside of the quantiles given by :math:`[k, 1 - k]`.
    3. ``mad``: This will clamp data at :math:`\\textrm{median}(X)\pm k \\textrm{MAD}` where MAD are median absolute deviations.
    
    If only one of the two bounds is ``None`` instead, the unspecified 
    bound will be interpreted as meaning no clamping in this direction
    to be desired.
    
    Parameters
    ----------
    lower : Optional[float], default=None
        Lower bound for clamping. If ``None``, no lower bound is applied.
    upper : Optional[float], default=None
        Upper bound for clamping, If ``None``, no upper bound is applied.
    method : {'iqr', 'quantile', 'mad'}, default='iqr'
        If both :py:attr:`~mvpy.preprocessing.Clamp.lower` and :py:attr:`~mvpy.preprocessing.Clamp.upper` are ``None``, what method to use for fitting bounds?
    k : Optional[float], default=None
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``iqr``, scale the :math:`[0.25, 0.75]` quantiles by :math:`k` (with ``default=1.5``). 
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``quantile``, clamp tails outside :math:`[k, 1 - k]` (with ``default = 0.05``).
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``mad``, scale the median absolute deviation by :math:`k` (with ``default=3.0``).
        Otherwise unused.
    eps : float, default=1e-9
        When checking span correctness, epsilon to apply as jitter.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    
    Attributes
    ----------
    lower : Optional[float], default=None
        Lower bound for clamping. If ``None``, no lower bound is applied.
    upper : Optional[float], default=None
        Upper bound for clamping, If ``None``, no upper bound is applied.
    method : {'iqr', 'quantile', 'mad'}, default='iqr'
        If both :py:attr:`~mvpy.preprocessing.Clamp.lower` and :py:attr:`~mvpy.preprocessing.Clamp.upper` are ``None``, what method to use for fitting bounds?
    k : Optional[float], default=None
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``iqr``, scale the :math:`[0.25, 0.75]` quantiles by :math:`k` (with ``default=1.5``). 
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``quantile``, clamp tails outside :math:`[k, 1 - k]` (with ``default = 0.05``).
        For :py:attr:`~mvpy.preprocessing.Clamp.method` ``mad``, scale the median absolute deviation by :math:`k` (with ``default=3.0``).
        Otherwise unused.
    eps : float, default=1e-9
        When checking span correctness, epsilon to apply as jitter.
    dims : int, list or tuple of ints, default=None
        The dimensions over which to scale (None for first dimension).
    lower_ : float | np.ndarray | torch.Tensor, default=None
        Lower bound for clamping, either prespecified or fitted.
    upper_ : float | np.ndarray | torch.Tensor, default=None
        Upper bound for clamping, either prespecified or fitted.
    dims_ : tuple[int], default=None
        Tuple specifying the dimensions to scale over.
    
    See also
    --------
    mvpy.preprocessing.Scaler, mvpy.preprocessing.RobustScaler : Complementary scalers.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.preprocessing import Clamp
    >>> X = torch.normal(0, 1, (1000, 5))
    >>> X[500,0] = 1e3
    >>> X.max(0).values
    tensor([10.0000,  3.9375,  3.2070,  3.0591,  3.0165])
    >>> Z = Clamp().fit_transform(X)
    >>> Z.max(0).values
    tensor([2.6926, 2.7263, 2.6343, 2.6616, 2.5378])
    >>> Z = Clamp(upper = 5.0).fit_transform(X)
    >>> Z.max(0).values
    tensor([5.0000, 3.9375, 3.2070, 3.0591, 3.0165])
    """

    def __init__(self, lower: Optional[float] = None, upper: Optional[float] = None, method: str = 'iqr', k: Optional[float] = None, eps: float = 1e-9, dims: Union[list, tuple, int, None] = None):
        """Obtain a clamp.
        
        Parameters
        ----------
        lower : Optional[float], default=None
            Lower bound for clamping. If ``None``, no lower bound is applied.
        upper : Optional[float], default=None
            Upper bound for clamping, If ``None``, no upper bound is applied.
        method : {'iqr', 'quantile', 'mad'}, default='iqr'
            If both :py:attr:`~mvpy.preprocessing.Clamp.lower` and :py:attr:`~mvpy.preprocessing.Clamp.upper` are ``None``, what method to use for fitting bounds?
        k : Optional[float], default=None
            For :py:attr:`~mvpy.preprocessing.Clamp.method` ``iqr``, scale the :math:`[0.25, 0.75]` quantiles by :math:`k` (with ``default=1.5``). :py:attr:`~mvpy.preprocessing.Clamp.method` ``mad``, scale the median absolute deviation by :math:`k` (with ``default=3.0``). Otherwise unused.
        eps : float, default=1e-9
            When checking span correctness, epsilon to apply as jitter.
        dims : int, list or tuple of ints, default=None
            The dimensions over which to scale (None for first dimension).
        """

        # store options
        self.lower = lower
        self.upper = upper
        self.method = method
        self.k = k
        self.dims = dims
        self.eps = eps
    
    def _get_estimator(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> sklearn.base.BaseEstimator:
        """Given the data, determine which clamp backend to use.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The data of arbitrary shape.
        args : Any
            Additional arguments.
        
        Returns
        -------
        clamp : sklearn.base.BaseEstimator
            The clamp.
        """
        
        if isinstance(X, torch.Tensor):
            return _Clamp_torch
        elif isinstance(X, np.ndarray):
            return _Clamp_numpy
        
        raise TypeError(f'`X` must be either torch.Tensor or np.ndarray, but got {type(X)}.')
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> "Clamp":
        """Fit the clamp.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The data of arbitrary shape.
        args : Any
            Additional arguments.
        
        Returns
        -------
        clamp : sklearn.base.BaseEstimator
            The fitted clamp.
        """
        
        return self._get_estimator(X, *args)(
            lower = self.lower,
            upper = self.upper,
            method = self.method,
            k = self.k,
            eps = self.eps,
            dims = self.dims
        ).fit(X, *args)
    
    def transform(self, X: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Transform the data using the clamp.
        
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
            lower = self.lower,
            upper = self.upper,
            method = self.method,
            k = self.k,
            eps = self.eps,
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
        
        .. warning::
            Clamping cannot be inverse transformed. Consequently,
            this returns the clamped values in :math:`X` as is.
        """
        
        return self._get_estimator(X, *args)(
            lower = self.lower,
            upper = self.upper,
            method = self.method,
            k = self.k,
            eps = self.eps,
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
            lower = self.lower,
            upper = self.upper,
            method = self.method,
            k = self.k,
            eps = self.eps,
            dims = self.dims
        ).fit_transform(X, *args)
    
    def to_torch(self) -> "_Clamp_torch":
        """Select the torch backend. Note that this cannot be called for conversion.
        
        Returns
        -------
        clamp : _Clamp_torch
            The clamp using the torch backend.
        """
        
        return self._get_estimator(torch.tensor([1.0]))(
            lower = self.lower,
            upper = self.upper,
            method = self.method,
            k = self.k,
            eps = self.eps,
            dims = self.dims
        )
    
    def to_numpy(self) -> "_Clamp_numpy":
        """Select the numpy backend. Note that this cannot be called for conversion.

        Returns
        -------
        clamp : _Clamp_numpy
            The clamp using the numpy backend.
        """

        return self._get_estimator(np.array([1.0]))(
            lower = self.lower,
            upper = self.upper,
            method = self.method,
            k = self.k,
            eps = self.eps,
            dims = self.dims
        )
    
    def clone(self) -> "Clamp":
        """Obtain a clone of this class.
        
        Returns
        -------
        clamp : Clamp
            The cloned clamp.
        """
        
        return Clamp(
            lower = self.lower,
            upper = self.upper,
            method = self.method,
            k = self.k,
            eps = self.eps,
            dims = self.dims
        )
    
    def copy(self) -> "Clamp":
        """Obtain a copy of this class.

        Returns
        -------
        clamp : Clamp
            The copied clamp.
        """
        
        return self.clone()