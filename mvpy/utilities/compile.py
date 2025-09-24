'''
Exposes decorators for compilation.
'''

import os, warnings

from .env import is_enabled
from .version import compare

from typing import Callable, List, Dict, Any

def _should_enable_in(backend: str) -> bool:
    """Internal function to check environment variables for compilation flags.
    
    Parameters
    ----------
    backend : str
        Which backend to check (torch or numpy).
    
    Returns
    -------
    enabled : bool
        True if compilation is enabled for backend.
    """
    
    # check torch
    if backend == 'torch':
        return is_enabled('MVPY_COMPILE_TORCH', default = True)
    
    # check numpy
    if backend == 'numpy':
        return is_enabled('MVPY_COMPILE_NUMPY', default = True)
    
    raise ValueError(f'Unknown backend={backend}, expected torch or numpy.')

def torch(*args: List[Any], **kwargs: Dict[str, Any]) -> Callable:
    """Decorator that compiles a function with torch.compile, if available.
    
    Parameters
    ----------
    *args : List[Any]
        Arguments to pass.
    **kwargs : Dict[str, Any]
        Keyword arguments to pass.
    
    Returns
    -------
    compiled : Callable
        Compiled function.
    
    Examples
    --------
    Let's look at a very silly toy example:
    >>> import torch
    >>> from mvpy.utilities import compile
    >>> @compile.torch
    >>> def maths_a(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    >>>     return x @ y.t()
    >>> @compile.torch(disable = True)
    >>> def maths_b(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    >>>     return x @ y.t()
    """

    def _decorate(fn: Callable) -> Callable:
        """Internal decorator.
        
        Parameters
        ----------
        fn : Callable
            Function to decorate.
        
        Returns
        -------
        df : Callable
            Decorated function.
        """
        
        # check env
        if not _should_enable_in('torch'):
            return fn
        
        # check availability
        try:
            import torch
            
            compile_fn = getattr(torch, 'compile', None)
            if compile_fn is None:
                return fn
            
            return compile_fn(fn, *args, **kwargs)
        except Exception as e:
            return fn
    
    # allow @decorator and @decorator(...)
    if len(args) == 1 and callable(args[0]) and not kwargs:
        fn = args[0]
        args = ()
        return _decorate(fn)

    return _decorate

def numpy(*args: List[Any], nopython: bool = True, nogil: bool = True, fastmath: bool = True, cache: bool = True, disable: bool = False, **kwargs: Dict[str, Any]) -> Callable:
    """Decorator that compiles a function with numba.jit, if available.
    
    Parameters
    ----------
    *args : List[Any]
        Arguments to pass.
    nopython : bool, default=True
        Default to pass to jit.
    nogil : bool, default=True
        Default to pass to jit.
    fastmath : bool, default=True
        Default to pass to jit.
    cache : bool, default=True
        Default to pass to jit.
    disable : bool, default=False
        Flag to allow disabling compilation (for debugging).
    **kwargs : Dict[str, Any]
        Keyword arguments to pass.
    
    Returns
    -------
    compiled : Callable
        Compiled function.
    
    Examples
    --------
    Let's look at a very silly toy example:
    >>> import numpy as np
    >>> from mvpy.utilities import compile
    >>> @compile.numpy
    >>> def maths_a(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    >>>     return x @ y.t()
    >>> @compile.numpy(disable = True)
    >>> def maths_b(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    >>>     return x @ y.t()
    """
    
    # make kwargs
    kwargs_all = dict(nopython = nopython, nogil = nogil, fastmath = fastmath, cache = cache)
    for key in kwargs:
        kwargs_all[key] = kwargs[key]
    
    def _decorate(fn: Callable) -> Callable:
        """Internal decorator.
        
        Parameters
        ----------
        fn : Callable
            Function to decorate.
        
        Returns
        -------
        df : Callable
            Decorated function.
        """
        
        # check env
        if not _should_enable_in('numpy'):
            return fn

        # check disable flag
        if disable:
            return fn
        
        # check availability
        try:
            import numba
            
            if compare(numba.__version__, '<', '0.56.4'):
                raise NotImplementedError()
            
            jit_fn = getattr(numba, 'jit', None)
            if jit_fn is None:
                return fn
            
            return jit_fn(*args, **kwargs)(fn)
        except Exception as e:
            return fn
    
    # allow @decorator and @decorator(...)
    if len(args) == 1 and callable(args[0]) and not kwargs:
        fn = args[0]
        args = ()
        return _decorate(fn)

    return _decorate