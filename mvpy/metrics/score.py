'''
Base metric class.
'''

import torch
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from copy import deepcopy
from math import prod

from typing import Union, Tuple, Optional, Dict

from .metric import Metric

def reduce_(X: Union[np.ndarray, torch.Tensor], dims: Union[int, Tuple[int]]) -> Union[np.ndarray, torch.Tensor]:
    """
    """
    
    # check dims
    if isinstance(dims, int):
        dims = (dims,)
    
    # set backend
    bk = torch if isinstance(X, torch.Tensor) else np
    
    # grab number
    ndim = X.ndim
    
    # normalise and validate
    dims = tuple(d if d >= 0 else d + ndim for d in dims)
    
    if len(set(dims)) != len(dims):
        raise ValueError(f'Specified dimensions in `dims` must be unique for `reduce_` in `metrics.score()`.')

    if any(d < 0 or d >= ndim for d in dims):
        raise ValueError(f'Specified dimensions in `dims` are out of range for `reduce_` in `metrics.score()`.')
    
    # move dims to final position(s)
    dims_sorted = tuple(sorted(dims))
    others = tuple(i for i in range(ndim) if i not in dims_sorted)
    X_m = bk.moveaxis(X, dims_sorted, tuple(range(len(others), len(others) + len(dims_sorted))))
    
    # flatten last
    final = prod(X_m.shape[len(others):])
    return X_m.reshape(*X_m.shape[:len(others)], final)

def score(model: Union[Pipeline, sklearn.base.BaseEstimator], metric: Union[Metric, Tuple[Metric]], X: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]:
    """
    """
    
    # check metrics
    if isinstance(metric, Metric):
        metric = (metric,)
    
    # setup cache
    cache = {'X': X, 'y': y}
    
    # setup reduced cache
    cache_reduced = {}
    
    # setup dummy
    out = {m.name: [] for m in metric}
    
    # compute metrics
    for m in metric:
        # setup args
        arg = []
        
        # loop over requested data
        for r in m.request:
            # check cache status
            if r not in cache:
                # check metadata
                if r in m.metadata:
                    cache[r] = m.metadata[r]
                # check model attributes
                elif not hasattr(model, r):
                    # check for pipeline
                    if isinstance(model, Pipeline):
                        # request from final element
                        cache[r] = getattr(model[-1], r, None)
                    else:
                        # otherwise, unavailable element
                        cache[r] = None
                else:
                    # retrieve directly
                    cache[r] = getattr(model, r, None)

                # check method
                if callable(cache[r]):
                    cache[r] = cache[r](X)
            
            # check reduced cache
            if m.reduce not in cache_reduced:
                # setup cache
                cache_reduced[m.reduce] = {}
            
            # search reduced cache
            if r not in cache_reduced[m.reduce]:
                # handle cache miss
                arg_i = cache[r]
                
                # handle None types
                if arg_i is not None:
                    arg_i = reduce_(arg_i, m.reduce)
                
                # fill cache
                cache_reduced[m.reduce][r] = arg_i
            
            # safely append argument
            arg.append(cache_reduced[m.reduce][r])
        
        # compute metric
        out[m.name] = m(*arg)
    
    # check output data
    if len(metric) == 1:
        out = out[metric[0].name]
    
    return out