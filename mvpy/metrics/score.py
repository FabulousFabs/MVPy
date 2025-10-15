'''
Base metric class.
'''

import torch
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from math import prod

from typing import Union, Tuple, Optional, List

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

def score(model: Union[Pipeline, sklearn.base.BaseEstimator], metric: Union[Metric, Tuple[Metric]], X: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]] = None) -> List[Union[np.ndarray, torch.Tensor]]:
    """
    """
    
    # check metrics
    if isinstance(metric, Metric):
        metric = (metric,)
    
    # setup cache
    cache = {'X': X, 'y': y}
    
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
                # if not available, grab name
                cache[r] = getattr(model, r, None)
                
                # check validity
                if cache[r] is None:
                    raise ValueError(f'Attribute or method {r} requested by {m.name} does not exist in {model.__repr__}.')

                # check method
                if callable(cache[r]):
                    cache[r] = cache[r](X)
            
            # add as argument
            arg_i = reduce_(cache[r], m.reduce)
            arg.append(arg_i)
        
        # compute metric
        out[m.name] = m(*arg)
    
    # check output data
    if len(metric) == 1:
        out = out[metric[0].name]
    
    return out