'''
Functions to compute euclidean distances in a 
nice and vectorised manner using either numpy
or torch.
'''

import numpy as np
import torch

from typing import Union

def _euclidean_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes euclidean distance between vectors in x and y using
    NumPy as a backend.
    
    INPUTS:
        x   -   Tensor ([[samples x ]samples x ]features)
        y   -   Tensor ([[samples x ]samples x ]features)
    
    OUTPUTS:
        s   -   Distance
    '''
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return np.sqrt(np.sum((x - y) ** 2, axis = -1))

def _euclidean_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Computes euclidean distance between vectors in x and y using
    torch as a backend.
    
    INPUTS:
        x   -   Tensor ([[samples x ]samples x ]features)
        y   -   Tensor ([[samples x ]samples x ]features)
    
    OUTPUTS:
        s   -   Distance
    '''
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return torch.sqrt(torch.sum((x - y) ** 2, -1))

def euclidean(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    '''
    Computes euclidean distance between `x` and `y`. Note that
    distance is always computed over the final dimension in
    your inputs.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _euclidean_torch(x, y)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _euclidean_numpy(x, y)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')