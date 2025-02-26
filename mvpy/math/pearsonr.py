'''
Functions to compute Pearson correlation or distance in a 
nice and vectorised manner using either numpy or 
torch.
'''

import numpy as np
import torch

from typing import Union

def _pearsonr_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes pearson correlation between vectors in x and y using
    NumPy as a backend.
    
    INPUTS:
        x   -   Tensor (samples x samples x features)
        y   -   Tensor (samples x samples x features)
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    Δx, Δy = (x - x.mean(-1, keepdims = True)), (y - y.mean(-1, keepdims = True))
    
    return np.sum(Δx * Δy, axis = -1) / np.sqrt(np.sum(Δx ** 2, axis = -1) * np.sum(Δy ** 2, axis = -1))

def _pearsonr_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Computes pearson correlation between vectors in x and y using
    torch as a backend.
    
    INPUTS:
        x   -   Tensor (samples x samples x features)
        y   -   Tensor (samples x samples x features)
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    Δx, Δy = (x - x.mean(-1, keepdim = True)), (y - y.mean(-1, keepdim = True))
    
    return torch.sum(Δx * Δy, -1) / torch.sqrt(torch.sum(Δx ** 2, -1) * torch.sum(Δy ** 2, -1))

def pearsonr(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    '''
    Computes Pearson r between `x` and `y`. Note that
    correlation is always computed over the final dimension in
    your inputs.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _pearsonr_torch(x, y)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _pearsonr_numpy(x, y)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')

def pearsonr_d(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    '''
    Computes Pearson distance between `x` and `y`. Note that
    distance is always computed over the final dimension in
    your inputs.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        d   -   Distances
    '''
    
    return 1 - pearsonr(x, y)