'''

'''

import numpy as np
import torch

from typing import Union

def _cosine_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes cosine similarity between vectors in x and y using
    NumPy as a backend.
    
    INPUTS:
        x   -   Tensor (samples x samples x features)
        y   -   Tensor (samples x samples x features)
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return np.sum(x * y, axis = -1) / (np.linalg.norm(x, axis = -1) * np.linalg.norm(y, axis = -1))

def _cosine_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Computes cosine similarity between vectors in x and y using
    torch as a backend.
    
    INPUTS:
        x   -   Tensor (samples x samples x features)
        y   -   Tensor (samples x samples x features)
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return (x * y).sum(-1) / (torch.linalg.norm(x, dim = -1) * torch.linalg.norm(y, dim = -1))

def cosine(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    '''
    Computes cosine similarity between `x` and `y`. Note that
    similarity is always computed over the final dimension in
    your inputs.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _cosine_torch(x, y)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _cosine_numpy(x, y)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')

def cosine_d(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    '''
    Computes cosine dissimilarity between `x` and `y`. Note that
    dissimilarity is always computed over the final dimension in
    your inputs.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        d   -   Distances
    '''
    
    return 1 - cosine(x, y)