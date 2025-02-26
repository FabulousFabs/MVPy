'''
Functions to compute Spearman correlation or distance in a 
nice and vectorised manner using either numpy or 
torch.
'''

import numpy as np
import torch

from typing import Union

from .pearsonr import _pearsonr_numpy, _pearsonr_torch
from .rank import _rank_numpy, _rank_torch

def spearmanr(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    '''
    Computes Spearman r between `x` and `y`. Note that
    correlation is always computed over the final dimension in
    your inputs.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _pearsonr_torch(_rank_torch(x), _rank_torch(y))
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _pearsonr_numpy(_rank_numpy(x), _rank_numpy(y))
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')

def spearmanr_d(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    '''
    Computes Spearman distance between `x` and `y`. Note that
    distance is always computed over the final dimension in
    your inputs.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        d   -   Distances
    '''
    
    return 1 - spearmanr(x, y)