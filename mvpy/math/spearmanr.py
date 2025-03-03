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
    """Compute Spearman correlation between x and y. Note that correlations are always computed over the final dimension in your inputs.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Matrix to compute correlation with.
    y : Union[np.ndarray, torch.Tensor]
        Matrix to compute correlation with.
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Spearman correlations.
    
    Notes
    -----
    Spearman correlations are defined as Pearson correlations between the ranks of x and y.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import spearmanr
    >>> x = torch.tensor([1, 5, 9])
    >>> y = torch.tensor([1, 50, 60])
    >>> spearmanr(x, y)
    tensor(1., dtype=torch.float64)
    
    See also
    --------
    :func:`mvpy.math.pearsonr`
    :func:`mvpy.math.rank`
    """
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _pearsonr_torch(_rank_torch(x), _rank_torch(y))
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _pearsonr_numpy(_rank_numpy(x), _rank_numpy(y))
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')

def spearmanr_d(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Compute Spearman distance between x and y. Note that distances are always computed over the final dimension in your inputs.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Matrix to compute distance with.
    y : Union[np.ndarray, torch.Tensor]
        Matrix to compute distance with.
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Spearman distances.
    
    Notes
    -----
    Spearman distances are defined as :math:`1 - \\text{spearmanr}(x, y)`.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import spearmanr
    >>> x = torch.tensor([1, 5, 9])
    >>> y = torch.tensor([1, 50, 60])
    >>> spearmanr_d(x, y)
    tensor(0., dtype=torch.float64)
    
    See also
    --------
    :func:`mvpy.math.spearmanr`
    :func:`mvpy.math.pearsonr`
    :func:`mvpy.math.rank`
    """
    
    return 1 - spearmanr(x, y)