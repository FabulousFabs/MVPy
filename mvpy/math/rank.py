'''
Functions to rank data in a nice and vectorised manner using 
either numpy or torch.
'''

import numpy as np
import torch

from typing import Union

def _rank_numpy(x: np.ndarray) -> np.ndarray:
    """Rank numpy array along final dimension. Ties are computed as averages.

    Parameters
    ----------
    x : np.ndarray
        Tensor to rank.
    
    Returns
    -------
    np.ndarray
        Ranked data.
    """
    
    # make sure x is floating point
    if np.issubdtype(x.dtype, np.floating) == False:
        x = x.astype(float)
    
    # get sorted indices
    sorted_idx = np.argsort(x, axis = -1)
    sorted_x = np.take_along_axis(x, sorted_idx, axis = -1)
    
    # create rank tensor
    r = np.zeros_like(sorted_x, dtype = x.dtype)
    
    # compute ranks (without ties)
    for f_i in range(r.shape[-1]):
        r[..., f_i] = np.sum(sorted_x < sorted_x[..., f_i, None], axis = -1).astype(np.float64) + 1.0
    
    # resolve ties through averaging
    for f_i in range(r.shape[-1]):
        delta = np.sum(sorted_x == sorted_x[..., f_i, None], axis = -1).astype(np.float64) - 1.0
        r[..., f_i] += delta / 2.0
    
    # unsort the ranked data
    inv_sorted_idx = np.argsort(sorted_idx, axis = -1)
    r = np.take_along_axis(r, inv_sorted_idx, axis = -1)
    
    return r

def _rank_torch(x: torch.Tensor) -> torch.Tensor:
    """Rank torch tensor along final dimension. Ties are computed as averages.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to rank.
    
    Returns
    -------
    torch.Tensor
        Ranked data.
    """
    
    # make sure x is a floating point tensor
    if torch.is_floating_point(x) == False:
        device = x.device
        x = x.to(torch.float64).to(device)
    
    # sort tensor
    v, i = x.sort(dim = -1)
    
    # setup rank tensor
    r = torch.zeros_like(v, dtype = x.dtype, device = x.device)

    # loop over features to find ranks (with ties)
    for f_i in range(r.shape[-1]):
        r[...,f_i,None] = (v < v[...,f_i,None]).sum(-1, keepdim = True) + 1

    # resolve ties through average
    for f_i in range(r.shape[-1]):
        delta = (v == v[...,f_i,None]).sum(-1, keepdim = True) - 1
        r[...,f_i,None] += (delta / 2).to(dtype = x.dtype)

    # unsort ranked data
    r = r.gather(-1, i.argsort(-1))
    
    return r

def rank(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Rank data in x along its final feature dimension. Ties are computed as averages.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Unranked data ([samples x ...] x features).
    
    Returns
    -------
    r : Union[np.ndarray, torch.Tensor]
        Ranked data ([samples x ...] x features).
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import rank
    >>> x = torch.tensor([2, 0.5, 1, 1])
    >>> rank(x)
    tensor([4.0000, 1.0000, 2.5000, 2.5000])
    """
    
    if isinstance(x, torch.Tensor):
        return _rank_torch(x)
    elif isinstance(x, np.ndarray):
        return _rank_numpy(x)
    
    raise ValueError(f'`x` must be of type np.ndarray or torch.Tensor, but received `{type(x)}` instead.')