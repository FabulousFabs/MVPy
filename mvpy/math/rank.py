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
    
    # save shape
    shape = x.shape
    n_features = shape[-1]
    
    # flatten leading dimensions
    x_f = x.reshape(-1, n_features)
    
    # sort along final dimension
    i = np.argsort(x_f, axis = -1)
    v = np.take_along_axis(x_f, i, axis = -1)
    
    # make positions
    pos = np.arange(n_features, dtype = np.int64)[None,:]
    
    # identitfy starts of tie groups
    mask_s = np.empty(v.shape, dtype = bool)
    mask_s[:,0] = True
    mask_s[:,1:] = v[:,1:] != v[:,:-1]
    
    # forward fill group starts
    group_s = np.maximum.accumulate(
        np.where(
            mask_s,
            pos,
            0
        ),
        axis = -1
    )
    
    # identify ends of tie groups
    mask_e = np.empty(v.shape, dtype = bool)
    mask_e[:,-1] = True
    mask_e[:,:-1] = v[:,:-1] != v[:,1:]
    
    # backward fill group ends
    group_e = np.minimum.accumulate(
        np.where(
            mask_e,
            pos,
            n_features - 1
        )[:,::-1],
        axis = 1
    )[:,::-1]
    
    # average rank of each tie group
    ranks_sorted = (group_s + group_e).astype(x_f.dtype, copy = False) / 2.0 + 1.0
    
    # scatter ranks back to original order
    out = np.empty_like(ranks_sorted, dtype = x_f.dtype)
    np.put_along_axis(out, i, ranks_sorted, axis = 1)
    
    return out.reshape(shape)

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
    
    # save shape
    shape = x.shape
    n_features = shape[-1]
    
    # flatten leading dimensions
    x_f = x.reshape(-1, n_features)
    
    # sort along final dimension
    v, i = x_f.sort(-1)
    
    # make positions
    pos = torch.arange(
        n_features,
        device = x_f.device,
        dtype = torch.int64
    ).unsqueeze(0)
    
    # identifiy starts of tie groups
    mask_s = torch.empty(v.shape, dtype = torch.bool, device = x_f.device)
    mask_s[:,0] = True
    mask_s[:,1:] = v[:,1:] != v[:,:-1]
    
    # forward fill group starts
    group_s = torch.where(
        mask_s, 
        pos,
        0
    ).cummax(1).values
    
    # identifiy ends of tie groups
    mask_e = torch.empty(v.shape, dtype = torch.bool, device = x_f.device)
    mask_e[:,-1] = True
    mask_e[:,:-1] = v[:,:-1] != v[:,1:]
    
    # backward fill group ends
    group_e = torch.where(
        mask_e,
        pos,
        n_features - 1
    ).flip(1).cummin(1).values.flip(1)
    
    # average rank of each tie group
    ranks_sorted = (group_s + group_e).to(dtype = x_f.dtype) / 2.0 + 1.0
    
    # scatter back to original
    out = torch.empty_like(ranks_sorted, dtype = x_f.dtype, device = x_f.device)
    out.scatter_(1, i, ranks_sorted)
    
    return out.reshape(shape)

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