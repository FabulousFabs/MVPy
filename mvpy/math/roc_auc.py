'''
Functions to compute accuracy in a 
nice and vectorised manner using either numpy or 
torch.
'''

import numpy as np
import torch

from typing import Union

from ..math import rank

def _roc_auc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Compute ROC AUC between y_true and y_score. Note that this function is not exported and should not be called directly.
    
    Parameters
    ----------
    y_true : np.ndarray
        Vector/Matrix/Tensor
    y_score : np.ndarray
        Vector/Matrix/Tensor
    
    Returns
    -------
    np.ndarray
        ROC AUC
    """
    
    # get unique labels
    L = np.unique(y_true)
    
    # ensure at least two labels
    if L.shape[0] < 2:
        raise ValueError('`y_true` must have at least two unique values.')
    
    # if multi-label, we need to create a new axis for an OvR style approach
    if L.shape[0] > 2:
        # add our new dimension
        y = np.full(y_true.shape, -1)[...,np.newaxis]
        
        # loop over each class
        for i, L_i in enumerate(L):
            y[y_true == L_i,i] = 1
        
        # update labels
        y_true = y.swapaxes(-1, -2)
    
    # update unique labels
    L_a = np.unique(y_true)
    
    # force conversion to [0, 1]
    y_true = (y_true == L_a.max()).astype(int)
    
    # compute ranks
    ranks = rank(y_score)
    
    # find sum of ranks for positive class
    R_pos = np.sum(ranks * (y_true == 1), axis = -1)
    
    # find positive and negative samples
    P = np.sum(y_true, axis = -1)
    N = y_true.shape[-1] - P
    
    # avoid division by zero
    mask = (P != 0.0) & (N != 0.0)
    roc_auc = np.full(y_true.shape[0:-1], np.nan)
    roc_auc[mask] = (R_pos[mask] - P[mask] * (P[mask] + 1) / 2) / (P[mask] * N[mask])
    
    # take macro average if multi-class
    if len(L.shape) != len(L_a.shape):
        roc_auc = np.nanmean(roc_auc, axis = -1)
    
    return roc_auc

def _roc_auc_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
    """Compute ROC AUC between y_true and y_score. Note that this function is not exported and should not be called directly.
    
    Parameters
    ----------
    y_true : torch.Tensor
        Vector/Matrix/Tensor
    y_score : torch.Tensor
        Vector/Matrix/Tensor
    
    Returns
    -------
    torch.Tensor
        ROC AUC
    """
    
    # get unique labels
    L = torch.unique(y_true)
    
    # ensure at least two labels
    if L.shape[0] < 2:
        raise ValueError('`y_true` must have at least two unique values.')
    
    # if multi-label, we need to create a new axis for an OvR style approach
    if L.shape[0] > 2:
        # add our new dimension
        y = torch.full(y_true.shape, -1, dtype = y_true.dtype, device = y_true.device).unsqueeze(-1)
        
        # loop over each class
        for i, L_i in enumerate(L):
            y[y_true == L_i,i] = 1
        
        # update labels
        y_true = y.swapaxes(-1, -2)
    
    # update unique labels
    L_a = torch.unique(y_true)
    
    # force conversion to [0, 1]
    y_true = (y_true == L_a.max()).to(y_true.dtype)
    
    # compute ranks
    ranks = rank(y_score)
    
    # find sum of ranks for positive class
    R_pos = torch.sum(ranks * (y_true == 1), -1)
    
    # find positive and negative samples
    P = torch.sum(y_true, -1)
    N = y_true.shape[-1] - P
    
    # avoid division by zero
    mask = (P != 0.0) & (N != 0.0)
    roc_auc = torch.full(y_true.shape[0:-1], torch.nan, dtype = y_score.dtype, device = y_score.device)
    roc_auc[mask] = (R_pos[mask] - P[mask] * (P[mask] + 1) / 2) / (P[mask] * N[mask])
    
    # take macro average if multi-class
    if len(L.shape) != len(L_a.shape):
        roc_auc = torch.nanmean(roc_auc, -1)
    
    return roc_auc

def roc_auc(y_true: Union[np.ndarray, torch.Tensor], y_score: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Compute ROC AUC score between y_true and y_score. Note that ROC AUC is always computed over the final dimension.
    
    Parameters
    ----------
    y_true : Union[np.ndarray, torch.Tensor]
        Vector/Matrix/Tensor
    y_score : Union[np.ndarray, torch.Tensor]
        Vector/Matrix/Tensor
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Accuracy
    
    Notes
    -----
    ROC AUC is computed using the Mann-Whitney U formula:
    
    .. math::

        \\text{ROCAUC}(y, \\hat{y}) = \\frac{R_{+} - \\frac{P * (P + 1)}{2}}}{P * N}
    
    where :math:`R_{+}` is the sum of ranks for positive classes, :math:`P` is the number of positive samples, and :math:`N` is the number of negative samples.
    
    In the case that labels are not binary, we create unique binary labels by one-hot encoding the labels and then take the macro-average over classes.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import roc_auc
    >>> y_true = torch.tensor([1., 0.])
    >>> y_score = torch.tensor([-1., 1.])
    >>> roc_auc(y_true, y_score)
    tensor(0.)
    """
    
    if isinstance(y_true, torch.Tensor) & isinstance(y_score, torch.Tensor):
        return _roc_auc_torch(y_true, y_score)
    elif isinstance(y_true, np.ndarray) & isinstance(y_score, np.ndarray):
        return _roc_auc_numpy(y_true, y_score)
    
    raise ValueError(f'`y_true` and `y_score` must be of the same type, but got `{type(y_true)}` and `{type(y_score)}` instead.')