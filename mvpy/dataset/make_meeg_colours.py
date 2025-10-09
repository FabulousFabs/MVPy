'''
Functions to generate a layout of M-/EEG sensors.
'''

import numpy as np
import torch
import math

from typing import Union

def make_meeg_colours(ch_pos: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Create spatial colours for sensor layout.
    
    Parameters
    ----------
    ch_pos : Union[np.ndarray, torch.Tensor]
        Channel positons of shape (n_channels, n_dims) where n_dims is (x, y, z).
    
    Returns
    -------
    colours : Union[np.ndarray, torch.Tensor]
        The sensor colours of shape (n_channels, rgb).
    """
    
    # check types
    is_torch = isinstance(ch_pos, torch.Tensor)
    is_numpy = isinstance(ch_pos, np.ndarray)
    
    if (not is_torch) and not (is_numpy):
        raise ValueError(f'Expected `ch_pos` to be either np.ndarray or torch.Tensor, but got {type(ch_pos)}.')
    
    # check length
    if len(ch_pos.shape) != 2:
        raise ValueError(f'Expected `ch_pos` to be of shape (n_channels, n_coords), but got {ch_pos.shape}.')
    
    # check position dims
    if ch_pos.shape[1] < 3:
        raise ValueError(f'Expected `ch_pos` to be contain at least three coordinates (x, y, z), but got {ch_pos.shape[1]}.')
    
    if ch_pos.shape[1] > 3:
        ch_pos = ch_pos[:,0:3]
    
    # make colours
    if is_torch:
        ch_col = ch_pos.clone()
        ch_col = ch_col - ch_col.min(dim = 0, keepdim = True)[0]
        ch_col = ch_col / ch_col.max(dim = 0, keepdim = True)[0]
    else:
        ch_col = ch_pos.copy()
        ch_col -= ch_col.min(axis = 0, keepdims = True)
        ch_col /= ch_col.max(axis = 0, keepdims = True)
    
    return ch_col