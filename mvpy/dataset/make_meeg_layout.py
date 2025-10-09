'''
Functions to generate a layout of M-/EEG sensors.
'''

import numpy as np
import torch
import math

from typing import Union

def make_meeg_layout(n_channels: int, backend: str = 'torch', device: str = 'cpu') -> Union[np.ndarray, torch.Tensor]:
    """Create a concentric channel layout for M-EEG sensors.
    
    Parameters
    ----------
    n_channels : int
        Number of channels/sensors.
    backend : str, default='torch'
        Which backend to use (torch or numpy)?
    device : str, default='cpu'
        Which device to use?
    
    Returns
    -------
    layout : Union[np.ndarray, torch.Tensor]
        The sensor layout of shape (n_channels, n_directions) where n_directions corresponds to three, i.e. (x, y, z).
    """
    
    # choose grid shape close to square
    n_cols = math.ceil(math.sqrt(n_channels))
    n_rows = math.ceil(n_channels / n_cols)
    
    # center points in each cell
    x = torch.linspace(0.5 / n_cols, 1 - 0.5 / n_cols, n_cols, device = device)
    y = torch.linspace(0.5 / n_rows, 1 - 0.5 / n_rows, n_rows, device = device)
    
    # compute grid
    x, y = torch.meshgrid(x, y, indexing = 'xy')
    grid = torch.stack([x.flatten(), y.flatten()], dim = -1)
    
    # make concentric grid
    grid = 2.0 * grid - 1.0
    a, b = grid[:,0], grid[:,1]
    
    # handle regions
    mask = (a != 0.) | (b != 0.)
    a_m, b_m = a[mask], b[mask]
    abs_a, abs_b = a_m.abs(), b_m.abs()
    
    m1 = abs_a > abs_b
    r1 = abs_a[m1]
    phi1 = (math.pi / 4) * (b_m[m1] / (a_m[m1] + 1e-12))
    x1 = torch.sign(a_m[m1]) * r1 * torch.cos(phi1)
    y1 = torch.sign(a_m[m1]) * r1 * torch.sin(phi1)
    
    m2 = ~m1
    r2 = abs_b[m2]
    phi2 = (math.pi / 2) - (math.pi / 4) * (a_m[m2] / (b_m[m2] + 1e-12))
    x2 = torch.sign(b_m[m2]) * r2 * torch.cos(phi2)
    y2 = torch.sign(b_m[m2]) * r2 * torch.sin(phi2)
    
    x_mask, y_mask = torch.zeros_like(a), torch.zeros_like(b)
    x_mask[mask] = torch.cat([x1, x2], dim = 0)
    y_mask[mask] = torch.cat([y1, y2], dim = 0)
    disk = torch.stack([x_mask, y_mask], dim = -1)
    
    # square to upper hemisphere
    x, y = disk[...,0], disk[...,1]
    r2 = x * x + y * y
    z = torch.sqrt(torch.clamp(1 - r2, min = 0.0))
    grid = torch.stack([x, y, z], dim = -1)
    
    # if required, convert
    if backend == 'numpy':
        grid = grid.cpu().numpy()
    
    return grid[0:n_channels,:]