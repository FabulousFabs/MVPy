'''
Functions to compute cosine (dis-)similarity in a 
nice and vectorised manner using either numpy or 
torch.
'''

import numpy as np
import torch

from typing import Union

def _cosine_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between x and y. Note that this function is not exported and should not be called directly.
    
    Parameters
    ----------
    x : np.ndarray
        Vector/Matrix/Tensor
    y : np.ndarray
        Vector/Matrix/Tensor
    
    Returns
    -------
    np.ndarray
        Similarities
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return np.sum(x * y, axis = -1) / (np.linalg.norm(x, axis = -1) * np.linalg.norm(y, axis = -1))

def _cosine_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarities between x and y. Note that this function is not exported and should not be called directly.
    
    Parameters
    ----------
    x : torch.Tensor
        Vector/Matrix/Tensor
    y : torch.Tensor
        Vector/Matrix/Tensor
    
    Returns
    -------
    torch.Tensor
        Similarities
    """
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return (x * y).sum(-1) / (torch.linalg.norm(x, dim = -1) * torch.linalg.norm(y, dim = -1))

def cosine(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Compute cosine similarities between x and y. Note that similarities are always computed over the final dimension.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Vector/Matrix/Tensor
    y : Union[np.ndarray, torch.Tensor]
        Vector/Matrix/Tensor
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Similarities
    
    Notes
    -----
    Cosine similarity is defined as:
    
    .. math::

        \\text{cosine}(x, y) = \\frac{\\mathbf{x} \\cdot \\mathbf{y}}{\\|\\mathbf{x}\\| \\|\\mathbf{y}\\|}
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import cosine
    >>> x = torch.tesor([1, 0])
    >>> y = torch.tensor([-1, 0])
    >>> cosine(x, y)
    tensor([-1.])
    """
    
    if isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor):
        return _cosine_torch(x, y)
    elif isinstance(x, np.ndarray) & isinstance(y, np.ndarray):
        return _cosine_numpy(x, y)
    
    raise ValueError(f'`x` and `y` must be of the same type, but got `{type(x)}` and `{type(y)}` instead.')

def cosine_d(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Compute cosine distances between x and y. Note that distances are always computed over the final dimension.
    
    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Vector/Matrix/Tensor
    y : Union[np.ndarray, torch.Tensor]
        Vector/Matrix/Tensor
    
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Distances
    
    Notes
    -----
    Cosine distances are computed as :math:`1 - \\text{cosine}(x, y)`.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.math import cosine_d
    >>> x = torch.tesor([1, 0])
    >>> y = torch.tensor([-1, 0])
    >>> cosine_d(x, y)
    tensor([2.])
    
    See also
    --------
    :func:`mvpy.math.cosine`
    """
    
    return 1 - cosine(x, y)