import numpy as np
import torch

from typing import Union

def _raised_cosine_window_numpy(n: int, alpha: float) -> np.ndarray:
    """
    """
    
    return alpha - (1 - alpha) * np.cos(2 * np.pi * np.linspace(0, 1, n))

def _raised_cosine_window_torch(n: int, alpha: float, device: str = 'cpu') -> np.ndarray:
    """
    """
    
    return alpha - (1 - alpha) * torch.cos(2 * torch.pi * torch.linspace(0, 1, n, device = device))

def raised_cosine_window(n: int, alpha: float, backend: str = 'torch', device: str = 'cpu') -> Union[np.ndarray, torch.Tensor]:
    """
    """
    
    if backend == 'torch':
        return _raised_cosine_window_torch(n, alpha, device = device)
    elif backend == 'numpy':
        return _raised_cosine_window_numpy(n, alpha)

    raise ValueError(f'Unknown backend `{backend}`. Expected torch or numpy.')