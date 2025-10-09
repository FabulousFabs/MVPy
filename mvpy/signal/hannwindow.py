import numpy as np
import torch

from typing import Union

from .raisedcosinewindow import raised_cosine_window

def hann_window(n: int, backend: str = 'torch', device: str = 'cpu') -> Union[np.ndarray, torch.Tensor]:
    """
    """
    
    return raised_cosine_window(n, 0.5, backend = backend, device = device)