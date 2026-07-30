import numpy as np
import torch
import pytest

from typing import Any

# define datatype conversion
DTYPE_NUMPY_TORCH = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64
}

DTYPE_TORCH_NUMPY = {
    DTYPE_NUMPY_TORCH[key]: key
    for key in DTYPE_NUMPY_TORCH
}

def find_dtype(backend: str, dtype: Any) -> Any:
    """Find converted dtype in opposite backend.
    
    Parameters
    ----------
    backend : {'numpy', 'torch', 'torch-cpu', 'torch-cuda', 'torch-mps', str}
        Backend to convert to.
    dtype : Any
        Current datatype.
    
    Returns
    -------
    dtype_conv : Any
        Converted datatype in other backend.
    """
    
    # find source
    source = DTYPE_NUMPY_TORCH
    
    if backend == 'numpy':
        source = DTYPE_TORCH_NUMPY
    
    # check key
    if np.dtype(dtype) not in source:
        raise ValueError(
            f'Unknown dtype {np.dtype(dtype)} for {source}.'
        )
    
    return source[np.dtype(dtype)]

def to_numpy(value: Any) -> np.ndarray:
    """Convert value to numpy.
    
    Parameters
    ----------
    value : Any
        Value to be converted.
    
    Returns
    -------
    array : np.ndarray
        Converted array.
    """
    
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    
    return np.asarray(value)