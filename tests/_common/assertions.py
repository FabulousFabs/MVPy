import numpy as np
import torch

from tests._common.dtypes import to_numpy

from typing import Union

def assert_allclose(observed: Union[np.ndarray, torch.Tensor], expected: Union[np.ndarray | torch.Tensor], *, rtol: float = 1e-6, atol: float = 1e-8) -> None:
    """Assert all values are close (with automatic conversions).
    
    Parameters
    ----------
    observed : np.ndarray | torch.Tensor
        Observed values.
    expected : np.ndarray | torch.Tensor
        Expected values.
    rtol : float, default=1e-6
        Relative tolerance.
    atol : float, default=1e-8
        Absolute tolerance.
    """
    
    np.testing.assert_allclose(
        to_numpy(observed),
        to_numpy(expected),
        rtol = rtol,
        atol = atol,
    )

def assert_equalshape(observed: Union[np.ndarray, torch.Tensor], expected: Union[np.ndarray, torch.Tensor]) -> None:
    """Assert arrays have equal shapes (with automatic conversions).
    
    Parameters
    ----------
    observed : np.ndarray | torch.Tensor
        Observed values.
    expected : np.ndarray | torch.Tensor
        Expected values.
    """
    
    np.testing.assert_equal(
        to_numpy(observed).shape, 
        to_numpy(expected).shape
    )

def assert_equaldevice(observed: torch.Tensor, expected: torch.Tensor) -> None:
    """Assert that two tensors live on the same device type.
    
    Parameters
    ----------
    observed : torch.Tensor
        Observed values.
    expected : torch.Tensor
        Expected values.
    """
    
    np.testing.assert_string_equal(
        observed.device.type,
        expected.device.type
    )
    