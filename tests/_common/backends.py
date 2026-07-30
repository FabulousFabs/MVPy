import numpy as np
import torch
import pytest

from dataclasses import dataclass
from tests._common.dtypes import find_dtype

from typing import Union, Any

# define backend types
NUMPY = pytest.param(
    'numpy',
    id = 'numpy'
)

TORCH_CPU = pytest.param(
    'torch-cpu',
    id = 'torch-cpu'
)

TORCH_CUDA = pytest.param(
    'torch-cuda',
    id = 'torch-cuda',
    marks = pytest.mark.cuda
)

TORCH_MPS = pytest.param(
    'torch-mps',
    id = 'torch-mps',
    marks = pytest.mark.mps
)

# define combinations
# NOTE: Currently, MPS devices are not capable
# of many of the computations required in MVPy.
# Therefore, we exclude them here, for the time
# being.
CPU_BACKENDS = [NUMPY, TORCH_CPU]
TORCH_BACKENDS = [TORCH_CPU, TORCH_CUDA]
NUMPY_BACKENDS = [NUMPY]
ALL_BACKENDS = [NUMPY, TORCH_CPU, TORCH_CUDA]

# define backend class
@dataclass(frozen = True)
class Backend:
    """
    """
    
    name: str
    device: torch.device | None = None
    
    @property
    def is_numpy(self) -> bool:
        return self.name == "numpy"

    @property
    def is_torch(self) -> bool:
        return self.name.startswith('torch')
    
    @property
    def is_cuda(self) -> bool:
        return self.device is not None and self.device.type == 'cuda'
    
    @property
    def is_mps(self) -> bool:
        return self.device is not None and self.device.type == 'mps'
    
    def asarray(self, value: Any, *, dtype: Any = np.float64) -> Union[np.ndarray, torch.Tensor]:
        """
        """
        
        # make array
        array = np.asarray(value, dtype = dtype)
        
        # check output type
        if self.name == 'numpy':
            return array
        
        # otherwise, handle torch
        torch_dtype = find_dtype(self.name, dtype)
        return torch.as_tensor(array, dtype = torch_dtype, device = self.device)
    
    def asarrays(self, *args: Any, dtype: Any = np.float64) -> Union[tuple[np.ndarray], tuple[torch.Tensor]]:
        """
        """
        
        return tuple([
            self.asarray(args[i], dtype = dtype)
            for i in range(len(args))
        ])
    
    def to_numpy(self, value: Any) -> np.ndarray:
        """
        """
        
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)
    
    def to_torch(self, value: Any) -> torch.Tensor:
        """
        """
        
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return torch.as_tensor(value)