"""Helpers for asv benchmarking.
"""

import numpy as np
import torch

from dataclasses import dataclass

from typing import Dict, Union, Any

"""Define all backends.
"""
BACKENDS_NUMPY = ["numpy"]
BACKENDS_TORCH = ["torch"]
BACKENDS_CUDA = ["cuda"]
BACKENDS_MPS = ["mps"]

"""Combine CPU backends.
"""
BACKENDS_CPU = BACKENDS_NUMPY + BACKENDS_TORCH

"""Check and combine all GPU backends.

Note that this will check for availability. This means that
BACKENDS_GPU contains exclusively those backends that are
currently available, but may therefore not include all backends
that are directly targetable.
"""
BACKENDS_GPU = []
if torch.backends.mps.is_available():
    BACKENDS_GPU += BACKENDS_MPS
if torch.cuda.is_available():
    BACKENDS_GPU += BACKENDS_CUDA

"""Combine CPU and GPU backends.
"""
BACKENDS = BACKENDS_CPU + BACKENDS_GPU

"""Define parameter types.
"""

class Variable:
    """Dummy class for type checking.
    """
    
    def __init__(self, *args, **kwargs):
        """Simply pass initiation.
        """
        
        pass
    
    def sample(self, *args, **kwargs) -> None:
        """Sample from variable.
        """
        
        raise NotImplementedError(
            f'Sampling has not been implemented.'
        )

@dataclass(frozen = True)
class Normal(Variable):
    """Continuous variable from normal distribution.
    
    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of parameter to generate.
    mean : float, default=0.0
        Mean of the distribution.
    std : float, default=1.0
        Standard deviation of the distribution.
    dtype_np : Any, default=np.float32
        Datatype to use for numpy arrays.
    dtype_tr : Any, default=torch.float32
        Datatype to use for torch tensors.
    """
    
    shape: tuple[int, ...]
    mean: float = 0.0
    std: float = 1.0
    dtype_np: Any = np.float32
    dtype_tr: Any = torch.float32
    
    def sample(self, rng_np: Any, rng_tr: Any, backend: str) -> np.ndarray | torch.Tensor:
        """Sample from this variable.
        
        Parameters
        ----------
        rng_np : Any
            Random number generator for numpy.
        rng_tr : Any
            Random number generator for torch.
        backend : str
            Backend to use.
        
        Returns
        -------
        data : np.ndarray | torch.Tensor
            The sampled data.
        """
        
        # verify backend
        if backend not in BACKENDS:
            raise ValueError(
                f'Unknown backend: {backend}.'
            )
        
        # check backend
        if backend == 'numpy':
            # make numpy
            return rng_np.normal(
                loc = self.mean,
                scale = self.std,
                size = self.shape
            ).astype(self.dtype_np)
        
        # otherwise, make torch
        device = 'cpu' if backend in BACKENDS_TORCH else backend
        
        return self.mean + self.std * torch.randn(
            *self.shape,
            dtype = self.dtype_tr,
            generator = rng_tr
        ).to(device)

@dataclass(frozen = True)
class Categorical(Variable):
    """Categorical data from uniform distribution. 
    
    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of parameter to generate.
    low : int, default=0
        Lower bound for sampling.
    high : int, default=1
        Upper bound for sampling.
    dtype_np : Any, default=np.float32
        Datatype to use for numpy arrays.
    dtype_tr : Any, default=torch.float32
        Datatype to use for torch tensors.
    """
    
    shape: tuple[int, ...]
    low: int = 0
    high: int = 2
    dtype_np: Any = np.int32
    dtype_tr: Any = torch.int32
    
    def sample(self, rng_np: Any, rng_tr: Any, backend: str) -> np.ndarray | torch.Tensor:
        """Sample from this variable.
        
        Parameters
        ----------
        rng_np : Any
            Random number generator for numpy.
        rng_tr : Any
            Random number generator for torch.
        backend : str
            Backend to use.
        
        Returns
        -------
        data : np.ndarray | torch.Tensor
            The sampled data.
        """
        
        # verify backend
        if backend not in BACKENDS:
            raise ValueError(
                f'Unknown backend: {backend}.'
            )
        
        # check backend
        if backend == 'numpy':
            # make numpy
            return rng_np.integers(
                self.low,
                self.high,
                size = self.shape,
            ).astype(self.dtype_np)
        
        # make torch
        device = 'cpu' if backend in BACKENDS_TORCH else backend
        
        return torch.randint(
            low = self.low,
            high = self.high,
            size = self.shape,
            generator = rng_tr,
            dtype = self.dtype_tr
        ).to(device)

@dataclass(frozen = True)
class Constant(Variable):
    """Constant data to use in samples.
    
    Parameters
    ----------
    value : Any
        Constants to use.
    """
    
    value: Any
    
    def sample(self, rng_np: Any, rng_tr: Any, backend: str) -> Any:
        """Sample from this variable.
        
        Parameters
        ----------
        rng_np : Any
            Random number generator for numpy.
        rng_tr : Any
            Random number generator for torch.
        backend : str
            Backend to use.
        
        Returns
        -------
        data : Any
            The sampled data.
        """
        
        # verify backend
        if backend not in BACKENDS:
            raise ValueError(
                f'Unknown backend: {backend}.'
            )
        
        # check backend
        if backend == 'numpy':
            # make numpy
            return self.value

        # otherwise, make torch
        device = 'cpu' if backend in BACKENDS_TORCH else backend

        return torch.tensor(self.value).to(device)

class BackendCase:
    """Generate benchmark data for backends and cases of varying shapes.
    
    Attributes
    ----------
    name : str
        Unique name of the benchmark.
    cases : Dict[str, Tuple]
        Dictionary of (case, shape) pairs.
    backends : List[str]
        List of backends to use for this benchmark.
    param_names : List[str], default=["case", "backend"]
        Named parameters of the benchmark.
    """
    
    name = "anonymous"
    cases = {}
    backends = BACKENDS
    param_names = ["case", "backend"]
    
    def _build_cache(self) -> Dict:
        """Create a cache entry.
        
        Returns
        -------
        cache : Dict[Tuple, np.ndarray | torch.Tensor]
            The cache structure with a tuple identifier (name, case, backend) and the corresponding data.
        """
        
        # setup generators
        self.rng_np = np.random.default_rng(0)
        self.rng_tr = torch.Generator().manual_seed(0)
        
        # make cases
        return {
            (self.name, case, backend): self.make_case(
                variable = self.cases[case],
                backend = backend
            )
            for case in self.cases
            for backend in self.backends
        }
    
    def make_case(self, variable: Variable | Dict[str, Variable], backend: str) -> Union[np.ndarray, torch.Tensor | Dict[str, np.ndarray | torch.Tensor]]:
        """Make data for a given case.
        
        Parameters
        ----------
        variable : Variable | Dict[str, Variable]
            The variables to create for this benchmark. If dictionary, creates data as a dictionary with same keys.
        backend : str
            Which backend to use.
        
        Returns
        -------
        data : np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor]
            The created data structure.
        """
        
        # verify type
        if (not isinstance(variable, dict)) and (not isinstance(variable, Variable)):
            raise ValueError(
                f'Must supply either dictionary of variables or variable.'
            )
        
        # check type
        if isinstance(variable, dict):
            # if desired, make dictionary data
            return {
                key: variable[key].sample(
                    self.rng_np,
                    self.rng_tr,
                    backend
                )
                for key in variable
            }
        
        # otherwise, make tensor
        return variable.sample(
            self.rng_np,
            self.rng_tr,
            backend
        )
    
    def setup(self, cache, case, backend) -> None:
        """Setup the data from our cache.
        
        Parameters
        ----------
        cache : Dict[Tuple, np.ndarray | torch.Tensor | Tuple[np.ndarray | torch.Tensor]
            Cache structure.
        case : str
            Name of the case.
        backend : str
            Backend to use.
        """
        
        self.data = cache[(self.name, case, backend)]