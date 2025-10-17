'''
Base metric class.
'''

import torch
import numpy as np

from dataclasses import dataclass, replace
from typing import Union, Tuple, Any, Callable, Optional

@dataclass
class Metric:
    """
    """

    # defaults are set for sphinx linkcode to work
    name: str = 'metric'
    request: Tuple[str] = ('y', 'predict')
    reduce: Union[int, Tuple[int]] = (0,)
    f: Callable = lambda x: x
    
    def __call__(self, *args: Any, **kwargs: Any) -> Union[np.ndarray, torch.Tensor]:
        """
        """
        
        return self.f(*args, **kwargs)
    
    def mutate(self, name: Optional[str] = None, request: Optional[str] = None, reduce: Optional[Union[int, Tuple[int]]] = None, f: Optional[Callable] = None) -> "Metric":
        """
        """
        
        return Metric(
            name = name or self.name,
            request = request or self.request,
            reduce = reduce or self.reduce,
            f = f or self.f
        )