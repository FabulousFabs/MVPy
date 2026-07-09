import torch
import numpy as np

from dataclasses import dataclass, replace
from typing import Union, Tuple, Callable, Dict

from .metric import Metric
from ..math import r2

@dataclass
class R2(Metric):
    """Implements :py:func:`mvpy.math.r2` as a :py:class:`~mvpy.metrics.Metric`.
    
    .. warning::
        This class extends :py:class:`~mvpy.metrics.Metric`. If you
        would like to apply this metric, please use the instance 
        exposed under :py:attr:`mvpy.metrics.r2`.
        
        For more information on this, please consult the documentation
        of :py:class:`~mvpy.metrics.Metric` and :py:func:`~mvpy.metrics.score`.
    
    Parameters
    ----------
    name : str, default='r2'
        The name of this metric.
    request : str | tuple[str], default=('y', 'predict', 'y_b')
        The values to request for scoring.
    reduce : int | tuple[int], default= (0,)
        The dimension(s) to reduce over.
    f : Callable, default=mvpy.math.r2
        The function to call.
    metadata : Dict[str, float | np.ndarray | torch.Tensor], default=None
        Additional metadata that was granted to this metric that is merged with requested data.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.dataset import make_meeg_categorical
    >>> from mvpy.estimators import RidgeClassifier
    >>> from mvpy.crossvalidation import cross_val_score
    >>> from mvpy.metric import r2
    >>> X, y = make_meeg_categorical()
    >>> clf = RidgeClassifier()
    >>> cross_val_score(clf, X, y, metric = r2)
    """
    
    name: str = 'r2'
    request: Union[str, Tuple[str]] = ('y', 'predict', 'y_b')
    reduce: Union[int, Tuple[int]] = (0,)
    f: Callable = r2
    metadata: Dict[str, float | np.ndarray | torch.Tensor | None] = None

# expose metric
r2 = R2()