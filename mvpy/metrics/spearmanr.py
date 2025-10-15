import torch
import numpy as np

from dataclasses import dataclass, replace
from typing import Union, Tuple, Callable

from .metric import Metric
from ..math import spearmanr

@dataclass
class Spearmanr(Metric):
    """Implements :py:func:`mvpy.math.spearmanr` as a :py:class:`~mvpy.metrics.Metric`.
    
    .. warning::
        This class extends :py:class:`~mvpy.metrics.Metric`. If you
        would like to apply this metric, please use the instance 
        exposed under :py:attr:`mvpy.metrics.spearmanr`.
        
        For more information on this, please consult the documentation
        of :py:class:`~mvpy.metrics.Metric` and :py:func:`~mvpy.metrics.score`.
    
    Parameters
    ----------
    name : str, default='spearmanr'
        The name of this metric.
    request : str | tuple[str], default=('y', 'predict')
        The values to request for scoring.
    reduce : int | tuple[int], default= (0,)
        The dimension(s) to reduce over.
    f : Callable, default=mvpy.math.spearmanr
        The function to call.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.dataset import make_meeg_categorical
    >>> from mvpy.estimators import RidgeDecoder
    >>> from mvpy.crossvalidation import cross_val_score
    >>> from mvpy.metric import spearmanr
    >>> X, y = make_meeg_continuous()
    >>> clf = RidgeClassifier()
    >>> cross_val_score(clf, X, y, metric = spearmanr)
    """
    
    name: str = 'spearmanr'
    request: Union[str, Tuple[str]] = ('y', 'predict')
    reduce: Union[int, Tuple[int]] = (0,)
    f: Callable = spearmanr

# expose metric
spearmanr = Spearmanr()