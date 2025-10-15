import torch
import numpy as np

from dataclasses import dataclass, replace
from typing import Union, Tuple, Callable

from .metric import Metric
from ..math import accuracy

@dataclass
class Accuracy(Metric):
    """Implements :py:func:`mvpy.math.accuracy` as a :py:class:`~mvpy.metrics.Metric`.
    
    .. warning::
        This class extends :py:class:`~mvpy.metrics.Metric`. If you
        would like to apply this metric, please use the instance 
        exposed under :py:attr:`mvpy.metrics.accuracy`.
        
        For more information on this, please consult the documentation
        of :py:class:`~mvpy.metrics.Metric` and :py:func:`~mvpy.metrics.score`.
    
    Parameters
    ----------
    name : str, default='accuracy'
        The name of this metric.
    request : str | tuple[str], default=('y', 'predict')
        The values to request for scoring.
    reduce : int | tuple[int], default= (0,)
        The dimension(s) to reduce over.
    f : Callable, default=mvpy.math.accuracy
        The function to call.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.dataset import make_meeg_categorical
    >>> from mvpy.estimators import RidgeClassifier
    >>> from mvpy.crossvalidation import cross_val_score
    >>> from mvpy.metric import accuracy
    >>> X, y = make_meeg_categorical()
    >>> clf = RidgeClassifier()
    >>> cross_val_score(clf, X, y, metric = accuracy)
    """
    
    name: str = 'accuracy'
    request: Union[str, Tuple[str]] = ('y', 'predict')
    reduce: Union[int, Tuple[int]] = (0,)
    f: Callable = accuracy

# expose metric
accuracy = Accuracy()