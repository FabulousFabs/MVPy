import torch
import numpy as np

from dataclasses import dataclass, replace
from typing import Union, Tuple, Callable

from .metric import Metric
from ..math import roc_auc

@dataclass
class Roc_auc(Metric):
    """Implements :py:func:`mvpy.math.roc_auc` as a :py:class:`~mvpy.metrics.Metric`.
    
    .. warning::
        This class extends :py:class:`~mvpy.metrics.Metric`. If you
        would like to apply this metric, please use the instance 
        exposed under :py:attr:`mvpy.metrics.roc_auc`.
        
        For more information on this, please consult the documentation
        of :py:class:`~mvpy.metrics.Metric` and :py:func:`~mvpy.metrics.score`.
    
    Parameters
    ----------
    name : str, default='roc_auc'
        The name of this metric.
    request : str | tuple[str], default=('y', 'decision_function')
        The values to request for scoring.
    reduce : int | tuple[int], default= (0,)
        The dimension(s) to reduce over.
    f : Callable, default=mvpy.math.roc_auc
        The function to call.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.dataset import make_meeg_categorical
    >>> from mvpy.estimators import RidgeClassifier
    >>> from mvpy.crossvalidation import cross_val_score
    >>> from mvpy.metric import roc_auc
    >>> X, y = make_meeg_categorical()
    >>> clf = RidgeClassifier()
    >>> cross_val_score(clf, X, y, metric = roc_auc)
    """
    
    name: str = 'roc_auc'
    request: Union[str, Tuple[str]] = ('y', 'decision_function')
    reduce: Union[int, Tuple[int]] = (0,)
    f: Callable = roc_auc
    
    def __call__(self, y: Union[np.ndarray, torch.Tensor], df: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Compute ROC-AUC scores.
        
        This overwrites the default behaviour specified in :py:class:`~mvpy.metrics.Metric`
        to make sure ROC-AUC scores are computed appropriately per feature, even when there
        is an additional time dimension.
        
        Parameters
        ----------
        y : np.ndarray | torch.Tensor
            The true labels of shape ``(n_features, [n_timepoints, ]n_samples)``.
        df : np.ndarray | torch.Tensor
            The decision functions of shape ``(n_classes, [n_timepoints, ]n_samples)``.
        
        Returns
        -------
        roc_auc : np.ndarray | torch.Tensor
            The score of shape ``(n_features[, n_timepoints])``.
        """
        
        # check number of dimensions
        nd = y.ndim
        
        # check backend
        is_torch = isinstance(y, torch.Tensor) & isinstance(df, torch.Tensor)
                
        # check if we need to re-order
        if nd > 2:
            # in this case, we now also have a time dimension to worry about
            # currently, it will be ([...,] n_features, n_timepoints, n_samples)
            y = y.swapaxes(-2, -3)
            df = df.swapaxes(-2, -3)
                
        # what we get from classifiers is of shape ([..., ]n_features, n_samples)
        # so first, check how many classes there are per feature
        if is_torch:
            features = [torch.unique(y[...,i,:]).shape[0] for i in range(y.shape[-2])]
        else:
            features = [np.unique(y[...,i,:]).shape[0] for i in range(y.shape[-2])]
        
        # next, make start of features
        offsets = [0] + features[:-1]
        
        # score feature-wise
        score = [
            self.f(y[...,i,:], df[...,offsets[i]:offsets[i]+features[i],:])
            for i in range(len(features))
        ]
        
        # stack scores
        if is_torch:
            score = torch.stack(score, dim = 0)
        else:
            score = np.array(score)
        
        return score
        
# expose metric
roc_auc = Roc_auc()