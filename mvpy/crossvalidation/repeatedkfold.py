'''
A collection of classes for repeated k-fold cross-validation.
'''

import torch
import numpy as np

from .kfold import _KFold_numpy, _KFold_torch

from typing import Optional, Union
from collections.abc import Generator

class _RepeatedKFold_numpy:
    """Implements repeated k-folds in numpy backend.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits to use.
    n_repeats : int, default=1
        Number of repeats to use.
    random_state : Optional[Union[int, np.random._generator.Generator]], default=None
        Random state to use for shuffling (either integer seed or numpy generator), if any.
    
    Attributes
    ----------
    n_splits : int, default=5
        Number of splits to use.
    n_repeats : int, default=1
        Number of repeats to use.
    random_state : Optional[Union[int, np.random._generator.Generator]], default=None
        Random state to use for shuffling (either integer seed or numpy generator), if any.
    kfold_ : _KFold_numpy
        The underlying k-fold class.
    """
    
    def __init__(self, n_splits: int = 5, n_repeats: int = 1, random_state: Optional[Union[int, np.random._generator.Generator]] = None):
        """Obtain a new repeated k-fold splitter.
        
        Parameters
        ----------
        n_splits : int, default=5
            Number of splits to use.
        n_repeats : int, default=1
            Number of repeats to use.
        random_state : Optional[Union[int, np.random._generator.Generator]], default=None
            Random state to use for shuffling (either integer seed or numpy generator), if any.
        """
        
        # setup opts
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        
        # setup internals
        self.kfold_ = _KFold_numpy(
            n_splits = n_splits, 
            shuffle = True, 
            random_state = random_state
        )
    
    def __repr__(self) -> str:
        """String representation of the class.
        
        Returns
        -------
        repr : str
            String representation describing the class.
        """
        
        return f'RepeatedKFold(n_splits={self.n_splits}, n_repeats={self.n_repeats}, random_state={self.random_state})'
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Repeatedly split the dataset into iterable (train, test).
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, ...)
        y : Optional[np.ndarray], default=None
            Target data of shape (n_samples, ...). Unused, but parameter available for consistency.
        
        Returns
        -------
        kf : collections.abc.Generator[tuple[np.ndarray, np.ndarray], None, None]
            Iterable generator of (train, test) pairs.
        """
        
        # loop over repeats
        for i in range(self.n_repeats):
            # loop over folds
            for j, (train, test) in enumerate(self.kfold_.split(X, y)):
                # yield data
                yield (train, test)

class _RepeatedKFold_torch:
    """Implements repeated k-folds in torch backend.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits to use.
    n_repeats : int, default=1
        Number of repeats to use.
    random_state : Optional[Union[int, torch._C.Generator]], default=None
        Random state to use for shuffling (either integer seed or torch generator), if any.
    
    Attributes
    ----------
    n_splits : int, default=5
        Number of splits to use.
    n_repeats : int, default=1
        Number of repeats to use.
    random_state : Optional[Union[int, torch._C.Generator]], default=None
        Random state to use for shuffling (either integer seed or torch generator), if any.
    kfold_ : _KFold_torch
        The underlying k-fold class.
    """
    
    def __init__(self, n_splits: int = 5, n_repeats: int = 1, random_state: Optional[Union[int, torch._C.Generator]] = None):
        """Obtain a new k-fold splitter.
        
        Parameters
        ----------
        n_splits : int, default=5
            Number of splits to use.
        n_repeats : int, default=1
            Number of repeats to use.
        random_state : Optional[Union[int, torch._C.Generator]], default=None
            Random state to use for shuffling (either integer seed or torch generator), if any.
        """
        
        # setup opts
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        
        # setup internals
        self.kfold_ = _KFold_torch(
            n_splits = n_splits, 
            shuffle = True, 
            random_state = random_state
        )
    
    def __repr__(self) -> str:
        """String representation of the class.
        
        Returns
        -------
        repr : str
            String representation describing the class.
        """
        
        return f'RepeatedKFold(n_splits={self.n_splits}, n_repeats={self.n_repeats}, random_state={self.random_state})'
    
    def split(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Repeatedly split the dataset into iterable (train, test).
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, ...)
        y : Optional[torch.Tensor], default=None
            Target data of shape (n_samples, ...). Unused, but parameter available for consistency.
        
        Returns
        -------
        kf : collections.abc.Generator[tuple[torch.Tensor, torch.Tensor], None, None]
            Iterable generator of (train, test) pairs.
        """
        
        # loop over repeats
        for i in range(self.n_repeats):
            # loop over folds
            for j, (train, test) in enumerate(self.kfold_.split(X, y)):
                # yield data
                yield (train, test)

class RepeatedKFold:
    """Implements a repeated k-folds cross-validator.
    
    In principle, this class is redundant with sklearn.model_selection.RepeatedKFold. However, for the torch backend, this class is useful because it automatically creates indices on the desired device.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits to use.
    n_repeats : int, default=1
        Number of repeats to use.
    random_state : Optional[Union[int, np.random._generator.Generator, torch._C.Generator]], default=None
        Random state to use for shuffling (either integer seed or numpy/torch generator), if any.
    
    Attributes
    ----------
    n_splits : int, default=5
        Number of splits to use.
    n_repeats : int, default=1
        Number of repeats to use.
    random_state : Optional[Union[int, np.random._generator.Generator, torch._C.Generator]], default=None
        Random state to use for shuffling (either integer seed or numpy/torch generator), if any.
    
    Notes
    -----
    For reproducability when using shuffling, you can set the random_state to an integer.
    
    Note also that, when using shuffling, please make sure to instantiate and transform immediately to the backend you would like. Otherwise, each call to split will instantiate a new object with the same random seed.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.crossvalidation import RepeatedKFold
    >>> X = torch.arange(6)
    >>> kf = RepeatedKFold(n_splits = 2, n_repeats = 2).to_torch()
    >>> for f_i, (train, test) in enumerate(kf.split(X)):
    >>>     print(f'Fold{f_i}: train={train}\ttest={test}')
    Fold0: train=tensor([4, 1, 2])	test=tensor([0, 5, 3])
    Fold1: train=tensor([0, 5, 3])	test=tensor([4, 1, 2])
    Fold2: train=tensor([2, 1, 3])	test=tensor([5, 0, 4])
    Fold3: train=tensor([5, 0, 4])	test=tensor([2, 1, 3])
    """
    
    def __init__(self, n_splits: int = 5, n_repeats: int = 1, random_state: Optional[Union[int, np.random._generator.Generator, torch._C.Generator]] = None):
        """Obtain a new repeated k-fold splitter.
        
        Parameters
        ----------
        n_splits : int, default=5
            Number of splits to use.
        n_repeats : int, default=1
            Number of repeats to use.
        random_state : Optional[Union[int, np.random._generator.Generator, torch._C.Generator]], default=None
            Random state to use for shuffling (either integer seed or numpy/torch generator), if any.
        """
        
        # setup opts
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
    
    def __repr__(self) -> str:
        """String representation of the class.
        
        Returns
        -------
        repr : str
            String representation describing the class.
        """
        
        return f'RepeatedKFold(n_splits={self.n_splits}, n_repeats={self.n_repeats}, random_state={self.random_state})'
    
    def split(self, X: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Union[Generator[tuple[np.ndarray, np.ndarray], None, None], Generator[tuple[torch.Tensor, torch.Tensor], None, None]]:
        """Repeatedly split the dataset into iterable (train, test).
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Input data of shape (n_samples, ...)
        y : Optional[Union[np.ndarray, torch.Tensor]], default=None
            Target data of shape (n_samples, ...). Unused, but parameter available for consistency.
        
        Returns
        -------
        kf : Union[collections.abc.Generator[tuple[np.ndarray, np.ndarray], None, None], collections.abc.Generator[tuple[torch.Tensor, torch.Tensor], None, None]]
            Iterable generator of (train, test) pairs.
        """
        
        # determine type
        if isinstance(X, torch.Tensor) & ((y is None) or isinstance(y, torch.Tensor)):
            return _RepeatedKFold_torch(
                n_splits = self.n_splits,
                n_repeats = self.n_repeats,
                random_state = self.random_state
            ).split(X, y)
        elif isinstance(X, np.ndarray) & ((y is None) or isinstance(y, np.ndarray)):
            return _RepeatedKFold_numpy(
                n_splits = self.n_splits,
                n_repeats = self.n_repeats,
                random_state = self.random_state
            ).split(X, y)

        raise ValueError(f'If both X and y are supplied, they must both be of type np.ndarray or torch.Tensor, but got {type(X)} and {type(y)}.')

    def to_torch(self) -> "_RepeatedKFold_torch":
        """Convert class to torch backend.
        
        Returns
        -------
        kf : _RepeatedKFold_torch
            The repeated k-fold cross-validator in torch.
        """
        
        return _RepeatedKFold_torch(
            n_splits = self.n_splits,
            n_repeats = self.n_repeats,
            random_state = self.random_state
        )
    
    def to_numpy(self) -> "_RepeatedKFold_numpy":
        """Convert class to torch backend.
        
        Returns
        -------
        kf : _RepeatedKFold_numpy
            The repeated k-fold cross-validator in numpy.
        """
        
        return _RepeatedKFold_numpy(
            n_splits = self.n_splits,
            n_repeats = self.n_repeats,
            random_state = self.random_state
        )