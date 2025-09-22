'''
A collection of classes for k-fold cross-validation.
'''

import torch
import numpy as np

from typing import Optional, Union
from collections.abc import Generator

class _KFold_numpy:
    """Implements k-folds in numpy backend.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits to use.
    shuffle : bool, default=False
        Should we shuffle indices before splitting?
    random_state : Optional[Union[int, np.random._generator.Generator]], default=None
        Random state to use for shuffling (either integer seed or numpy generator), if any.
    
    Attributes
    ----------
    n_splits : int, default=5
        Number of splits to use.
    shuffle : bool, default=False
        Should we shuffle indices before splitting?
    random_state : Optional[Union[int, np.random._generator.Generator]], default=None
        Random state to use for shuffling (either integer seed or numpy generator), if any.
    rng_ : np.random._generator.Generator
        Random generator derived from random_state.
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[Union[int, np.random._generator.Generator]] = None):
        """Obtain a new k-fold splitter.
        
        Parameters
        ----------
        n_splits : int, default=5
            Number of splits to use.
        shuffle : bool, default=False
            Should we shuffle indices before splitting?
        random_state : Optional[Union[int, np.random._generator.Generator]], default=None
            Random state to use for shuffling (either integer seed or numpy generator), if any.
        """
        
        # setup opts
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        # setup internals
        if isinstance(self.random_state, int) or self.random_state is None:
            if self.random_state is not None:
                self.rng_ = np.random.default_rng(self.random_state)
            else:
                self.rng_ = np.random.default_rng()
        else:
            self.rng_ = self.random_state
    
    def __repr__(self) -> str:
        """String representation of the class.
        
        Returns
        -------
        repr : str
            String representation describing the class.
        """
        
        return f'KFold(n_splits={self.n_splits}, random_state={self.random_state}, shuffle={self.shuffle})'
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Split the dataset into iterable (train, test).
        
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
        
        # check samples
        n_samples = X.shape[0]
        
        if n_samples < self.n_splits:
            raise ValueError(f'n_samples must be greater than n_splits, but got {n_samples} and {self.n_splits}.')
        
        # if required, draw random indices
        if self.shuffle:
            indc = self.rng_.permutation(n_samples)
        else:
            indc = np.arange(n_samples)
        
        # setup sizes
        n_size = np.full(
            (self.n_splits,), n_samples // self.n_splits
        )
        n_size[:n_samples % self.n_splits] += 1
        
        # setup masks
        mask = np.zeros((n_samples,)).astype(bool)
        
        # setup loop
        f_i = 0
        
        for size in n_size:
            # test indices
            s, e = f_i, f_i + size
            
            # make mask
            mask[:] = False
            mask[s:e] = True
            
            # yield data
            yield indc[~mask], indc[mask]
            
            # update
            f_i = e

class _KFold_torch:
    """Implements k-folds in torch backend.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits to use.
    shuffle : bool, default=False
        Should we shuffle indices before splitting?
    random_state : Optional[Union[int, torch._C.Generator]], default=None
        Random state to use for shuffling (either integer seed or torch generator), if any.
    
    Attributes
    ----------
    n_splits : int, default=5
        Number of splits to use.
    shuffle : bool, default=False
        Should we shuffle indices before splitting?
    random_state : Optional[Union[int, torch._C.Generator]], default=None
        Random state to use for shuffling (either integer seed or torch generator), if any.
    rng_ : torch._C.Generator
        Random generator derived from random_state.
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[Union[int, torch._C.Generator]] = None):
        """Obtain a new k-fold splitter.
        
        Parameters
        ----------
        n_splits : int, default=5
            Number of splits to use.
        shuffle : bool, default=False
            Should we shuffle indices before splitting?
        random_state : Optional[Union[int, torch._C.Generator]], default=None
            Random state to use for shuffling (either integer seed or torch generator), if any.
        """
        
        # setup opts
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        # setup internals
        if isinstance(self.random_state, int) or self.random_state is None:
            self.rng_ = torch.Generator()
            
            if self.random_state is not None:
                self.rng_.manual_seed(self.random_state)
        else:
            self.rng_ = self.random_state
    
    def __repr__(self) -> str:
        """String representation of the class.
        
        Returns
        -------
        repr : str
            String representation describing the class.
        """
        
        return f'KFold(n_splits={self.n_splits}, random_state={self.random_state}, shuffle={self.shuffle})'
    
    def split(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Split the dataset into iterable (train, test).
        
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
        
        # check samples
        n_samples = X.shape[0]
        
        if n_samples < self.n_splits:
            raise ValueError(f'n_samples must be greater than n_splits, but got {n_samples} and {self.n_splits}.')
        
        # if required, draw random indices
        if self.shuffle:
            # check device of generator
            if self.rng_.device != X.device:
                self.rng_ = torch.Generator(device = X.device)
                
                if self.random_state is not None:
                    self.rng_.manual_seed(self.random_state)
            
            indc = torch.randperm(n_samples, generator = self.rng_, device = X.device).long()
        else:
            indc = torch.arange(n_samples, device = X.device).long()
        
        # setup sizes
        n_size = torch.full(
            (self.n_splits,), n_samples // self.n_splits,
            dtype = torch.long, device = X.device
        )
        n_size[:n_samples % self.n_splits] += 1
        
        # setup masks
        mask = torch.zeros(n_samples, dtype = torch.bool, device = X.device)
        
        # setup loop
        f_i = 0
        
        for size in n_size:
            # test indices
            s, e = f_i, f_i + size
            
            # make mask
            mask[:] = False
            mask[s:e] = True
            
            # yield data
            yield indc[~mask], indc[mask]
            
            # update
            f_i = e

class KFold:
    """Implements a k-folds cross-validator.
    
    In principle, this class is redundant with sklearn.model_selection.KFold. However, for the torch backend, this class is useful because it automatically creates indices on the desired device.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits to use.
    shuffle : bool, default=False
        Should we shuffle indices before splitting?
    random_state : Optional[Union[int, np.random._generator.Generator, torch._C.Generator]], default=None
        Random state to use for shuffling (either integer seed or numpy/torch generator), if any.
    
    Attributes
    ----------
    n_splits : int, default=5
        Number of splits to use.
    shuffle : bool, default=False
        Should we shuffle indices before splitting?
    random_state : Optional[Union[int, np.random._generator.Generator, torch._C.Generator]], default=None
        Random state to use for shuffling (either integer seed or numpy/torch generator), if any.
    rng_ : Union[np.random._generator.Generator, torch._C.Generator]
        Random generator derived from random_state.
    
    Notes
    -----
    For reproducability when using shuffling, you can set the random_state to an integer.
    
    Note also that, when using shuffling, please make sure to instantiate and transform immediately to the backend you would like. Otherwise, each call to split will instantiate a new object with the same random seed. See examples for a demonstration.
    
    Examples
    --------
    If we are not using shuffling, we can simply do:
    
    >>> import torch
    >>> from mvpy.crossvalidation import KFold
    >>> X = torch.arange(10)
    >>> kf = KFold()
    >>> for f_i, (train, test) in enumerate(kf.split(X)):
    >>>     print(f'Fold{f_i}: train={train}\ttest={test}')
    Fold0: train=tensor([2, 3, 4, 5, 6, 7, 8, 9])	test=tensor([0, 1])
    Fold1: train=tensor([0, 1, 4, 5, 6, 7, 8, 9])	test=tensor([2, 3])
    Fold2: train=tensor([0, 1, 2, 3, 6, 7, 8, 9])	test=tensor([4, 5])
    Fold3: train=tensor([0, 1, 2, 3, 4, 5, 8, 9])	test=tensor([6, 7])
    Fold4: train=tensor([0, 1, 2, 3, 4, 5, 6, 7])	test=tensor([8, 9])
    
    However, let's assume we want to use shuffling. We might be inclined to do:
    
    >>> import torch
    >>> from mvpy.crossvalidation import KFold
    >>> X = torch.arange(6)
    >>> kf = KFold(n_splits = 2, shuffle = True, random_state = 42)
    >>> print(f'Run 1:')
    >>> for f_i, (train, test) in enumerate(kf.split(X)):
    >>>     print(f'Fold{f_i}: train={train}\ttest={test}')
    >>> print(f'Run 2:')
    >>> for f_i, (train, test) in enumerate(kf.split(X)):
    >>>     print(f'Fold{f_i}: train={train}\ttest={test}')
    Run 1:
    Fold0: train=tensor([4, 1, 5])	test=tensor([0, 3, 2])
    Fold1: train=tensor([0, 3, 2])	test=tensor([4, 1, 5])
    Run 2:
    Fold0: train=tensor([4, 1, 5])	test=tensor([0, 3, 2])
    Fold1: train=tensor([0, 3, 2])	test=tensor([4, 1, 5])
    
    Note that here we pass random_state to make this reproducible on your end. As you can see, the randomisation is now static across runs. This occurs because, up until the call to split the data, MVPy cannot consistently infer the desired data type. Therefore, the backend class is instantiated only upon calling split where types become explicit. However, this means that each call to split will re-instantiate the class. We can easily work around this in two ways:
    
    >>> import torch
    >>> from mvpy.crossvalidation import KFold
    >>> X = torch.arange(6)
    >>> kf = KFold(n_splits = 2, shuffle = True, random_state = 42).to_torch()
    >>> print(f'Run 1:')
    >>> for f_i, (train, test) in enumerate(kf.split(X)):
    >>>     print(f'Fold{f_i}: train={train}\ttest={test}')
    >>> print(f'Run 2:')
    >>> for f_i, (train, test) in enumerate(kf.split(X)):
    >>>     print(f'Fold{f_i}: train={train}\ttest={test}')
    Run 1:
    Fold0: train=tensor([4, 1, 5])	test=tensor([0, 3, 2])
    Fold1: train=tensor([0, 3, 2])	test=tensor([4, 1, 5])
    Run 2:
    Fold0: train=tensor([4, 0, 3])	test=tensor([5, 1, 2])
    Fold1: train=tensor([5, 1, 2])	test=tensor([4, 0, 3])
    
    Here, we explicitly instantiate a torch operator that is not reinstantiated across runs, which works perfectly. We could, however, also use an external generator to achieve the same result:
    
    >>> import torch
    >>> from mvpy.crossvalidation import KFold
    >>> X = torch.arange(6)
    >>> rng = torch.Generator()
    >>> rng.manual_seed(42)
    >>> kf = KFold(n_splits = 2, shuffle = True, random_state = rng)
    >>> print('Run 1:')
    >>> for f_i, (train, test) in enumerate(kf.split(X)):
    >>>     print(f'Fold{f_i}: train={train}\ttest={test}')
    >>> print('Run 2:')
    >>> for f_i, (train, test) in enumerate(kf.split(X)):
    >>>     print(f'Fold{f_i}: train={train}\ttest={test}')
    Run 1:
    Fold0: train=tensor([4, 1, 5])	test=tensor([0, 3, 2])
    Fold1: train=tensor([0, 3, 2])	test=tensor([4, 1, 5])
    Run 2:
    Fold0: train=tensor([4, 0, 3])	test=tensor([5, 1, 2])
    Fold1: train=tensor([5, 1, 2])	test=tensor([4, 0, 3])
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[Union[int, np.random._generator.Generator, torch._C.Generator]] = None):
        """Obtain a new k-fold splitter.
        
        Parameters
        ----------
        n_splits : int, default=5
            Number of splits to use.
        shuffle : bool, default=False
            Should we shuffle indices before splitting?
        random_state : Optional[Union[int, np.random._generator.Generator, torch._C.Generator]], default=None
            Random state to use for shuffling (either integer seed or numpy/torch generator), if any.
        """
        
        # setup opts
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def __repr__(self) -> str:
        """String representation of the class.
        
        Returns
        -------
        repr : str
            String representation describing the class.
        """
        
        return f'KFold(n_splits={self.n_splits}, random_state={self.random_state}, shuffle={self.shuffle})'
    
    def split(self, X: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Union[Generator[tuple[np.ndarray, np.ndarray], None, None], Generator[tuple[torch.Tensor, torch.Tensor], None, None]]:
        """Split the dataset into iterable (train, test).
        
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
            return _KFold_torch(
                n_splits = self.n_splits,
                shuffle = self.shuffle,
                random_state = self.random_state
            ).split(X, y)
        elif isinstance(X, np.ndarray) & ((y is None) or isinstance(y, np.ndarray)):
            return _KFold_numpy(
                n_splits = self.n_splits,
                shuffle = self.shuffle,
                random_state = self.random_state
            ).split(X, y)

        raise ValueError(f'If both X and y are supplied, they must both be of type np.ndarray or torch.Tensor, but got {type(X)} and {type(y)}.')

    def to_torch(self) -> "_KFold_torch":
        """Convert class to torch backend.
        
        Returns
        -------
        kf : _KFold_torch
            The k-fold cross-validator in torch.
        """
        
        return _KFold_torch(
            n_splits = self.n_splits,
            shuffle = self.shuffle,
            random_state = self.random_state
        )
    
    def to_numpy(self) -> "_KFold_numpy":
        """Convert class to torch backend.
        
        Returns
        -------
        kf : _KFold_numpy
            The k-fold cross-validator in numpy.
        """
        
        return _KFold_numpy(
            n_splits = self.n_splits,
            shuffle = self.shuffle,
            random_state = self.random_state
        )