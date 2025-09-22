'''
A collection of classes for stratified k-fold cross-validation.
'''

import torch
import numpy as np

from ..estimators import LabelBinariser

from typing import Optional, Union
from collections.abc import Generator

class _StratifiedKFold_numpy:
    """Implements stratified k-folds in numpy backend.
    
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
        """Obtain a new stratified k-fold splitter.
        
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
        
        return f'StratifiedKFold(n_splits={self.n_splits}, random_state={self.random_state}, shuffle={self.shuffle})'
    
    def split(self, X: np.ndarray, y: np.ndarray) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Split the dataset into stratified iterable (train, test).
        
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
        
        # check y
        if len(y.shape) == 1:
            y = y[:,None]
        
        # check high y dims
        if len(y.shape) > 2:
            # select only (n_samples, ...., n_features, n_timepoints)
            dims = [0] * len(y.shape)
            dims[0] = slice(None)
            dims[-2] = slice(None)
            y = y[tuple(dims)]
        
        # create binary labels
        L = LabelBinariser(
            neg_label = 0,
            pos_label = 1
        ).to_numpy().fit_transform(y)
        
        # make unique through bit representation
        R = L.reshape(L.shape[0], -1)
        w = (2 ** np.arange(R.shape[1] - 1, -1, -1))[None,:]
        R = (R * w).sum(axis = 1).astype(int)
        
        # grab indicies and inversion
        _, R_idx, R_inv = np.unique(R, return_index = True, return_inverse = True)
        
        # encode by order appearance
        _, class_perm = np.unique(R_idx, return_inverse = True)
        R_encoded = class_perm[R_inv]
        
        # check grouping
        n_classes = len(R_idx)
        R_counts = np.bincount(R_encoded)
        min_groups = np.min(R_counts)
        if np.all(self.n_splits > R_counts):
            raise ValueError(f'`n_splits` must be smaller or equal to the number of members in each class.')
        
        # round robin over sorted to determine number of samples
        R_order = np.sort(R_encoded)
        allocation = np.asarray(
            [np.bincount(R_order[i::self.n_splits], minlength = n_classes) for i in range(self.n_splits)]
        )
        
        # make test folds
        test_folds = np.zeros((R.shape[0],)).astype(int)
        for k in range(n_classes):
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:,k])
            
            if self.shuffle:
                self.rng_.shuffle(folds_for_class)
            
            test_folds[R_encoded == k] = folds_for_class
        
        # finally, yield data
        indc = np.arange(n_samples)
        for i in range(self.n_splits):
            test = (test_folds == i)
            train = ~test
            
            yield indc[train], indc[test]

class _StratifiedKFold_torch:
    """Implements stratified k-folds in torch backend.
    
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
        """Obtain a new stratified k-fold splitter.
        
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
        
        return f'StratifiedKFold(n_splits={self.n_splits}, random_state={self.random_state}, shuffle={self.shuffle})'
    
    def split(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Split the dataset into stratified iterable (train, test).
        
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
        
        # check y
        if len(y.shape) == 1:
            y = y[:,None]
        
        # check high y dims
        if len(y.shape) > 2:
            # select only (n_samples, ...., n_features, n_timepoints)
            dims = [0] * len(y.shape)
            dims[0] = slice(None)
            dims[-2] = slice(None)
            y = y[tuple(dims)]
        
        # create binary labels
        L = LabelBinariser(
            neg_label = 0,
            pos_label = 1
        ).to_torch().fit_transform(y)
        
        # make unique through bit representation
        R = L.reshape(L.shape[0], -1).float()
        w = (2 ** torch.arange(R.shape[1] - 1, -1, -1, device = X.device))[None,:].float()
        R = (R * w).sum(1).long()
        
        # grab indicies and inversion
        R_unq, R_inv = torch.unique(R, return_inverse = True)
        R_idx = torch.tensor([torch.where(R == R_i)[0].min() for R_i in R_unq], device = X.device)
        
        # encode by order appearance
        _, class_perm = torch.unique(R_idx, return_inverse = True)
        R_encoded = class_perm[R_inv]
        
        # check grouping
        n_classes = len(R_idx)
        R_counts = torch.bincount(R_encoded)
        min_groups = torch.min(R_counts)
        if torch.all(self.n_splits > R_counts):
            raise ValueError(f'`n_splits` must be smaller or equal to the number of members in each class.')
        
        # round robin over sorted to determine number of samples
        R_order, _ = torch.sort(R_encoded, dim = 0)
        allocation = torch.stack(
            [torch.bincount(R_order[i::self.n_splits], minlength = n_classes) for i in range(self.n_splits)],
        ).long()
        
        # make test folds
        test_folds = torch.zeros((R.shape[0],), device = X.device).long()
        for k in range(n_classes):
            folds_for_class = [torch.tensor([i] * allocation[i,k], device = X.device) for i in range(self.n_splits)]
            folds_for_class = torch.cat(folds_for_class)
            
            if self.shuffle:
                if self.rng_.device != X.device:
                    self.rng_ = torch.Generator(device = X.device)
                    
                    if self.random_state is not None:
                        self.rng_.manual_seed(self.random_state)
                
                perm = torch.randperm(folds_for_class.shape[0], device = X.device).long()
                folds_for_class = folds_for_class[perm]
            
            test_folds[(R_encoded == k).bool()] = folds_for_class.long()
        
        # finally, yield data
        indc = torch.arange(n_samples, device = X.device)
        for i in range(self.n_splits):
            test = (test_folds == i)
            train = ~test
            
            yield indc[train], indc[test]

class StratifiedKFold:
    """Implements a stratified k-folds cross-validator.
    
    Unlike sklearn, this will also stratify across features of (n_samples[, ...], n_features[, n_timepoints]).
    
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
    First, let's assume we have just one feature:
    
    >>> import torch
    >>> from mvpy.crossvalidation import StratifiedKFold
    >>> X = torch.randn(75, 5)
    >>> y = torch.tensor([0] * 40 + [1] * 25 + [2] * 10)
    >>> kf = StratifiedKFold()
    >>> for f_i, (train, test) in enumerate(kf.split(X, y)):
    >>>     train_idx, train_cnt = torch.unique(y[train], return_counts = True)
    >>>     _, test_cnt = torch.unique(y[test], return_counts = True)
    >>>     print(f'Fold {f_i}: classes={train_idx}\tN(train)={train_cnt}\tN(test)={test_cnt}')
    Fold 0: classes=tensor([0, 1, 2])	N(train)=tensor([32, 20,  8])	N(test)=tensor([8, 5, 2])
    Fold 1: classes=tensor([0, 1, 2])	N(train)=tensor([32, 20,  8])	N(test)=tensor([8, 5, 2])
    Fold 2: classes=tensor([0, 1, 2])	N(train)=tensor([32, 20,  8])	N(test)=tensor([8, 5, 2])
    Fold 3: classes=tensor([0, 1, 2])	N(train)=tensor([32, 20,  8])	N(test)=tensor([8, 5, 2])
    Fold 4: classes=tensor([0, 1, 2])	N(train)=tensor([32, 20,  8])	N(test)=tensor([8, 5, 2])
    
    Second, let's assume we have multiple features and we want to shuffle indices. Note that this will also work if features have overlapping class names, but for clarity here we use different offsets:
    
    >>> import torch
    >>> from mvpy.crossvalidation import StratifiedKFold
    >>> X = torch.randn(75, 5)
    >>> y0 = torch.tensor([0] * 40 + [1] * 25 + [2] * 10)[:,None]
    >>> y1 = torch.tensor([3] * 15 + [4] * 45 + [5] * 15)[:,None]
    >>> y = torch.stack((y0, y1), dim = 1)
    >>> kf = StratifiedKFold(shuffle = True).to_torch()
    >>> for f_i, (train, test) in enumerate(kf.split(X, y)):
    >>>     train_idx, train_cnt = torch.unique(y[train], return_counts = True)
    >>>     _, test_cnt = torch.unique(y[test], return_counts = True)
    >>>     print(f'Fold {f_i}: classes={train_idx}\tN(train)={train_cnt}\tN(test)={test_cnt}')
    Fold 0: classes=tensor([0, 1, 2, 3, 4, 5])	N(train)=tensor([32, 20,  8, 12, 36, 12])	N(test)=tensor([8, 5, 2, 3, 9, 3])
    Fold 1: classes=tensor([0, 1, 2, 3, 4, 5])	N(train)=tensor([32, 20,  8, 12, 36, 12])	N(test)=tensor([8, 5, 2, 3, 9, 3])
    Fold 2: classes=tensor([0, 1, 2, 3, 4, 5])	N(train)=tensor([32, 20,  8, 12, 36, 12])	N(test)=tensor([8, 5, 2, 3, 9, 3])
    Fold 3: classes=tensor([0, 1, 2, 3, 4, 5])	N(train)=tensor([32, 20,  8, 12, 36, 12])	N(test)=tensor([8, 5, 2, 3, 9, 3])
    Fold 4: classes=tensor([0, 1, 2, 3, 4, 5])	N(train)=tensor([32, 20,  8, 12, 36, 12])	N(test)=tensor([8, 5, 2, 3, 9, 3])
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[Union[int, np.random._generator.Generator, torch._C.Generator]] = None):
        """Obtain a new stratified k-fold splitter.
        
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
        
        return f'StratifiedKFold(n_splits={self.n_splits}, random_state={self.random_state}, shuffle={self.shuffle})'
    
    def split(self, X: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Union[Generator[tuple[np.ndarray, np.ndarray], None, None], Generator[tuple[torch.Tensor, torch.Tensor], None, None]]:
        """Split the dataset into stratified iterable (train, test).
        
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
            return _StratifiedKFold_torch(
                n_splits = self.n_splits,
                shuffle = self.shuffle,
                random_state = self.random_state
            ).split(X, y)
        elif isinstance(X, np.ndarray) & ((y is None) or isinstance(y, np.ndarray)):
            return _StratifiedKFold_numpy(
                n_splits = self.n_splits,
                shuffle = self.shuffle,
                random_state = self.random_state
            ).split(X, y)

        raise ValueError(f'If both X and y are supplied, they must both be of type np.ndarray or torch.Tensor, but got {type(X)} and {type(y)}.')

    def to_torch(self) -> "_StratifiedKFold_torch":
        """Convert class to torch backend.
        
        Returns
        -------
        kf : _StratifiedKFold_torch
            The k-fold cross-validator in torch.
        """
        
        return _StratifiedKFold_torch(
            n_splits = self.n_splits,
            shuffle = self.shuffle,
            random_state = self.random_state
        )
    
    def to_numpy(self) -> "_StratifiedKFold_numpy":
        """Convert class to torch backend.
        
        Returns
        -------
        kf : _StratifiedKFold_numpy
            The k-fold cross-validator in numpy.
        """
        
        return _StratifiedKFold_numpy(
            n_splits = self.n_splits,
            shuffle = self.shuffle,
            random_state = self.random_state
        )