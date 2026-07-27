'''
A collection of classes for k-fold cross-validation.
'''

import torch
import numpy as np

from typing import Optional, Union, List
from collections.abc import Generator

from .repeatedkfold import RepeatedKFold

class LeaveGroupsOut:
    """Implements a leave-groups-out cross-validator.
    
    Functionally, this class is analogous to using :py:class:`~mvpy.crossvalidation.RepeatedKFold` 
    over unique values of the supplied :py:attr:`groups`. For example, for inputs :math:`X` of shape 
    ``(100, ...)`` and ``groups`` :math:`G` of shape ``(100, 1)`` where :math:`G\\in {1...5}` with :math:`k = 5`, 
    each group in :math:`G` corresponds to one fold :math:`k`. 
    
    However, :py:class:`~mvpy.crossvalidation.LeaveGroupsOut` also supports multi-label situations. For 
    example, consider a gram matrix :math:`A(X, X^T)\\in\\mathcal{R}^{n\\times n}` where :math:`A_{i,j}` 
    is the neural similarity between participants :math:`i` and :math:`j` that we wish to model as a function
    of a gram matrix :math:`B(y, y^T)\\in\\mathcal{R}^{n\\times n}` where :math:`B_{i,j}` describes behavioural 
    similarity of participants :math:`i` and :math:`j`. Typically, we would like to model :math:`u(A) = \\beta u(B) + \\varepsilon`
    where :math:`u` simply defines the upper triangle of the matrix. In cross-validation, however, this would 
    lead to leakage because of the dyadic relationship present in each sample :math:`A_{i,j}`. Consequently, we 
    must treat participants as grouping variables such that train and test sets are constructed over participants 
    rather than samples. In this case, we would construct :math:`G\\in\\mathcal{R}^{n\\times n\\times 2}` where 
    :math:`G_{i,j} = \\left(\\begin{array}{c} i \\ j \\end{array}\\right)`to ensure that we train on a subset of 
    participants and test on a separate subset of participants.
    
    .. warning::
        If multiple labels per sample are present in :py:attr:`~mvpy.crossvalidation.LeaveGroupsOut`'s `group` 
        parameter such that `(n_samples, ..., n_groups)` where `n_groups > 1`, make sure that data 
        are roughly balanced. Otherwise, fold sizes may vary greatly.
    
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
    
    def __init__(self, n_splits: int = 5, n_repeats: int = 1, random_state: Optional[Union[int, np.random._generator.Generator, torch._C.Generator]] = None):
        """Obtain a new k-fold splitter.
        
        Parameters
        ----------
        n_splits : int, default=5
            Number of splits to use.
        n_repeats : int, default=1
            Number of repeats to perform.
        random_state : Optional[Union[int, np.random._generator.Generator, torch._C.Generator]], default=None
            Random state to use for shuffling (either integer seed or numpy/torch generator), if any.
        """
        
        # setup opts
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        
        # setup attributes
        self.rkf_ = RepeatedKFold(
            n_splits = self.n_splits,
            n_repeats = self.n_repeats,
            random_state = self.random_state
        )
    
    def __repr__(self) -> str:
        """String representation of the class.
        
        Returns
        -------
        repr : str
            String representation describing the class.
        """
        
        return f'LeaveGroupsOut(n_splits={self.n_splits}, random_state={self.random_state}, shuffle={self.shuffle})'
    
    def split(self, X: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]] = None, groups: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Union[Generator[tuple[np.ndarray, np.ndarray], None, None], Generator[tuple[torch.Tensor, torch.Tensor], None, None]]:
        """Split the dataset into iterable (train, test).
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Input data of shape (n_samples, ...)
        y : np.ndarray | torch.Tensor | None, default=None
            Target data of shape (n_samples, ...). Unused, but parameter available for consistency.
        groups : np.ndarray | torch.Tensor | None, default=None
            Group labels of data. Labels are of shape (n_samples, ..., n_labels). One sample may have multiple labels.
        
        Returns
        -------
        kf : Union[collections.abc.Generator[tuple[np.ndarray, np.ndarray], None, None], collections.abc.Generator[tuple[torch.Tensor, torch.Tensor], None, None]]
            Iterable generator of (train, test) pairs.
        """
        
        # check groups
        if groups is None:
            raise ValueError(
                f'`groups` must be specified when using LeaveGroupsOut.'
            )
        
        # check shape
        if groups.shape[0] != X.shape[0]:
            raise ValueError(
                f'Number of samples must be equal across X and groups, but got ' + 
                f'{X.shape[0]} and {groups.shape[0]}, respectively.'
            )
        
        # check data types
        if isinstance(X, torch.Tensor) & isinstance(groups, torch.Tensor) & (isinstance(y, torch.Tensor) | (y is None)):
            is_torch = True
        elif isinstance(X, np.ndarray) & isinstance(groups, np.ndarray) & (isinstance(y, np.ndarray) | (y is None)):
            is_torch = False
        else:
            raise ValueError(
                f'`X`, `y` and `groups` must be supplied as the same datatype. Available options ' + 
                f'are either np.ndarray or torch.Tensor, but got {type(X)}, {type(y)} and {type(groups)}.'
            )
        
        if is_torch:
            # obtain unique groups
            unq_groups = torch.unique(groups)
            
            n_dims = len(groups.shape)
            dims = torch.arange(n_dims)
            if n_dims > 1:
                dims = dims[1:]
            
            # obtain splits over groups
            for i, (train, test) in enumerate(self.rkf_.split(unq_groups[:,None])):
                # grab labels
                train_i = unq_groups[train]
                test_i = unq_groups[test]
                
                # find labels
                train_j = torch.isin(groups, train_i)
                test_j = torch.isin(groups, test_i)
                
                if n_dims > 1:
                    train_j = train_j.all(dim = tuple(dims))
                    test_j = test_j.all(dim = tuple(dims))
                
                train_j = torch.where(train_j)[0]
                test_j = torch.where(test_j)[0]
                
                # yield final result
                yield (train_j, test_j)
        else:
            # obtain unique groups
            unq_groups = np.unique(groups)
            
            n_dims = len(groups.shape)
            dims = np.arange(n_dims)
            if n_dims > 1:
                dims = dims[1:]
            
            # obtain splits over groups
            for i, (train, test) in enumerate(self.rkf_.split(unq_groups[:,None])):
                # grab labels
                train_i = unq_groups[train]
                test_i = unq_groups[test]
                
                # find labels
                train_j = np.isin(groups, train_i)
                test_j = np.isin(groups, test_i)
                
                if n_dims > 1:
                    train_j = train_j.all(axis = tuple(dims))
                    test_j = test_j.all(axis = tuple(dims))
                
                train_j = np.where(train_j)[0]
                test_j = torch.where(test_j)[0]
                
                # yield final result
                yield (train_j, test_j)