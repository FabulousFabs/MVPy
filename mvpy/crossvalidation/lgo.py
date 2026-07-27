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
    :math:`G_{i,j} = \\left(i, j\\right)`to ensure that we train on a subset of 
    participants and test on a separate subset of participants.
    
    .. warning::
        If multiple labels per sample are present in :py:class:`~mvpy.crossvalidation.LeaveGroupsOut`'s ``group`` 
        parameter such that ``(n_samples, ..., n_groups)`` where ``n_groups > 1``, make sure that data 
        are roughly balanced. Otherwise, fold sizes may vary greatly.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits to use.
    n_repeats : int, default=1
        Number of repeats to perform.
    random_state : Optional[Union[int, np.random._generator.Generator, torch._C.Generator]], default=None
        Random state to use for shuffling (either integer seed or numpy/torch generator), if any.
    
    Attributes
    ----------
    n_splits : int, default=5
        Number of splits to use.
    n_repeats : int, default=1
        Number of repeats to perform.
    random_state : Optional[Union[int, np.random._generator.Generator, torch._C.Generator]], default=None
        Random state to use for shuffling (either integer seed or numpy/torch generator), if any.
    rkf_ : RepeatedKFold
        Repeated k-fold cross-validation object used under the hood.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.crossvalidation import LeaveGroupsOut
    >>> X = torch.arange(6)
    >>> g = torch.arange(2).repeat(3)
    >>> kf = LeaveGroupsOut(n_splits = 2, n_repeats = 1)
    >>> for f_i, (train, test) in enumerate(kf.split(X, groups = g)):
    >>>     print(f'Fold{f_i}: train={train}|{g[train]}    test={test}|{g[test]}')
    Fold0: train=tensor([1, 3, 5])|tensor([1, 1, 1])    test=tensor([0, 2, 4])|tensor([0, 0, 0])
    Fold1: train=tensor([0, 2, 4])|tensor([0, 0, 0])    test=tensor([1, 3, 5])|tensor([1, 1, 1])
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