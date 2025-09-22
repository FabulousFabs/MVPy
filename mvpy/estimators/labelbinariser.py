'''
A collection of estimators for binarising label data.
'''

import numpy as np
import torch
import sklearn

from typing import Union, Any

class _LabelBinariser_numpy(sklearn.base.BaseEstimator):
    r"""Class to create and handle multiclass and multifeature one-hot encodings using numpy backend.
    
    Parameters
    ----------
    neg_label : int, default=0
        Label to use for negatives.
    pos_label : int, default=1
        Label to use for positives.
    
    Attributes
    ----------
    neg_label : int, default=0
        Label to use for negatives.
    pos_label : int, default=1
        Label to use for positives.
    n_features_ : int
        Number of unique features in y of shape (n_samples, n_features).
    n_classes_ : List[int]
        Number of unique classes per feature.
    labels_ : List[List[Any]]
        List including lists of original labels in y.
    classes_ : List[List[Any]]
        List including lists of class identities in y.
    N_ : Union[int, np.ndarray]
        Total number of classes (across features).
    C_ : np.ndarray
        Offsets for each unique feature in one-hot matrix.
    map_L_to_C_ : List[Dict[Any, int]]
        Lists containing each label->class mapping per feature.
    """
    
    def __init__(self, neg_label: int = 0, pos_label: int = 1):
        """Obtain a binariser.
        
        Parameters
        ----------
        neg_label : int, default=0
            Label to use for negatives.
        pos_label : int, default=1
            Label to use for positives.
        """
        
        # check labels
        if pos_label <= neg_label:
            raise ValueError(f'Negative label must be smaller than positive label.')
        
        # setup opts
        self.neg_label = neg_label
        self.pos_label = pos_label
        
        # setup internals
        self.n_features_ = None
        self.n_classes_ = None
        self.labels_ = None
        self.classes_ = None
        self.N_ = None
        self.C_ = None
        self.map_L_to_C_ = None
    
    def fit(self, y: np.ndarray) -> "_LabelBinariser_numpy":
        """Fit the binariser.

        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            The data of shape (n_samples[, n_features]).
        args : Any
            Additional arguments.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The binariser.
        """
        
        # check shape of y
        if len(y.shape) == 1:
            y = y[:, None]
        
        # grab unique labels
        self.n_features_ = y.shape[1]
        self.n_classes_ = [np.unique(y[:,i]).shape[0] for i in range(self.n_features_)]
        self.labels_ = [np.unique(y[:,i]) for i in range(self.n_features_)]
        self.classes_ = [np.arange(len(self.labels_[i])) for i in range(self.n_features_)]
        
        # setup N and cumsums
        self.N_ = np.sum(self.n_classes_)
        self.C_ = np.cumsum(self.n_classes_)
        self.C_ = np.concatenate((np.array([0]), self.C_[:-1]))
        
        # setup mapping
        self.map_L_to_C_ = [{label_: class_ for class_, label_ in zip(self.classes_[i], self.labels_[i])} for i in range(self.n_features_)]
        
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform the data based on fitted binariser.

        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            The data of shape (n_samples[, n_features]).
        args : Any
            Additional arguments.
        
        Returns
        -------
        L : Union[np.ndarray, torch.Tensor]
            The binarised data of shape (n_samples, n_classes).
        """
        
        # check shape of y
        if len(y.shape) == 1:
            y = y[:, None]
        
        if y.shape[1] != self.n_features_:
            raise ValueError(f'LabelBinariser expected {self.n_features_} features, but got {y.shape[1]}.')
        
        # check fit
        if self.labels_ is None or self.classes_ is None or self.map_L_to_C_ is None:
            raise ValueError(f'LabelBinariser must be fit before calling transform.')
        
        # check valid labels
        labels = [np.unique(y[:,i]) for i in range(self.n_features_)]
        
        for i in range(self.n_features_):
            if not np.isin(labels[i], self.labels_[i]).all():
                raise ValueError(f'Unknown labels in `y`.')
        
        # create mapped classes
        y_mapped = np.full(
            (y.shape[0], self.N_), 
            self.neg_label
        )
        
        for i in range(self.n_features_):
            for label in self.map_L_to_C_[i]:
                mask_l = (y[:,i] == label).squeeze()
                class_l = self.map_L_to_C_[i][label]
                class_o = self.C_[i]
                
                y_mapped[mask_l,class_o + class_l] = self.pos_label
        
        return y_mapped
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform the data in one step.
        
        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            The data of shape (n_samples[, n_features]).
        args : Any
            Additional arguments.
        
        Returns
        -------
        L : Union[np.ndarray, torch.Tensor]
            The binarised data of shape (n_samples, n_classes).
        """
        
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Obtain labels from transformed data.
        
        Parameters
        ----------
        L : Union[np.ndarray, torch.Tensor]
            The binarised data of shape (n_samples, n_classes).
        args : Any
            Additional arguments.
        
        Returns
        -------
        y : Union[np.ndarray, torch.Tensor]
            The labels of shape (n_samples, n_features).
        """
        
        # check shape of y
        if len(y.shape) == 1:
            y = y[:, None]
        
        if y.shape[1] != self.N_:
            raise ValueError(f'LabelBinariser expected {self.N_} subfeatures, but got {y.shape[1]}.')
        
        # check fit
        if self.labels_ is None or self.classes_ is None or self.map_L_to_C_ is None:
            raise ValueError(f'LabelBinariser must be fit before calling inverse_transform.')
        
        # create labels
        y_mapped = []
        
        for i in range(self.n_features_):
            y_i = y[:,self.C_[i]:self.C_[i] + len(self.classes_[i])]
            classes = self.labels_[i][np.argmax(y_i == self.pos_label, axis = 1)]
            y_mapped.append(classes)
        
        y_mapped = np.array(y_mapped).T
        
        return y_mapped
    
    def clone(self) -> "_LabelBinariser_numpy":
        """Obtain a clone of this class.
        
        Returns
        -------
        LabelBinariser
            The clone.
        """
        
        return _LabelBinariser_numpy(
            neg_label = self.neg_label, 
            pos_label = self.pos_label
        )

class _LabelBinariser_torch(sklearn.base.BaseEstimator):
    r"""Class to create and handle multiclass and multifeature one-hot encodings using torch backend.
    
    Parameters
    ----------
    neg_label : int, default=0
        Label to use for negatives.
    pos_label : int, default=1
        Label to use for positives.
    
    Attributes
    ----------
    neg_label : int, default=0
        Label to use for negatives.
    pos_label : int, default=1
        Label to use for positives.
    n_features_ : int
        Number of unique features in y of shape (n_samples, n_features).
    n_classes_ : List[int]
        Number of unique classes per feature.
    labels_ : List[List[Any]]
        List including lists of original labels in y.
    classes_ : List[List[Any]]
        List including lists of class identities in y.
    N_ : Union[int, torch.Tensor]
        Total number of classes (across features).
    C_ : torch.Tensor
        Offsets for each unique feature in one-hot matrix.
    map_L_to_C_ : List[Dict[Any, int]]
        Lists containing each label->class mapping per feature.
    """
    
    def __init__(self, neg_label: int = 0, pos_label: int = 1):
        """Obtain a binariser.
        
        Parameters
        ----------
        neg_label : int, default=0
            Label to use for negatives.
        pos_label : int, default=1
            Label to use for positives.
        """
        
        # check labels
        if pos_label <= neg_label:
            raise ValueError(f'Negative label must be smaller than positive label.')
        
        # setup opts
        self.neg_label = neg_label
        self.pos_label = pos_label
        
        # setup internals
        self.n_features_ = None
        self.n_classes_ = None
        self.labels_ = None
        self.classes_ = None
        self.N_ = None
        self.C_ = None
        self.map_L_to_C_ = None
    
    def fit(self, y: torch.Tensor) -> "_LabelBinariser_torch":
        """Fit the binariser.

        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            The data of shape (n_samples[, n_features]).
        args : Any
            Additional arguments.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The binariser.
        """
        
        # check shape of y
        if len(y.shape) == 1:
            y = y[:, None]
        
        # grab unique labels
        self.n_features_ = y.shape[1]
        self.n_classes_ = [torch.unique(y[:,i]).shape[0] for i in range(self.n_features_)]
        self.labels_ = [torch.unique(y[:,i]) for i in range(self.n_features_)]
        self.classes_ = [torch.arange(len(self.labels_[i])) for i in range(self.n_features_)]
        
        # setup N and cumsums
        self.N_ = torch.sum(torch.tensor(self.n_classes_, dtype = y.dtype, device = y.device))
        self.C_ = torch.tensor(self.n_classes_, dtype = y.dtype, device = y.device).cumsum(0)
        if self.C_.shape[0] > 1: self.C_ = torch.cat((torch.tensor([0.0], dtype = y.dtype, device = y.device), self.C_[:-1]))
        else: self.C_ = torch.tensor([0.0], dtype = y.dtype, device = y.device)
        
        # setup mapping
        self.map_L_to_C_ = [{label_: class_ for class_, label_ in zip(self.classes_[i], self.labels_[i])} for i in range(self.n_features_)]
        
        return self
    
    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """Transform the data based on fitted binariser.

        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            The data of shape (n_samples[, n_features]).
        args : Any
            Additional arguments.
        
        Returns
        -------
        L : Union[np.ndarray, torch.Tensor]
            The binarised data of shape (n_samples, n_classes).
        """
        
        # check shape of y
        if len(y.shape) == 1:
            y = y[:, None]
        
        if y.shape[1] != self.n_features_:
            raise ValueError(f'LabelBinariser expected {self.n_features_} features, but got {y.shape[1]}.')
        
        # check fit
        if self.labels_ is None or self.classes_ is None or self.map_L_to_C_ is None:
            raise ValueError(f'LabelBinariser must be fit before calling transform.')
        
        # check valid labels
        labels = [torch.unique(y[:,i]) for i in range(self.n_features_)]
        
        for i in range(self.n_features_):
            if not torch.isin(labels[i], self.labels_[i]).all():
                raise ValueError(f'Unknown labels in `y`.')
        
        # create mapped classes
        y_mapped = torch.full(
            (y.shape[0], int(self.N_.item())), 
            self.neg_label, device = y.device
        )
        
        for i in range(self.n_features_):
            for label in self.map_L_to_C_[i]:
                mask_l = (y[:,i] == label).squeeze()
                class_l = self.map_L_to_C_[i][label]
                class_o = self.C_[i]
                
                y_mapped[mask_l,(class_o + class_l).long()] = self.pos_label
        
        return y_mapped
    
    def fit_transform(self, y: torch.Tensor) -> torch.Tensor:
        """Fit and transform the data in one step.
        
        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            The data of shape (n_samples[, n_features]).
        args : Any
            Additional arguments.
        
        Returns
        -------
        L : Union[np.ndarray, torch.Tensor]
            The binarised data of shape (n_samples, n_classes).
        """
        
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """Obtain labels from transformed data.
        
        Parameters
        ----------
        L : Union[np.ndarray, torch.Tensor]
            The binarised data of shape (n_samples, n_classes).
        args : Any
            Additional arguments.
        
        Returns
        -------
        y : Union[np.ndarray, torch.Tensor]
            The labels of shape (n_samples, n_features).
        """
        
        # check shape of y
        if len(y.shape) == 1:
            y = y[:, None]
        
        if y.shape[1] != self.N_:
            raise ValueError(f'LabelBinariser expected {self.N_} subfeatures, but got {y.shape[1]}.')
        
        # check fit
        if self.labels_ is None or self.classes_ is None or self.map_L_to_C_ is None:
            raise ValueError(f'LabelBinariser must be fit before calling inverse_transform.')
        
        # create labels
        y_mapped = []
        
        for i in range(self.n_features_):
            y_i = y[:,self.C_[i].long():self.C_[i].long() + len(self.classes_[i])]
            classes = self.labels_[i][torch.argmax((y_i == self.pos_label).long(), dim = 1)]
            y_mapped.append(classes)
        
        y_mapped = torch.stack(y_mapped).T
        
        return y_mapped
    
    def clone(self) -> "_LabelBinariser_torch":
        """Obtain a clone of this class.
        
        Returns
        -------
        LabelBinariser
            The clone.
        """
        
        return _LabelBinariser_torch(
            neg_label = self.neg_label, 
            pos_label = self.pos_label
        )

class LabelBinariser(sklearn.base.BaseEstimator):
    r"""Class to create and handle multiclass and multifeature one-hot encodings.
    
    Parameters
    ----------
    neg_label : int, default=0
        Label to use for negatives.
    pos_label : int, default=1
        Label to use for positives.
    
    Attributes
    ----------
    neg_label : int, default=0
        Label to use for negatives.
    pos_label : int, default=1
        Label to use for positives.
    n_features_ : int
        Number of unique features in y of shape (n_samples, n_features).
    n_classes_ : List[int]
        Number of unique classes per feature.
    labels_ : List[List[Any]]
        List including lists of original labels in y.
    classes_ : List[List[Any]]
        List including lists of class identities in y.
    N_ : Union[int, np.ndarray, torch.Tensor]
        Total number of classes (across features).
    C_ : Union[np.ndarray, torch.Tensor]
        Offsets for each unique feature in one-hot matrix.
    map_L_to_C_ : List[Dict[Any, int]]
        Lists containing each label->class mapping per feature.

    Notes
    -----
    In the multifeature case where y is of shape (n_samples, n_features), the output matrix is still 2D:
    
    .. math::

        y \sim (s, f)
        L \sim (s, n)\qquad where\qquad n = \sum_{i = 0}^{f}\sum_{j = 0}^{c_i} 1
    
    As implied here, the output matrix will therefore always include one column per class per feature. This also means that, in the binary case where we have only one feature and two classes, we will obtain a matrix of shape (n_samples, 2). This behaviour differs from what libraries like sklearn implement, where this would result in (n_samples, 1) matrices. This choice made on purpose because, for some algorithms, this can make handling the data easier and more intuitive.
    
    Examples
    --------
    First, let's consider one feature that has three classes.
    
    >>> import torch
    >>> from mvpy.estimators import LabelBinariser
    >>> label = LabelBinariser().to_torch()
    >>> y = torch.randint(0, 3, (100,))
    >>> L = label.fit_transform(y)
    >>> H = label.inverse_transform(L)
    >>> print(y[0:5])
    tensor([0, 1, 2, 1, 2])
    >>> print(L[0:5])
    tensor([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]])
    >>> print(H[0:5])
    tensor([0, 1, 2, 1, 2])
    
    Second, let's look at two features that have a different number of classes each.
    
    >>> import torch
    >>> from mvpy.estimators import LabelBinariser
    >>> label = LabelBinariser().to_torch()
    >>> y = torch.stack((torch.randint(10, 13, (50,)), torch.randint(20, 22, (50,))), dim = 1)
    >>> L = label.fit_transform(y)
    >>> H = label.inverse_transform(L)
    >>> print(y[0:5])
    tensor([[10, 21],
            [10, 20],
            [11, 21],
            [12, 21],
            [10, 20]])
    >>> print(L[0:5])
    tensor([[1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0]])
    >>> print(H[0:5])
    tensor([[10, 21],
            [10, 20],
            [11, 21],
            [12, 21],
            [10, 20]])
    """

    def __init__(self, neg_label: int = 0, pos_label: int = 1):
        """Obtain a binariser.
        
        Parameters
        ----------
        neg_label : int, default=0
            Label to use for negatives.
        pos_label : int, default=1
            Label to use for positives.
        """

        self.neg_label = neg_label
        self.pos_label = pos_label
    
    def _get_estimator(self, y: Union[np.ndarray, torch.Tensor], *args: Any) -> sklearn.base.BaseEstimator:
        """Given the data, determine which binariser to use.

        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            The data of shape (n_samples[, n_features]).
        args : Any
            Additional arguments.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The binariser.
        """
        
        if isinstance(y, torch.Tensor):
            return _LabelBinariser_torch
        elif isinstance(y, np.ndarray):
            return _LabelBinariser_numpy
        
        raise TypeError(f'`y` must be either torch.Tensor or np.ndarray, but got {type(y)}.')
    
    def fit(self, y: Union[np.ndarray, torch.Tensor], *args: Any) -> sklearn.base.BaseEstimator:
        """Fit the binariser.

        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            The data of shape (n_samples[, n_features]).
        args : Any
            Additional arguments.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The binariser.
        """
        
        return self._get_estimator(y, *args)(neg_label = self.neg_label, pos_label = self.pos_label).fit(y, *args)
    
    def transform(self, y: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Transform the data based on fitted binariser.

        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            The data of shape (n_samples[, n_features]).
        args : Any
            Additional arguments.
        
        Returns
        -------
        L : Union[np.ndarray, torch.Tensor]
            The binarised data of shape (n_samples, n_classes).
        """

        return self._get_estimator(y, *args)(neg_label = self.neg_label, pos_label = self.pos_label).transform(y, *args)
    
    def inverse_transform(self, y: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Obtain labels from transformed data.
        
        Parameters
        ----------
        L : Union[np.ndarray, torch.Tensor]
            The binarised data of shape (n_samples, n_classes).
        args : Any
            Additional arguments.
        
        Returns
        -------
        y : Union[np.ndarray, torch.Tensor]
            The labels of shape (n_samples, n_features).
        """
        
        return self._get_estimator(y, *args)(neg_label = self.neg_label, pos_label = self.pos_label).inverse_transform(y, *args)
    
    def fit_transform(self, y: Union[np.ndarray, torch.Tensor], *args: Any) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform the data in one step.
        
        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            The data of shape (n_samples[, n_features]).
        args : Any
            Additional arguments.
        
        Returns
        -------
        L : Union[np.ndarray, torch.Tensor]
            The binarised data of shape (n_samples, n_classes).
        """
        
        return self._get_estimator(y, *args)(neg_label = self.neg_label, pos_label = self.pos_label).fit_transform(y, *args)
    
    def to_torch(self):
        """Select the torch binariser. Note that this cannot be called for conversion.
        
        Returns
        -------
        _LabelBinariser_torch
            The torch binariser.
        """
        
        return self._get_estimator(torch.tensor([1]))(neg_label = self.neg_label, pos_label = self.pos_label)
    
    def to_numpy(self):
        """Select the numpy binariser. Note that this cannot be called for conversion.

        Returns
        -------
        _LabelBinariser_numpy
            The numpy binariser.
        """

        return self._get_estimator(np.array([1]))(neg_label = self.neg_label, pos_label = self.pos_label)
    
    def clone(self) -> "LabelBinariser":
        """Obtain a clone of this class.
        
        Returns
        -------
        LabelBinariser
            The clone.
        """
        
        return LabelBinariser(with_mean = self.with_mean, with_std = self.with_std, dims = self.dims)
    
    def copy(self) -> "LabelBinariser":
        """Obtain a copy of this class.

        Returns
        -------
        LabelBinariser
            The copy.
        """
        
        return self.clone()