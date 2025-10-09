'''
A collection of estimators for decoding features using ridge classifiers.
'''

import numpy as np
import torch
import sklearn

from ..preprocessing.labelbinariser import _LabelBinariser_numpy, _LabelBinariser_torch

from typing import Union, Any, Dict, List

class _Classifier_numpy(sklearn.base.BaseEstimator):
    r"""Implements a wrapper for classifiers with numpy backend.
    
    This class is public facing, but should not generally be used. In essence, it provides a convenient wrapper for classifiers that solves OvR/OvO problems.
    
    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : str, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[Any, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    
    Attributes
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : str, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[Any, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    estimators_ : Union[sklearn.base.BaseEstimator, List[sklearn.base.BaseEstimator]]
        All instances of the estimator class (only of type list if OvO).
    binariser_ : LabelBinariser
        Label binariser used internally.
    coef_ : np.ndarray
        If available, coefficients from all classifiers ([n_classifiers,], n_channels, n_classes).
    intercept_ : np.ndarray
        If available, intercepts from all classifiers ([n_classifiers,], n_classes).
    pattern_ : np.ndarray
        If available, patterns from all classifiers ([n_classifiers,], n_channels, n_classes).
    offsets_ : np.ndarray
        Numerical offsets for each feature in outputs, used internally.
    """
    
    def __init__(self, estimator: sklearn.base.BaseEstimator, method: str = 'OvR', arguments: List[Any] = [], kwarguments: Dict[Any, Any] = dict()):
        """Obtain a classifier wrapper.
        
        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            The estimator type wrapped by this class.
        method : str, default='OvR'
            For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
        arguments : List[Any], default=[]
            Arguments to pass to the estimator at initialisation.
        kwarguments : Dict[Any, Any], default=dict()
            Keyword arguments to pass to the estimator at initialisation.
        """
        
        # setup args
        self.estimator = estimator
        self.method = method
        self.arguments = arguments
        self.kwarguments = kwarguments
        
        # check method
        if method not in ['OvR', 'OvO']:
            raise ValueError(f'Method `{method}` unknown. Must be [\'OvR\', \'OvO\'].')

        # setup internals
        self.estimators_ = None
        self.binariser_ = _LabelBinariser_numpy()
        self.coef_ = None
        self.intercept_ = None
        self.pattern_ = None
        self.offsets_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Classifier_numpy":
        """Fit the estimator.

        Parameters
        ----------
        X : np.ndarray
            The features of shape (n_samples, n_channels).
        y : np.ndarray
            The targets of shape (n_samples[, n_features]).
        
        Returns
        -------
        clf : _Classifier_numpy
            The classifier.
        """
        
        # check shape
        if len(y.shape) == 1:
            y = y[:, None]
        
        # check method
        if self.method == 'OvR':
            # create single estimator for OvR
            self.estimators_ = self.estimator(*self.arguments, **self.kwarguments)
            self.estimators_.fit(X, y)
            
            # collect
            if hasattr(self.estimators_, 'coef_'):
                self.coef_ = self.estimators_.coef_
            
            if hasattr(self.estimators_, 'intercept_'):
                self.intercept_ = self.estimators_.intercept_
            
            if hasattr(self.estimators_, 'pattern_'):
                self.pattern_ = self.estimators_.pattern_
        else:
            # fit labels
            L = self.binariser_.fit_transform(y)
            
            # create estimator list
            self.estimators_ = []
            self.coef_ = []
            self.intercept_ = []
            self.pattern_ = []
            self.offsets_ = []
            offset = 0
            
            # loop over features
            for i in range(self.binariser_.n_features_):
                # add offset
                self.offsets_.append(offset)
                
                # loop over pairs
                for j in range(self.binariser_.n_classes_[i]):
                    for k in range(self.binariser_.n_classes_[i]):
                        # skip if unnecessary
                        if j <= k: continue
                        
                        # find samples
                        indc_jk = (y[:,i] == self.binariser_.labels_[i][j]) \
                                    | (y[:,i] == self.binariser_.labels_[i][k])
                        indc_jk = indc_jk.squeeze()
                        
                        # fit estimator
                        est_ijk = self.estimator(*self.arguments, **self.kwarguments)
                        est_ijk.fit(X[indc_jk], y[indc_jk,i])
                        self.estimators_.append(est_ijk)
                        
                        # collect
                        if hasattr(est_ijk, 'coef_'):
                            self.coef_.append(est_ijk.coef_)
                        
                        if hasattr(est_ijk, 'intercept_'):
                            self.intercept_.append(est_ijk.intercept_)
                        
                        if hasattr(est_ijk, 'pattern_'):
                            self.pattern_.append(est_ijk.pattern_)

            # stack
            self.coef_ = np.array(self.coef_)
            self.intercept_ = np.array(self.intercept_)
            self.pattern_ = np.array(self.pattern_)
            self.offsets_ = np.array(self.offsets_)
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Predict from the estimator.

        Parameters
        ----------
        X : np.ndarray
            The features (n_samples, n_channels).

        Returns
        -------
        df : np.ndarray
            The predictions of shape (n_samples, n_classes).
        """
        
        # check model fit
        if self.estimators_ is None:
            raise ValueError(f'Classifier must be fit before calling decision function.')
        
        if self.method == 'OvR':
            # compute simple decision function
            return self.estimators_.decision_function(X)
        else:
            df = []
            
            for i in range(len(self.estimators_)):
                df.append(self.estimators_[i].decision_function(X))
            
            return np.array(df)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : np.ndarray
            The features (n_samples, n_channels).
        
        Returns
        -------
        y_h : np.ndarray
            The predictions of shape (n_samples, n_features).
        """
        
        # check model fit
        if self.estimators_ is None:
            raise ValueError(f'Classifier must be fit before calling predict.')
        
        if self.method == 'OvR':
            # for OvR, we can just use the estimator prediction
            return self.estimators_.predict(X)
        else:
            ijk = 0
            y_h = []
            
            # loop over features
            for i in range(self.binariser_.n_features_):
                # setup voting
                votes = []
                
                for j in range(self.binariser_.n_classes_[i]):
                    for k in range(self.binariser_.n_classes_[i]):
                        # skip if unnecessary
                        if j <= k: continue
                        
                        # get prediction
                        votes.append(self.estimators_[ijk].predict(X).squeeze())
                        
                        # move tally
                        ijk += 1
                
                # convert votes
                votes = np.array(votes)
                
                # count votes
                labels = self.binariser_.labels_[i]
                n_counts = np.zeros((votes.shape[1], self.binariser_.n_classes_[i]))
                
                for j, label in enumerate(labels):
                    n_counts[:,j] = (votes == label).sum(axis = 0)
                
                # decide winner
                top = np.argmax(n_counts, axis = 1)
                y_h.append(labels[top])
            
            # convert
            return np.array(y_h).T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict from the estimator.

        Parameters
        ----------
        X : np.ndarray
            The features (n_samples, n_channels).

        Returns
        -------
        df : np.ndarray
            The predictions of shape (n_samples, n_classes).
        """
        
        return self.decision_function(X)
    
    def clone(self) -> "_Classifier_numpy":
        """Clone this class.
        
        Returns
        -------
        clf : _Classifier_numpy
            The cloned object.
        """
        
        return _Classifier_numpy(
            estimator = self.estimator,
            method = self.method,
            arguments = self.arguments,
            kwarguments = self.kwarguments
        )

class _Classifier_torch(sklearn.base.BaseEstimator):
    r"""Implements a wrapper for classifiers with torch backend.
    
    This class is public facing, but should not generally be used. In essence, it provides a convenient wrapper for classifiers that solves OvR/OvO problems.
    
    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : str, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[Any, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    
    Attributes
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : str, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[Any, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    estimators_ : Union[sklearn.base.BaseEstimator, List[sklearn.base.BaseEstimator]]
        All instances of the estimator class (only of type list if OvO).
    binariser_ : LabelBinariser
        Label binariser used internally.
    coef_ : torch.Tensor
        If available, coefficients from all classifiers ([n_classifiers,], n_channels, n_classes).
    intercept_ : torch.Tensor
        If available, intercepts from all classifiers ([n_classifiers,], n_classes).
    pattern_ : torch.Tensor
        If available, patterns from all classifiers ([n_classifiers,], n_channels, n_classes).
    offsets_ : torch.Tensor
        Numerical offsets for each feature in outputs, used internally.
    """
    
    def __init__(self, estimator: sklearn.base.BaseEstimator, method: str = 'OvR', arguments: List[Any] = [], kwarguments: Dict[Any, Any] = dict()):
        """Obtain a classifier wrapper.
        
        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            The estimator type wrapped by this class.
        method : str, default='OvR'
            For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
        arguments : List[Any], default=[]
            Arguments to pass to the estimator at initialisation.
        kwarguments : Dict[Any, Any], default=dict()
            Keyword arguments to pass to the estimator at initialisation.
        """
        
        # setup args
        self.estimator = estimator
        self.method = method
        self.arguments = arguments
        self.kwarguments = kwarguments
        
        # check method
        if method not in ['OvR', 'OvO']:
            raise ValueError(f'Method `{method}` unknown. Must be [\'OvR\', \'OvO\'].')

        # setup internals
        self.estimators_ = None
        self.binariser_ = _LabelBinariser_torch()
        self.coef_ = None
        self.intercept_ = None
        self.pattern_ = None
        self.offsets_ = None
        
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "_Classifier_torch":
        """Fit the estimator.

        Parameters
        ----------
        X : torch.Tensor
            The features of shape (n_samples, n_channels).
        y : torch.Tensor
            The targets of shape (n_samples[, n_features]).
        
        Returns
        -------
        clf : _Classifier_torch
            The classifier.
        """
        
        # check shape
        if len(y.shape) == 1:
            y = y[:, None]
        
        # check method
        if self.method == 'OvR':
            # create single estimator for OvR
            self.estimators_ = self.estimator(*self.arguments, **self.kwarguments)
            self.estimators_.fit(X, y)
            
            # collect
            if hasattr(self.estimators_, 'coef_'):
                self.coef_ = self.estimators_.coef_
            
            if hasattr(self.estimators_, 'intercept_'):
                self.intercept_ = self.estimators_.intercept_
            
            if hasattr(self.estimators_, 'pattern_'):
                self.pattern_ = self.estimators_.pattern_
        else:
            # fit labels
            L = self.binariser_.fit_transform(y)
            
            # create estimator list
            self.estimators_ = []
            self.coef_ = []
            self.intercept_ = []
            self.pattern_ = []
            self.offsets_ = []
            offset = 0
            
            # loop over features
            for i in range(self.binariser_.n_features_):
                # add offset
                self.offsets_.append(offset)
                
                # loop over pairs
                for j in range(self.binariser_.n_classes_[i]):
                    for k in range(self.binariser_.n_classes_[i]):
                        # skip if unnecessary
                        if j <= k: continue
                        
                        # find samples
                        indc_jk = (y[:,i] == self.binariser_.labels_[i][j]) \
                                    | (y[:,i] == self.binariser_.labels_[i][k])
                        indc_jk = indc_jk.squeeze()
                        
                        # fit estimator
                        est_ijk = self.estimator(*self.arguments, **self.kwarguments)
                        est_ijk.fit(X[indc_jk], y[indc_jk,i])
                        self.estimators_.append(est_ijk)
                        
                        # collect
                        if hasattr(est_ijk, 'coef_'):
                            self.coef_.append(est_ijk.coef_)
                        
                        if hasattr(est_ijk, 'intercept_'):
                            self.intercept_.append(est_ijk.intercept_)
                        
                        if hasattr(est_ijk, 'pattern_'):
                            self.pattern_.append(est_ijk.pattern_)

            # stack
            if len(self.coef_) > 0: 
                self.coef_ = torch.stack(self.coef_)
            else:
                self.coef_ = torch.tensor(self.coef_, dtype = X.dtype, device = X.device)
            
            self.intercept_ = torch.stack(self.intercept_)
            
            if len(self.pattern_) > 0: 
                self.pattern_ = torch.stack(self.pattern_)
            else:
                self.pattern = torch.tensor(self.pattern_, dtype = X.dtype, device = X.device)
            
            self.offsets_ = torch.tensor(self.offsets_)
        
        return self
    
    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Predict from the estimator.

        Parameters
        ----------
        X : torch.Tensor
            The features (n_samples, n_channels).

        Returns
        -------
        df : torch.Tensor
            The predictions of shape (n_samples, n_classes).
        """
        
        # check model fit
        if self.estimators_ is None:
            raise ValueError(f'Classifier must be fit before calling decision function.')
        
        if self.method == 'OvR':
            # compute simple decision function
            return self.estimators_.decision_function(X)
        else:
            df = []
            
            for i in range(len(self.estimators_)):
                df.append(self.estimators_[i].decision_function(X))
            
            return torch.stack(df)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            The features (n_samples, n_channels).
        
        Returns
        -------
        y_h : torch.Tensor
            The predictions of shape (n_samples, n_features).
        """
        
        # check model fit
        if self.estimators_ is None:
            raise ValueError(f'Classifier must be fit before calling predict.')
        
        if self.method == 'OvR':
            # for OvR, we can just use the estimator prediction
            return self.estimators_.predict(X)
        else:
            ijk = 0
            y_h = []
            
            # loop over features
            for i in range(self.binariser_.n_features_):
                # setup voting
                votes = []
                
                for j in range(self.binariser_.n_classes_[i]):
                    for k in range(self.binariser_.n_classes_[i]):
                        # skip if unnecessary
                        if j <= k: continue
                        
                        # get prediction
                        votes.append(self.estimators_[ijk].predict(X).squeeze())
                        
                        # move tally
                        ijk += 1
                
                # convert votes
                votes = torch.stack(votes)
                
                # count votes
                labels = self.binariser_.labels_[i]
                n_counts = torch.zeros((votes.shape[1], self.binariser_.n_classes_[i]), dtype = X.dtype, device = X.device)
                
                for j, label in enumerate(labels):
                    n_counts[:,j] = (votes == label).float().sum(0)
                
                # decide winner
                top = torch.argmax(n_counts, dim = 1)
                y_h.append(labels[top])
            
            # convert
            return torch.stack(y_h).T

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Predict from the estimator.

        Parameters
        ----------
        X : torch.Tensor
            The features (n_samples, n_channels).

        Returns
        -------
        df : torch.Tensor
            The predictions of shape (n_samples, n_classes).
        """
        
        return self.decision_function(X)
    
    def clone(self) -> "_Classifier_torch":
        """Clone this class.
        
        Returns
        -------
        clf : _Classifier_torch
            The cloned object.
        """
        
        return _Classifier_torch(
            estimator = self.estimator,
            method = self.method,
            arguments = self.arguments,
            kwarguments = self.kwarguments
        )

class Classifier(sklearn.base.BaseEstimator):
    """Implements a wrapper for classifiers that handle one-versus-one (OvO) and one-versus-rest (OvR) classification schemes.
    
    While this class is exposed publically, there are few (if any) direct use 
    cases for this class. In principle, it exists for other classifiers that
    want to handle multi-class cases as OvO or OvR as a wrapper function that
    can either be inherited or created as a super class, specifying the desired
    estimator (recommended option).
    
    One-versus-rest (``OvR``) classification computes the decision functions over
    inputs :math:`X` and then takes the maximum value across decision values
    to predict the most likely classes :math:`\\hat{y}`.
    
    One-versus-one (``OvO``) classification computes all decision functions from
    binary classifiers (e.g., :math:`c_0` vs :math:`c_1`, :math:`c_0` vs :math:`c_2`,
    :math:`c_1` vs :math:`c_2`, ...). For each individual classification problem,
    the maximum value is recorded as one vote for the winning class. Votes are
    then aggregated across all classifiers and the maximum number of votes decides
    the most likely classes :math:`\\hat{y}`.
    
    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : {'OvR', 'OvO'}, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[str, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    
    Attributes
    ----------
    estimator : sklearn.base.BaseEstimator
        The estimator type wrapped by this class.
    method : {'OvR', 'OvO'}, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    arguments : List[Any], default=[]
        Arguments to pass to the estimator at initialisation.
    kwarguments : Dict[str, Any], default=dict()
        Keyword arguments to pass to the estimator at initialisation.
    estimators_ : sklearn.base.BaseEstimator | List[sklearn.base.BaseEstimator]
        All instances of the estimator class (only of type list if OvO).
    binariser_ : mvpy.estimators.LabelBinariser
        Label binariser used internally.
    coef_ : np.ndarray | torch.Tensor
        If available, coefficients from all classifiers ``([n_classifiers,] n_channels, n_classes)``.
    intercept_ : np.ndarray | torch.Tensor
        If available, intercepts from all classifiers ``([n_classifiers,] n_classes)``.
    pattern_ : np.ndarray | torch.Tensor
        If available, patterns from all classifiers ``([n_classifiers,] n_channels, n_classes)``.
    offsets_ : np.ndarray | torch.Tensor
        Numerical offsets for each feature in outputs, used internally.
    
    See also
    --------
    mvpy.estimators.RidgeClassifier, mvpy.estimators.SVC : Classifiers that use this class as a wrapper.
    mvpy.preprocessing.LabelBinariser : Label binariser used internally to generated one-hot encodings.
    """
    
    def __init__(self, estimator: sklearn.base.BaseEstimator, method: str = 'OvR', arguments: List[Any] = [], kwarguments: Dict[Any, Any] = dict()):
        """Obtain a classifier wrapper.
        
        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            The estimator type wrapped by this class.
        method : {'OvR', 'OvO}, default='OvR'
            For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
        arguments : List[Any], default=[]
            Arguments to pass to the estimator at initialisation.
        kwarguments : Dict[str, Any], default=dict()
            Keyword arguments to pass to the estimator at initialisation.
        """
        
        # setup args
        self.estimator = estimator
        self.method = method
        self.arguments = arguments
        self.kwarguments = kwarguments
    
    def _get_estimator(self, X: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> sklearn.base.BaseEstimator:
        """Obtain the wrapper and estimator for this SVC.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        y : np.ndarray | torch.Tensor
            Input labels of shape ``(n_samples[, n_features])``.
        
        Returns
        -------
        clf : mvpy.estimators.Classifier
            The classifier.
        """
        
        if isinstance(X, torch.Tensor) & isinstance(y, torch.Tensor):
            return _Classifier_torch(
                self.estimator,
                method = self.method,
                arguments = self.arguments,
                kwarguments = self.kwarguments
            )
        elif isinstance(X, np.ndarray) & isinstance(y, np.ndarray):
            return _Classifier_numpy(
                self.estimator,
                method = self.method,
                arguments = self.arguments,
                kwarguments = self.kwarguments
            )
        
        raise TypeError(f'`X` and `y` must be either torch.Tensor or np.ndarray, but got {type(X)} and {type(y)}.')
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> sklearn.base.BaseEstimator:
        """Fit the estimator.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The features of shape ``(n_samples, n_channels)``.
        y : np.ndarray | torch.Tensor
            The targets of shape ``(n_samples[, n_features])``.
        
        Returns
        -------
        clf : mvpy.estimators.Classifier
            The classifier.
        """
        
        return self._get_estimator(X, y).fit(X, y)
    
    def decision_function(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The features ``(n_samples, n_channels)``.

        Returns
        -------
        df : np.ndarray | torch.Tensor
            The predictions of shape ``(n_samples, n_classes)``.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The features ``(n_samples, n_channels)``.
        
        Returns
        -------
        y_h : np.ndarray | torch.Tensor
            The predictions of shape ``(n_samples, n_features)``.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            The features ``(n_samples, n_channels)``.

        Returns
        -------
        df : np.ndarray | torch.Tensor
            The predictions of shape ``(n_samples, n_classes)``.
        
        .. warning::
            Methods that predict the probability of classes are currently
            not implemented and will return decision function outputs
            instead. This is because probabilities are not trivial to 
            compute and require careful calibration, which we will implement
            in the future.
        """

        raise NotImplementedError('This method is not implemented in the base class.')
    
    def to_torch(self) -> sklearn.base.BaseEstimator:
        """Obtain the estimator with torch as backend.
        
        Returns
        -------
        clf : mvpy.estimators.classifier._Classifier_torch
            The estimator.
        """
        
        return self._get_estimator(torch.tensor([1.0]), torch.tensor([1.0]))
    
    def to_numpy(self) -> sklearn.base.BaseEstimator:
        """Obtain the estimator with numpy as backend.
        
        Returns
        -------
        clf : mvpy.estimators.classifier._Classifier_numpy
            The estimator.
        """
        
        return self._get_estimator(np.array([1.0]), np.array([1.0]))
    
    def clone(self) -> "Classifier":
        """Clone this class.
        
        Returns
        -------
        clf : Classifier
            The cloned object.
        """
        
        return Classifier(
            self.estimator,
            method = self.method,
            arguments = self.arguments,
            kwarguments = self.kwarguments
        )
    
    def copy(self) -> "Classifier":
        """Clone this class.
        
        Returns
        -------
        clf : Classifier
            The cloned object.
        """
        
        return self.clone()