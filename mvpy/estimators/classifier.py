'''
A collection of estimators for decoding features using ridge classifiers.
'''

import numpy as np
import torch
import sklearn

from .decoder import _Decoder_numpy, _Decoder_torch

from typing import Union, Any

class _ClassifierSingle_numpy(sklearn.base.BaseEstimator):
    """Implements a simple ridge classifier.
    
    Parameters
    ----------
    alpha : np.ndarray
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to use a different penalty for each target.
    
    Attributes
    ----------
    estimator : mvpy.estimators.RidgeCV
        The ridge estimator.
    classes_ : dict
        The classes of the classifier.
    intercept__ : np.ndarray
        The intercepts of the classifier.
    coef_ : np.ndarray
        The coefficients of the classifier.
    pattern_ : np.ndarray
        The pattern of the classifier.
    """
    
    def __init__(self, alpha: np.ndarray, **kwargs):
        """Obtain a new classifier.
        
        Parameters
        ----------
        alphas : np.ndarray
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments.
        """
        
        # setup opts
        self.alpha = alpha
        self.fit_intercept = True if 'fit_intercept' not in kwargs else kwargs['fit_intercept']
        self.normalise = True if 'normalise' not in kwargs else kwargs['normalise']
        self.alpha_per_target = True if 'alpha_per_target' not in kwargs else kwargs['alpha_per_target']
        
        # setup estimator
        self.estimator = _Decoder_numpy(alpha = alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)
        
        # setup attributes
        self.classes_ = None
        self.intercept_ = None
        self.coef_ = None
        self.pattern_ = None
        
    def _get_mapped(self, y: np.ndarray) -> np.ndarray:
        """Find the mapping of this classifier.
        
        Parameters
        ----------
        y : np.ndarray
            The target data.
        
        Returns
        -------
        mapping : dict
            The mapping of the classifier.
        """
        
        # check if classes exist
        if self.classes_ is None:
            # get unique classes
            classes = np.unique(y).astype(int)
            
            # check if there are only two classes
            if classes.shape[0] != 2:
                raise ValueError(f'_ClassifierSingle_numpy takes exactly two classes, but got {classes}.')

            # map numerically
            self.classes_ = {i: j for i, j in zip(classes, np.array([-1, 1]).astype(y.dtype))}
        
        # convert y to classes
        y = y.copy()
        
        for k in self.classes_:
            y[(y == k).astype(bool)] = self.classes_[k]
        
        return y
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the classifier.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target data.
        """
        
        # obtain mapping
        y = self._get_mapped(y)
        
        # fit estimator
        self.estimator.fit(X, y)
        
        # copy data
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        self.pattern_ = self.estimator.pattern_
        
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Predict the target.
        
        Parameters
        ----------
        X : np.ndarray
            The input data.
        
        Returns
        -------
        y : np.ndarray
            The predicted target.
        """
        
        y = self.estimator.predict(X)
        
        return y
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        y : np.ndarray
            The predicted target.
        """
        
        # predict
        y = self.estimator.predict(X)
        l = np.zeros_like(y, dtype = X.dtype)
        
        # map back
        for k in self.classes_:
            if self.classes_[k] > 0:
                l[(y > 0).astype(bool)] = k
            else:
                l[(y < 0).astype(bool)] = k

        return l
    
    def clone(self):
        """Clone this estimator.
        
        Returns
        -------
        _ClassifierSingle_numpy
            The cloned estimator.
        """
        
        return _ClassifierSingle_numpy(alpha = self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)

class _ClassifierSingle_torch(sklearn.base.BaseEstimator):
    """Implements a simple ridge classifier.
    
    Parameters
    ----------
    alpha : torch.Tensor
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to use a different penalty for each target.
    
    Attributes
    ----------
    estimator : mvpy.estimators.RidgeCV
        The ridge estimator.
    classes_ : dict
        The classes of the classifier.
    intercept_ : torch.Tensor
        The intercept of the classifier.
    coef_ : torch.Tensor
        The coefficients of the classifier.
    pattern_ : torch.Tensor
        The pattern of the classifier.
    """
    
    def __init__(self, alpha: torch.Tensor, **kwargs):
        """Obtain a new classifier.
        
        Parameters
        ----------
        alphas : torch.Tensor
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments.
        """
        
        # setup opts
        self.alpha = alpha
        self.fit_intercept = True if 'fit_intercept' not in kwargs else kwargs['fit_intercept']
        self.normalise = True if 'normalise' not in kwargs else kwargs['normalise']
        self.alpha_per_target = True if 'alpha_per_target' not in kwargs else kwargs['alpha_per_target']
        
        # setup estimator
        self.estimator = _Decoder_torch(alpha = alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)
        
        # setup attributes
        self.classes_ = None
        self.intercept_ = None
        self.coef_ = None
        self.pattern_ = None
        
    def _get_mapped(self, y: torch.Tensor) -> torch.Tensor:
        """Find the mapping of this classifier.
        
        Parameters
        ----------
        y : torch.Tensor
            The target data.
        
        Returns
        -------
        mapping : dict
            The mapping of the classifier.
        """
        
        # check if classes exist
        if self.classes_ is None:
            # get unique classes
            classes = torch.unique(y).to(torch.int32)
            
            # check if there are only two classes
            if classes.shape[0] != 2:
                raise ValueError(f'_ClassifierSingle_torch takes exactly two classes, but got {classes}.')

            # map numerically
            self.classes_ = {i: j for i, j in zip(classes, torch.tensor([-1, 1]).to(y.dtype).to(y.device))}
        
        # convert y to classes
        y = y.clone()
        
        for k in self.classes_:
            y[(y == k).to(torch.bool)] = self.classes_[k]
        
        return y
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the classifier.

        Parameters
        ----------
        X : torch.Tensor
            The input data.
        y : torch.Tensor
            The target data.
        """
        
        # obtain mapping
        y = self._get_mapped(y)
        
        # fit estimator
        self.estimator.fit(X, y)
        
        # copy data
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        self.pattern_ = self.estimator.pattern_
        
    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the target.
        
        Parameters
        ----------
        X : torch.Tensor
            The input data.
        
        Returns
        -------
        y : torch.Tensor
            The predicted target.
        """
        
        y = self.estimator.predict(X)
        
        return y
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the target.

        Parameters
        ----------
        X : torch.Tensor
            The input data.

        Returns
        -------
        y : torch.Tensor
            The predicted target.
        """
        
        # predict
        y = self.estimator.predict(X)
        l = torch.zeros_like(y, dtype = X.dtype, device = X.device)
        
        # map back
        for k in self.classes_:
            if self.classes_[k] > 0:
                l[(y > 0).to(torch.bool)] = k
            else:
                l[(y < 0).to(torch.bool)] = k

        return l
    
    def clone(self):
        """Clone this estimator.
        
        Returns
        -------
        _ClassifierSingle_torch
            The cloned estimator.
        """
        
        return _ClassifierSingle_torch(alpha = self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)

class _ClassifierOvO_numpy(sklearn.base.BaseEstimator):
    """Implements a simple ridge classifier.
    
    Parameters
    ----------
    alpha : np.ndarray
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to use a different penalty for each target.
    
    Attributes
    ----------
    estimators_ : list
        The estimators.
    classes_ : dict
        The classes of the classifier.
    intercept_ : np.ndarray
        The intercept of the classifier.
    coef_ : np.ndarray
        The coefficients of the classifier.
    pattern_ : np.ndarray
        The pattern of the classifier.
    """
    
    def __init__(self, alpha: np.ndarray, **kwargs):
        """Obtain a new classifier.
        
        Parameters
        ----------
        alphas : np.ndarray
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments.
        """
        
        # setup opts
        self.alpha = alpha
        self.fit_intercept = True if 'fit_intercept' not in kwargs else kwargs['fit_intercept']
        self.normalise = True if 'normalise' not in kwargs else kwargs['normalise']
        self.alpha_per_target = True if 'alpha_per_target' not in kwargs else kwargs['alpha_per_target']
        
        # setup attributes
        self.classes_ = None
        self.estimators_ = None
        self.intercept_ = None
        self.coef_ = None
        self.pattern_ = None
    
    def _get_mapped(self, y: np.ndarray) -> np.ndarray:
        """Find the mapping of this classifier.

        Parameters
        ----------
        y : np.ndarray
            The target data.

        Returns
        -------
        mapping : dict
            The mapping of the classifier.
        """

        # check if classes exist
        if self.classes_ is None:
            # get unique classes
            classes = np.unique(y).astype(int)

            # set classes
            self.classes_ = {i: j for i, j in zip(classes, np.arange(classes.shape[0]).astype(y.dtype))}
        
        # convert y to classes
        y = y.copy()
        
        for k in self.classes_:
            y[(y == k).astype(bool)] = self.classes_[k]
        
        return y
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the classifier.
        
        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target data
        """
        
        # obtain mapping
        y = self._get_mapped(y)
        
        # fit estimators
        self.estimators_ = []
        self.coef_ = []
        self.intercept_ = []
        self.pattern_ = []
        
        for i, class_i in enumerate(self.classes_):
            for j, class_j in enumerate(self.classes_):
                if i <= j: continue
                
                # fit this classifier
                indc = ((y == class_i) | (y == class_j)).astype(bool).squeeze()
                estimator = _ClassifierSingle_numpy(alpha = self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)
                estimator.fit(X[indc], y[indc])
                self.estimators_.append(estimator)

                self.coef_.append(estimator.coef_)
                self.intercept_.append(estimator.intercept_)
                self.pattern_.append(estimator.pattern_)

        self.coef_ = np.stack(self.coef_)
        self.intercept_ = np.stack(self.intercept_)
        self.pattern_ = np.stack(self.pattern_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        
        Returns
        -------
        y : np.ndarray
            The predicted target.
        """
        
        # check if fit
        if (self.estimators_ is None) | (len(self.estimators_) == 0):
            raise ValueError('Classifier has not been fit.')

        # predict
        y = np.zeros((X.shape[0], len(self.classes_)), dtype = X.dtype)
        
        indx = 0
        for i, class_i in enumerate(self.classes_):
            for j, class_j in enumerate(self.classes_):
                if i <= j: continue

                # predict this classifier
                preds = self.estimators_[indx].predict(X).squeeze().astype(int)
                y[np.arange(preds.shape[0]),preds] += 1
                
                # tally
                indx += 1
        
        return y.argmax(axis = 1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the target.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        
        Returns
        -------
        y : np.ndarray
            The predicted target.
        """
        
        # check if fit
        if (self.estimators_ is None) | (len(self.estimators_) == 0):
            raise ValueError('Classifier has not been fit.')

        # predict
        y = np.zeros((X.shape[0], len(self.classes_), len(self.classes_)), dtype = X.dtype)
        
        indx = 0
        for i, class_i in enumerate(self.classes_):
            for j, class_j in enumerate(self.classes_):
                if i <= j: continue

                # predict this classifier
                df = self.estimators_[indx].decision_function(X).squeeze()
                y[:,i,j] = -df
                y[:,j,i] = df
                
                # tally
                indx += 1
        
        return y
    
    def clone(self):
        """Clone the classifier.
        
        Returns
        -------
        _ClassifierOvO_numpy
            The cloned classifier.
        """
        
        return _ClassifierOvO_numpy(alpha = self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)

class _ClassifierOvO_torch(sklearn.base.BaseEstimator):
    """Implements a simple ridge classifier.
    
    Parameters
    ----------
    alpha : torch.Tensor
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to use a different penalty for each target.
    
    Attributes
    ----------
    estimators_ : list
        The estimators.
    classes_ : dict
        The classes of the classifier.
    interceptt_ : np.ndarray
        The intercepts.
    coef_ : np.ndarray
        The coefficients.
    pattern_ : np.ndarray
        The pattern.
    """
    
    def __init__(self, alpha: torch.Tensor, **kwargs):
        """Obtain a new classifier.
        
        Parameters
        ----------
        alphas : torch.Tensor
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments.
        """
        
        # setup opts
        self.alpha = alpha
        self.fit_intercept = True if 'fit_intercept' not in kwargs else kwargs['fit_intercept']
        self.normalise = True if 'normalise' not in kwargs else kwargs['normalise']
        self.alpha_per_target = True if 'alpha_per_target' not in kwargs else kwargs['alpha_per_target']
        
        # setup attributes
        self.classes_ = None
        self.estimators_ = None
        self.intercept_ = None
        self.coef_ = None
        self.pattern_ = None
    
    def _get_mapped(self, y: torch.Tensor) -> torch.Tensor:
        """Find the mapping of this classifier.

        Parameters
        ----------
        y : torch.Tensor
            The target data.

        Returns
        -------
        mapping : dict
            The mapping of the classifier.
        """

        # check if classes exist
        if self.classes_ is None:
            # get unique classes
            classes = torch.unique(y).to(torch.int32)

            # set classes
            self.classes_ = {i: j for i, j in zip(classes, torch.arange(classes.shape[0]).to(y.dtype).to(y.device))}
        
        # convert y to classes
        y = y.clone()
        
        for k in self.classes_:
            y[(y == k).to(torch.bool)] = self.classes_[k]
        
        return y
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the classifier.
        
        Parameters
        ----------
        X : torch.Tensor
            The input data.
        y : torch.Tensor
            The target data
        """
        
        # obtain mapping
        y = self._get_mapped(y)
        
        # fit estimators
        self.estimators_ = []
        self.coef_ = []
        self.intercept_ = []
        self.pattern_ = []
        
        for i, class_i in enumerate(self.classes_):
            for j, class_j in enumerate(self.classes_):
                if i <= j: continue
                
                # fit this classifier
                indc = ((y == class_i) | (y == class_j)).to(torch.bool).squeeze()
                estimator = _ClassifierSingle_torch(alpha = self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)
                estimator.fit(X[indc], y[indc])
                self.estimators_.append(estimator)
                
                self.coef_.append(estimator.coef_)
                self.intercept_.append(estimator.intercept_)
                self.pattern_.append(estimator.pattern_)

        self.coef_ = torch.stack(self.coef_)
        self.intercept_ = torch.stack(self.intercept_)
        self.pattern_ = torch.stack(self.pattern_)
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the target.

        Parameters
        ----------
        X : torch.Tensor
            The input data.
        
        Returns
        -------
        y : torch.Tensor
            The predicted target.
        """
        
        # check if fit
        if (self.estimators_ is None) | (len(self.estimators_) == 0):
            raise ValueError('Classifier has not been fit.')

        # predict
        y = torch.zeros((X.shape[0], len(self.classes_)), dtype = X.dtype, device = X.device)
        
        indx = 0
        for i, class_i in enumerate(self.classes_):
            for j, class_j in enumerate(self.classes_):
                if i <= j: continue

                # predict this classifier
                preds = self.estimators_[indx].predict(X).squeeze().to(torch.int32)
                y[torch.arange(preds.shape[0]),preds] += 1
                
                # tally
                indx += 1
        
        return y.argmax(dim = 1)
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the target.

        Parameters
        ----------
        X : torch.Tensor
            The input data.
        
        Returns
        -------
        y : torch.Tensor
            The predicted target.
        """
        
        # check if fit
        if (self.estimators_ is None) | (len(self.estimators_) == 0):
            raise ValueError('Classifier has not been fit.')

        # predict
        y = torch.zeros((X.shape[0], len(self.classes_), len(self.classes_)), dtype = X.dtype, device = X.device)
        
        indx = 0
        for i, class_i in enumerate(self.classes_):
            for j, class_j in enumerate(self.classes_):
                if i <= j: continue

                # predict this classifier
                df = self.estimators_[indx].decision_function(X).squeeze()
                y[:,i,j] = -df
                y[:,j,i] = df
                
                # tally
                indx += 1
        
        return y
    
    def clone(self):
        """Clone the classifier.
        
        Returns
        -------
        _ClassifierOvO_torch
            The cloned classifier.
        """
        
        return _ClassifierOvO_torch(alpha = self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)

class _ClassifierOvR_numpy(sklearn.base.BaseEstimator):
    """Implements a one-vs-rest classifier.

    Parameters
    ----------
    alpha : Union[torch.Tensor, np.ndarray, float, int], default=1
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to use a different penalty for each target.

    Attributes
    ----------
    estimators_ : List[sklearn.base.BaseEstimator]
        The estimators.
    classes_ : Dict[int, int]
        The classes.
    intercept_ : Union[np.ndarray, torch.Tensor]
        The intercepts of the classifiers.
    coef_ : Union[np.ndarray, torch.Tensor]
        The coefficients of the classifiers.
    pattern_ : Union[np.ndarray, torch.Tensor]
        The patterns of the classifiers.
    """

    def __init__(self, alpha: np.ndarray, **kwargs):
        """Obtain a new classifier.
        
        Parameters
        ----------
        alpha : np.ndarray
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments.
        """
        
        # setup opts
        self.alpha = alpha
        self.fit_intercept = True if 'fit_intercept' not in kwargs else kwargs['fit_intercept']
        self.normalise = True if 'normalise' not in kwargs else kwargs['normalise']
        self.alpha_per_target = True if 'alpha_per_target' not in kwargs else kwargs['alpha_per_target']
        
        # setup attributes
        self.classes_ = None
        self.estimator_ = _Decoder_numpy(alpha = self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)
        self.intercept_ = None
        self.coef_ = None
        self.pattern_ = None
    
    def _get_mapped(self, y: np.ndarray) -> np.ndarray:
        """Find the mapping of this classifier.

        Parameters
        ----------
        y : np.ndarray
            The target data.

        Returns
        -------
        mapping : dict
            The mapping of the classifier.
        """

        # check if classes exist
        if self.classes_ is None:
            # get unique classes
            classes = np.unique(y).astype(int)

            # set classes
            self.classes_ = {i: j for i, j in zip(classes, np.arange(classes.shape[0]).astype(y.dtype))}
        
        # convert y to classes
        y = y.copy()
        
        for k in self.classes_:
            y[(y == k).astype(bool)] = self.classes_[k]
        
        return y
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the classifier.
        
        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target data
        """
        
        # obtain mapping
        y = self._get_mapped(y)
        
        # setup y masks
        y_l = np.ones((y.shape[0], len(self.classes_)), dtype = y.dtype) * -1.
        
        for k in self.classes_:
            y_l[(y == k).astype(bool).squeeze(),self.classes_[k].astype(int)] = 1.0
        
        # fit estimators
        self.estimator_.fit(X, y_l)
        
        # get attributes
        self.intercept_ = self.estimator_.intercept_
        self.coef_ = self.estimator_.coef_
        self.pattern_ = self.estimator_.pattern_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target data.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        y : np.ndarray
            The predicted target data.
        """

        # predict
        y = self.estimator_.predict(X)

        # get predictions
        y = np.argmax(y, axis = 1)
        
        return y

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the target data.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        y : np.ndarray
            The predicted target data.
        """
        
        return self.estimator_.predict(X)
    
    def clone(self):
        """Clone the classifier.
        
        Returns
        -------
        _ClassifierOvR_numpy
            The cloned classifier.
        """
        
        return _ClassifierOvR_numpy(alpha = self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)

class _ClassifierOvR_torch(sklearn.base.BaseEstimator):
    """Implements a one-vs-rest classifier.

    Parameters
    ----------
    alpha : Union[torch.Tensor, np.ndarray, float, int], default=1
        The penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to use a different penalty for each target.

    Attributes
    ----------
    estimators_ : List[sklearn.base.BaseEstimator]
        The estimators.
    classes_ : Dict[int, int]
        The classes.
    intercept_ : Union[np.ndarray, torch.Tensor]
        The intercepts of the classifiers.
    coef_ : Union[np.ndarray, torch.Tensor]
        The coefficients of the classifiers.
    pattern_ : Union[np.ndarray, torch.Tensor]
        The patterns of the classifiers.
    """

    def __init__(self, alpha: torch.Tensor, **kwargs):
        """Obtain a new classifier.
        
        Parameters
        ----------
        alpha : np.ndarray
            The penalties to use for estimation.
        kwargs : Any
            Additional arguments.
        """
        
        # setup opts
        self.alpha = alpha
        self.fit_intercept = True if 'fit_intercept' not in kwargs else kwargs['fit_intercept']
        self.normalise = True if 'normalise' not in kwargs else kwargs['normalise']
        self.alpha_per_target = True if 'alpha_per_target' not in kwargs else kwargs['alpha_per_target']
        
        # setup attributes
        self.classes_ = None
        self.estimator_ = _Decoder_torch(alpha = self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)
        self.intercept_ = None
        self.coef_ = None
        self.pattern_ = None
    
    def _get_mapped(self, y: torch.Tensor) -> torch.Tensor:
        """Find the mapping of this classifier.

        Parameters
        ----------
        y : torch.Tensor
            The target data.

        Returns
        -------
        mapping : dict
            The mapping of the classifier.
        """

        # check if classes exist
        if self.classes_ is None:
            # get unique classes
            classes = torch.unique(y).to(torch.int32)

            # set classes
            self.classes_ = {i: j for i, j in zip(classes, torch.arange(classes.shape[0]).to(y.dtype).to(y.device))}
        
        # convert y to classes
        y = y.clone()
        
        for k in self.classes_:
            y[(y == k).to(torch.bool)] = self.classes_[k]
        
        return y
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the classifier.
        
        Parameters
        ----------
        X : torch.Tensor
            The input data.
        y : torch.Tensor
            The target data
        """
        
        # obtain mapping
        y = self._get_mapped(y)
        
        # setup y masks
        y_l = torch.ones((y.shape[0], len(self.classes_)), dtype = y.dtype, device = y.device) * -1.
        
        for k in self.classes_:
            y_l[(y == k).to(torch.bool).squeeze(),self.classes_[k].to(torch.int32).item()] = 1.0
        
        # fit estimators
        self.estimator_.fit(X, y_l)
        
        # get attributes
        self.intercept_ = self.estimator_.intercept_
        self.coef_ = self.estimator_.coef_
        self.pattern_ = self.estimator_.pattern_
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the target data.

        Parameters
        ----------
        X : torch.Tensor
            The input data.

        Returns
        -------
        y : torch.Tensor
            The predicted target data.
        """

        # predict
        y = self.estimator_.predict(X)

        # get predictions
        y = torch.argmax(y, dim = 1)
        
        return y

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the target data.

        Parameters
        ----------
        X : torch.Tensor
            The input data.

        Returns
        -------
        y : torch
            The predicted target data.
        """
        
        return self.estimator_.predict(X)
    
    def clone(self):
        """Clone the classifier.
        
        Returns
        -------
        _ClassifierOvR_torch
            The cloned classifier.
        """
        
        return _ClassifierOvR_torch(self.alpha, fit_intercept = self.fit_intercept, normalise = self.normalise, alpha_per_target = self.alpha_per_target)

class Classifier(sklearn.base.BaseEstimator):
    """Implements a ridge classifier.
    
    Parameters
    ----------
    alphas : Union[torch.Tensor, np.ndarray, float, int], default=1
        The penalties to use for estimation.
    method : str, default='OvR'
        The method to use for estimation (available: 'OvR', 'OvO').
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    normalise : bool, default=True
        Whether to normalise the data.
    alpha_per_target : bool, default=False
        Whether to use a different penalty for each target.
    
    Attributes
    ----------
    estimators_ : List[sklearn.base.BaseEstimator], optional (only for OvO)
        The estimators.
    estimator_ : sklearn.base.BaseEstimator, optional (only for OvR)
        The estimator.
    classes_ : Dict[int, int]
        The classes.
    intercept_ : Union[np.ndarray, torch.Tensor]
        The intercepts of the classifiers.
    coef_ : Union[np.ndarray, torch.Tensor]
        The coefficients of the classifiers.
    pattern_ : Union[np.ndarray, torch.Tensor]
        The patterns of the classifiers.
    
    Notes
    -----
    For multi-class classification, the One-vs-Rest strategy is used by default.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import Classifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y = True)
    >>> X, y = torch.from_numpy(X).to(torch.float32), torch.from_numpy(y).to(torch.float32)
    >>> clf = Classifier(alphas = torch.logspace(-5, 10, 20))
    >>> clf.fit(X, y)
    >>> clf.predict(X).shape
    torch.Size([150])
    """
    
    def __new__(self, alphas: Union[torch.Tensor, np.ndarray, float, int] = 1, method: str = 'OvR', **kwargs) -> sklearn.base.BaseEstimator:
        """Obtain a new classifier.
        
        Parameters
        ----------
        alphas : Union[torch.Tensor, np.ndarray, float, int], default=1
            The penalties to use for estimation.
        method: str, default='OvR'
            The method to use for multi-class classification (OvR for One-vs-Rest or OvO for One-vs-One).
        kwargs : Any
            Additional arguments.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            The classifier.
        """
        
        # check alphas
        if isinstance(alphas, float) | isinstance(alphas, int):
            alphas = torch.tensor([alphas])
        
        if isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        # determine estimator
        if isinstance(alphas, torch.Tensor) & (method == 'OvR'):
            return _ClassifierOvR_torch(alpha = alphas, **kwargs)
        elif isinstance(alphas, np.ndarray) & (method == 'OvR'):
            return _ClassifierOvR_numpy(alpha = alphas, **kwargs)
        elif isinstance(alphas, torch.Tensor) & (method == 'OvO'):
            return _ClassifierOvO_torch(alpha = alphas, **kwargs)
        elif isinstance(alphas, np.ndarray) & (method == 'OvO'):
            return _ClassifierOvO_numpy(alpha = alphas, **kwargs)
        
        raise ValueError(f'Unknown combination of method=`{method}` and alpha type `{type(alphas)}`.')

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        """Fit the estimator.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The features.
        y : Union[np.ndarray, torch.Tensor]
            The targets.
        """

        raise NotImplementedError('This method is not implemented in the base class.')

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The features.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The predictions.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The features.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The predictions.
        """

        raise NotImplementedError('This method is not implemented in the base class.')
    
    def clone(self):
        """Clone this class.
        
        Returns
        -------
        Decoder
            The cloned object.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')