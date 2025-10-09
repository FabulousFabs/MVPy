'''
A collection of estimators for support vector classification.
'''

import numpy as np
import torch
import sklearn

from .classifier import _Classifier_numpy, _Classifier_torch
from ..preprocessing.labelbinariser import _LabelBinariser_numpy, _LabelBinariser_torch
from ..preprocessing.scaler import _Scaler_numpy, _Scaler_torch
from ..math import kernel_linear, kernel_rbf, kernel_poly, kernel_sigmoid

from typing import Union, Any

KERNELS = dict(
    linear = kernel_linear,
    rbf = kernel_rbf,
    poly = kernel_poly,
    sigmoid = kernel_sigmoid
)

class _SVC_numpy(sklearn.base.BaseEstimator):
    r"""Implements a support vector classifier with numpy backend.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularisation strength is inversely related to C.
    kernel : str, default='linear'
        Which kernel function should we use (linear, poly, rbf, sigmoid)?
    gamma : Union[str, float], default='scale'
        What gamma to use for poly, rbf and sigmoid. Available methods are scale or auto, or positive float.
    coef0 : float, default=0.0
        What offset to use for poly and sigmoid.
    degree : float, default=3.0
        What degree polynomial to use (if any).
    tol : float, default=1e-3
        Tolerance over maximum update step (i.e., when maximal gradient < tol, early stopping is triggered).
    lr : float, default=1e-3
        The learning rate.
    max_iter : int, default=1000
        The maximum number of iterations to perform while fitting, or -1 to disable.
    
    Attributes
    ----------
    C : float, default=1.0
        Regularisation strength is inversely related to C.
    kernel : str, default='linear'
        Which kernel function should we use (linear, poly, rbf, sigmoid)?
    gamma : Union[str, float], default='scale'
        What gamma to use for poly, rbf and sigmoid. Available methods are scale or auto, or positive float.
    coef0 : float, default=0.0
        What offset to use for poly and sigmoid.
    degree : float, default=3.0
        What degree polynomial to use (if any).
    tol : float, default=1e-3
        Tolerance over maximum update step (i.e., when maximal gradient < tol, early stopping is triggered).
    lr : float, default=1e-3
        The learning rate.
    max_iter : int, default=1000
        The maximum number of iterations to perform while fitting, or -1 to disable.
    X_train_ : Union[np.ndarray, torch.Tensor]
        A clone of the training data used internally for kernel estimation.
    A_ : Union[np.ndarray, torch.Tensor]
        A clone of the alpha data used internally for kernel estimation.
    gamma_ : float
        Estimated gamma parameter.
    eps_ : float, default=1e-12
        Error margin for support vectors used internally.
    w_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, estimated weights.
    p_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, estimated patterns.
    intercept_ : Union[np.ndarray, torch.Tensor]
        The intercept vector.
    coef_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, the coefficients of the model.
    pattern_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, the patterns used by the model.
    binariser_ : LabelBinariser
        The binariser used internally.
    scaler_ : Scaler
        The scaler used internally.
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'linear', gamma: Union[str, float] = 'scale', coef0: float = 0.0, degree: float = 3.0, tol: float = 1e-3, lr: float = 1e-3, max_iter: int = 1000):
        """Obtain a support vector classifier.
        
        Parameters
        ----------
        C : float, default=1.0
            Regularisation strength is inversely related to C.
        kernel : str, default='linear'
            Which kernel function should we use (linear, poly, rbf, sigmoid)?
        gamma : Union[str, float], default='scale'
            What gamma to use for poly, rbf and sigmoid. Available methods are scale or auto, or positive float.
        coef0 : float, default=0.0
            What offset to use for poly and sigmoid.
        degree : float, default=3.0
            What degree polynomial to use (if any).
        tol : float, default=1e-3
            Tolerance over maximum update step (i.e., when maximal gradient < tol, early stopping is triggered).
        lr : float, default=1e-3
            The learning rate.
        max_iter : int, default=1000
            The maximum number of iterations to perform while fitting, or -1 to disable.
        """
        
        # setup args
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.tol = tol
        self.lr = lr
        self.max_iter = max_iter
        
        # setup kernel
        if kernel not in KERNELS:
            raise ValueError(f'Kernel `{kernel}` unknown. Available: {list(KERNELS.keys())}.')
        self.kernel_ = KERNELS[kernel]
        
        # setup other internals
        self.X_train_ = None
        self.A_ = None
        self.alpha_ = None
        self.gamma_ = None
        self.eps_ = 1e-12
        self.w_ = None
        self.p_ = None
        self.intercept_ = None
        
        # setup binariser
        self.binariser_ = _LabelBinariser_numpy(
            neg_label = -1.0, 
            pos_label = 1.0
        )
        
        # setup scaler
        self.scaler_ = _Scaler_numpy()
    
    @property
    def coef_(self) -> np.ndarray:
        """Grab coefficients used for classification. Only available for linear kernels.
        
        Returns
        -------
        coef_ : np.ndarray
            The coefficients used for classification (n_channels, n_classes).
        """
        
        if self.kernel != 'linear':
            raise AttributeError(f'coef_ can only be computed for linear kernels.')

        return self.w_

    @property
    def pattern_(self) -> np.ndarray:
        """Grab patterns used for classification. Only available for linear kernels.
        
        Returns
        -------
        coef_ : np.ndarray
            The patterns used for classification (n_channels, n_classes).
        """
        
        if self.kernel != 'linear':
            raise AttributeError(f'pattern_ can only be computed for linear kernels.')

        return self.p_
    
    def compute_gamma_(self, X: np.ndarray) -> float:
        """Compute gamma.
        
        Parameters
        ----------
        X : np.ndarray
            The features of shape (n_samples, n_channels).
        
        Returns
        -------
        gamma : float
            Gamma parameter.
        """
        
        # check if we need to compute it
        if isinstance(self.gamma, str):
            # check method
            if self.gamma == 'scale':
                return 1.0 / (X.shape[1] * X.var())
            elif self.gamma == 'auto':
                return 1.0 / X.shape[1]

            raise ValueError(f'Gamma method `{self.gamma}` unknown. Must be positive float or [\'scale\', \'auto\'].')

        # otherwise, use float
        gamma = float(self.gamma)
        
        # check gamma
        if gamma < 0:
            raise ValueError(f'Gamma must be positive float, but got {gamma}.')

        return gamma

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SVC_numpy":
        """Fit the estimator.

        Parameters
        ----------
        X : np.ndarray
            The features of shape (n_samples, n_channels).
        y : np.ndarray
            The targets of shape (n_samples[, n_features]).
        
        Returns
        -------
        clf : Classifier
            The classifier.
        """
        
        # check shape of X
        if len(X.shape) != 2:
            raise ValueError(f'X must be of shape (n_samples, n_features), but got {X.shape}.')
        
        # check shape of y
        if len(y.shape) == 1:
            y = y[:, None]
        
        # grab dimensions
        n_samples, n_features = X.shape
        
        # transform labels
        L = self.binariser_.fit_transform(y)
        n_classes = L.shape[1]
        
        # scale data
        X = self.scaler_.fit_transform(X)
        
        # setup alpha and intercepts
        self.alpha_ = np.random.normal(size = (X.shape[0], L.shape[1]))
        self.intercept_ = np.zeros((n_classes,))
        
        # compute gamma
        self.gamma_ = self.compute_gamma_(X)
        
        # compute gram matrix
        K = self.kernel_(X, X, self.gamma_, self.coef0, self.degree)
        
        # setup tracking
        last_grad = -np.inf
        it = 0
        
        while True:
            # compute gradient
            grad = 1.0 - L * (K @ (self.alpha_ * L))
            
            # update alpha
            self.alpha_ = np.clip(
                self.alpha_ + self.lr * grad,
                0.0,
                self.C
            )
            
            # if desired, stop early if exceeding maximum iterations
            if (self.max_iter > 0) and (it >= self.max_iter):
                break
            
            # otherwise, check gradients and tolerance
            max_grad = np.abs(grad).max()
            
            if abs(max_grad - last_grad) < self.tol:
                break
            
            # update tallies
            it += 1
            last_grad = max_grad
        
        # compute bias per class
        A = self.alpha_ * L
        f = K @ A
        
        for i in range(n_classes):
            # select vectors
            sv = (self.alpha_[:,i] > self.eps_) \
                  & (self.alpha_[:,i] < (self.C - self.eps_))
            
            if np.any(sv):
                # compute intercept from valid vectors
                self.intercept_[i] = np.mean(L[sv,i] - f[sv,i])
            else:
                # fallback to best available
                idx = np.argmax(self.alpha_[:,i])
                self.intercept_[i] = L[idx,i] - f[idx,i]
        
        # cache X and A
        self.X_train_ = X.copy()
        self.A_ = A.copy()
        
        # if linear, compute weights and patterns
        if self.kernel == 'linear':
            # compute weights
            self.w_ = X.T @ A
            
            # compute covariance of X
            X = (X - X.mean(axis = 0, keepdims = True))
            S_X = np.cov(X.T)
            
            # compute precision of y
            L = (L - L.mean(axis = 0, keepdims = True))
            P_L = np.linalg.pinv(
                np.cov(L.T)
            )
            
            # compute pattern
            self.p_ = S_X.dot(self.w_).dot(P_L)
            
            # transform both
            self.w_ = self.scaler_.inverse_transform(self.w_.T).T
            self.p_ = self.scaler_.inverse_transform(self.p_.T).T
        
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
        if self.alpha_ is None or self.X_train_ is None or self.A_ is None or self.intercept_ is None:
            raise ValueError(f'The SVC has not been fitted yet.')

        # compute decision values
        return (
            self.intercept_[None,:] + self.kernel_(self.X_train_, self.scaler_.transform(X), self.gamma_, self.coef0, self.degree).T @ self.A_
        )
    
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
        
        # compute decision values
        df = self.decision_function(X)
        
        # prepare y
        y = np.zeros_like(df)
        
        # loop over our features
        for i in range(self.binariser_.n_features_):
            # find start and end of feature
            s = self.binariser_.C_[i]
            e = s + len(self.binariser_.classes_[i])
            
            # find maximum for feature
            idx = np.argmax(df[:,s:e], axis = 1)
            
            # set max values
            y[np.arange(idx.shape[0]),s + idx] = 1.0
        
        # return labels
        return self.binariser_.inverse_transform(y)
    
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
    
    def clone(self) -> "_SVC_numpy":
        """Clone this class.
        
        Returns
        -------
        clf : _SVC_numpy
            The cloned object.
        """
        
        return _SVC_numpy(
            C = self.C,
            kernel = self.kernel,
            gamma = self.gamma,
            coef0 = self.coef0,
            degree = self.degree,
            tol = self.tol,
            lr = self.lr,
            max_iter = self.max_iter
        )

class _SVC_torch(sklearn.base.BaseEstimator):
    r"""Implements a support vector classifier with torch backend.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularisation strength is inversely related to C.
    kernel : str, default='linear'
        Which kernel function should we use (linear, poly, rbf, sigmoid)?
    gamma : Union[str, float], default='scale'
        What gamma to use for poly, rbf and sigmoid. Available methods are scale or auto, or positive float.
    coef0 : float, default=0.0
        What offset to use for poly and sigmoid.
    degree : float, default=3.0
        What degree polynomial to use (if any).
    tol : float, default=1e-3
        Tolerance over maximum update step (i.e., when maximal gradient < tol, early stopping is triggered).
    lr : float, default=1e-3
        The learning rate.
    max_iter : int, default=1000
        The maximum number of iterations to perform while fitting, or -1 to disable.
    
    Attributes
    ----------
    C : float, default=1.0
        Regularisation strength is inversely related to C.
    kernel : str, default='linear'
        Which kernel function should we use (linear, poly, rbf, sigmoid)?
    gamma : Union[str, float], default='scale'
        What gamma to use for poly, rbf and sigmoid. Available methods are scale or auto, or positive float.
    coef0 : float, default=0.0
        What offset to use for poly and sigmoid.
    degree : float, default=3.0
        What degree polynomial to use (if any).
    tol : float, default=1e-3
        Tolerance over maximum update step (i.e., when maximal gradient < tol, early stopping is triggered).
    lr : float, default=1e-3
        The learning rate.
    max_iter : int, default=1000
        The maximum number of iterations to perform while fitting, or -1 to disable.
    X_train_ : Union[np.ndarray, torch.Tensor]
        A clone of the training data used internally for kernel estimation.
    A_ : Union[np.ndarray, torch.Tensor]
        A clone of the alpha data used internally for kernel estimation.
    gamma_ : float
        Estimated gamma parameter.
    eps_ : float, default=1e-12
        Error margin for support vectors used internally.
    w_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, estimated weights.
    p_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, estimated patterns.
    intercept_ : Union[np.ndarray, torch.Tensor]
        The intercept vector.
    coef_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, the coefficients of the model.
    pattern_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, the patterns used by the model.
    binariser_ : LabelBinariser
        The binariser used internally.
    scaler_ : Scaler
        The scaler used internally.
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'linear', gamma: Union[str, float] = 'scale', coef0: float = 0.0, degree: float = 3.0, tol: float = 1e-3, lr: float = 1e-3, max_iter: int = 1000):
        """Obtain a support vector classifier.
        
        Parameters
        ----------
        C : float, default=1.0
            Regularisation strength is inversely related to C.
        kernel : str, default='linear'
            Which kernel function should we use (linear, poly, rbf, sigmoid)?
        gamma : Union[str, float], default='scale'
            What gamma to use for poly, rbf and sigmoid. Available methods are scale or auto, or positive float.
        coef0 : float, default=0.0
            What offset to use for poly and sigmoid.
        degree : float, default=3.0
            What degree polynomial to use (if any).
        tol : float, default=1e-3
            Tolerance over maximum update step (i.e., when maximal gradient < tol, early stopping is triggered).
        lr : float, default=1e-3
            The learning rate.
        max_iter : int, default=1000
            The maximum number of iterations to perform while fitting, or -1 to disable.
        """
        
        # setup args
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.tol = tol
        self.lr = lr
        self.max_iter = max_iter
        
        # setup kernel
        if kernel not in KERNELS:
            raise ValueError(f'Kernel `{kernel}` unknown. Available: {list(KERNELS.keys())}.')
        self.kernel_ = KERNELS[kernel]
        
        # setup other internals
        self.X_train_ = None
        self.A_ = None
        self.alpha_ = None
        self.gamma_ = None
        self.eps_ = 1e-12
        self.w_ = None
        self.p_ = None
        self.intercept_ = None
        
        # setup binariser
        self.binariser_ = _LabelBinariser_torch(
            neg_label = -1.0, 
            pos_label = 1.0
        )
        
        # setup scaler
        self.scaler_ = _Scaler_torch()
    
    @property
    def coef_(self) -> torch.Tensor:
        """Grab coefficients used for classification. Only available for linear kernels.
        
        Returns
        -------
        coef_ : torch.Tensor
            The coefficients used for classification (n_channels, n_classes).
        """
        
        if self.kernel != 'linear':
            raise AttributeError(f'coef_ can only be computed for linear kernels.')

        return self.w_

    @property
    def pattern_(self) -> torch.Tensor:
        """Grab patterns used for classification. Only available for linear kernels.
        
        Returns
        -------
        pattern_ : torch.Tensor
            The pattern used for classification (n_channels, n_classes).
        """
        
        if self.kernel != 'linear':
            raise AttributeError(f'pattern_ can only be computed for linear kernels.')

        return self.p_
    
    def compute_gamma_(self, X: torch.Tensor) -> float:
        """Compute gamma.
        
        Parameters
        ----------
        X : torch.Tensor
            The features of shape (n_samples, n_channels).
        
        Returns
        -------
        gamma : float
            Gamma parameter.
        """
        
        # check if we need to compute it
        if isinstance(self.gamma, str):
            # check method
            if self.gamma == 'scale':
                return (1.0 / (X.shape[1] * X.var())).item()
            elif self.gamma == 'auto':
                return (1.0 / X.shape[1]).item()

            raise ValueError(f'Gamma method `{self.gamma}` unknown. Must be positive float or [\'scale\', \'auto\'].')

        # otherwise, use float
        gamma = float(self.gamma)
        
        # check gamma
        if gamma < 0:
            raise ValueError(f'Gamma must be positive float, but got {gamma}.')

        return gamma

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "_SVC_torch":
        """Fit the estimator.

        Parameters
        ----------
        X : torch.Tensor
            The features of shape (n_samples, n_channels).
        y : torch.Tensor
            The targets of shape (n_samples[, n_features]).
        
        Returns
        -------
        clf : Classifier
            The classifier.
        """
        
        # check shape of X
        if len(X.shape) != 2:
            raise ValueError(f'X must be of shape (n_samples, n_features), but got {X.shape}.')
        
        # check shape of y
        if len(y.shape) == 1:
            y = y[:, None]
        
        # grab dimensions
        n_samples, n_features = X.shape
        
        # transform labels
        L = self.binariser_.fit_transform(y).to(X.dtype).to(X.device)
        n_classes = L.shape[1]
        
        # scale data
        X = self.scaler_.fit_transform(X)
        
        # setup alpha and intercepts
        self.alpha_ = torch.randn(X.shape[0], L.shape[1], dtype = X.dtype, device = X.device)
        self.intercept_ = torch.zeros((n_classes,), dtype = X.dtype, device = X.device)
        
        # compute gamma
        self.gamma_ = self.compute_gamma_(X)
        
        # compute gram matrix
        K = self.kernel_(X, X, self.gamma_, self.coef0, self.degree)
        
        # setup tracking
        last_grad = -torch.inf
        it = 0
        
        while True:
            # compute gradient
            grad = 1.0 - L * (K @ (self.alpha_ * L))
            
            # update alpha
            self.alpha_ = torch.clip(
                self.alpha_ + self.lr * grad,
                0.0,
                self.C
            )
            
            # if desired, stop early if exceeding maximum iterations
            if (self.max_iter > 0) and (it >= self.max_iter):
                break
            
            # otherwise, check gradients and tolerance
            max_grad = grad.abs().max().item()
            
            if abs(max_grad - last_grad) < self.tol:
                break
            
            # update tallies
            it += 1
            last_grad = max_grad
        
        # compute bias per class
        A = self.alpha_ * L
        f = K @ A
        
        for i in range(n_classes):
            # select vectors
            sv = (self.alpha_[:,i] > self.eps_) \
                  & (self.alpha_[:,i] < (self.C - self.eps_))
            
            if torch.any(sv):
                # compute intercept from valid vectors
                self.intercept_[i] = torch.mean(L[sv,i] - f[sv,i])
            else:
                # fallback to best available
                idx = torch.argmax(self.alpha_[:,i])
                self.intercept_[i] = L[idx,i] - f[idx,i]
        
        # cache X and A
        self.X_train_ = X.clone()
        self.A_ = A.clone()
        
        # if linear, compute weights and patterns
        if self.kernel == 'linear':
            # compute weights
            self.w_ = X.T @ A
            
            # compute covariance of X
            X = (X - X.mean(0, keepdim = True))
            S_X = torch.cov(X.T)
            
            # compute precision of y
            L = (L - L.mean(0, keepdim = True))
            P_L = torch.linalg.pinv(
                torch.cov(L.T)
            )
            
            # compute pattern
            self.p_ = S_X.mm(self.w_).mm(P_L)
            
            # transform both
            self.w_ = self.scaler_.inverse_transform(self.w_.t()).t()
            self.p_ = self.scaler_.inverse_transform(self.p_.t()).t()
        
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
        if self.alpha_ is None or self.X_train_ is None or self.A_ is None or self.intercept_ is None:
            raise ValueError(f'The SVC has not been fitted yet.')

        # compute decision values
        return (
            self.intercept_[None,:] + self.kernel_(self.X_train_, self.scaler_.transform(X), self.gamma_, self.coef0, self.degree).T @ self.A_
        )
    
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
        
        # compute decision values
        df = self.decision_function(X)
        
        # prepare y
        y = torch.zeros_like(df)
        
        # loop over our features
        for i in range(self.binariser_.n_features_):
            # find start and end of feature
            s = self.binariser_.C_[i].long().item()
            e = s + len(self.binariser_.classes_[i])
            
            # find maximum for feature
            idx = torch.argmax(df[:,s:e], dim = 1)
            
            # set max values
            y[torch.arange(idx.shape[0]),s + idx] = 1.0
        
        # return labels
        return self.binariser_.inverse_transform(y)
    
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
    
    def clone(self) -> "_SVC_torch":
        """Clone this class.
        
        Returns
        -------
        clf : _SVC_torch
            The cloned object.
        """
        
        return _SVC_torch(
            C = self.C,
            kernel = self.kernel,
            gamma = self.gamma,
            coef0 = self.coef0,
            degree = self.degree,
            tol = self.tol,
            lr = self.lr,
            max_iter = self.max_iter
        )
        
class SVC(sklearn.base.BaseEstimator):
    """Implements a support vector classifier.
    
    Parameters
    ----------
    method : str, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    C : float, default=1.0
        Regularisation strength is inversely related to C.
    kernel : str, default='linear'
        Which kernel function should we use (linear, poly, rbf, sigmoid)?
    gamma : Union[str, float], default='scale'
        What gamma to use for poly, rbf and sigmoid. Available methods are scale or auto, or positive float.
    coef0 : float, default=0.0
        What offset to use for poly and sigmoid.
    degree : float, default=3.0
        What degree polynomial to use (if any).
    tol : float, default=1e-3
        Tolerance over maximum update step (i.e., when maximal gradient < tol, early stopping is triggered).
    lr : float, default=1e-3
        The learning rate.
    max_iter : int, default=1000
        The maximum number of iterations to perform while fitting, or -1 to disable.
    
    Attributes
    ----------
    method : str, default='OvR'
        For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
    C : float, default=1.0
        Regularisation strength is inversely related to C.
    kernel : str, default='linear'
        Which kernel function should we use (linear, poly, rbf, sigmoid)?
    gamma : Union[str, float], default='scale'
        What gamma to use for poly, rbf and sigmoid. Available methods are scale or auto, or positive float.
    coef0 : float, default=0.0
        What offset to use for poly and sigmoid.
    degree : float, default=3.0
        What degree polynomial to use (if any).
    tol : float, default=1e-3
        Tolerance over maximum update step (i.e., when maximal gradient < tol, early stopping is triggered).
    lr : float, default=1e-3
        The learning rate.
    max_iter : int, default=1000
        The maximum number of iterations to perform while fitting, or -1 to disable.
    X_train_ : Union[np.ndarray, torch.Tensor]
        A clone of the training data used internally for kernel estimation.
    A_ : Union[np.ndarray, torch.Tensor]
        A clone of the alpha data used internally for kernel estimation.
    gamma_ : float
        Estimated gamma parameter.
    eps_ : float, default=1e-12
        Error margin for support vectors used internally.
    w_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, estimated weights.
    p_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, estimated patterns.
    intercept_ : Union[np.ndarray, torch.Tensor]
        The intercept vector.
    coef_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, the coefficients of the model.
    pattern_ : Union[np.ndarray, torch.Tensor]
        If linear kernel, the patterns used by the model.
    binariser_ : LabelBinariser
        The binariser used internally.
    scaler_ : Scaler
        The scaler used internally.
    
    Notes
    -----
    This implementation, unlike ones in sklearn, for example, uses gradient ascent over vectorised features. Therefore, please do not expect identitical solutions. Early stopping is triggered by:
    
    .. math::

        \max\lvert grad_i\rvert - \max\lvert grad_{i-1}\rvert < \vartheta
    
    where :math:`\vartheta` is the tolerance. Note that this also diverges from sklearn's implementation which is violation-based.
    
    This is likely to change in the future. Ideally, we will want to use an SMO approach, which should be much faster than the current full gradient ascent. It is a bit of a pickle, because SVC is difficult to optimise for GPUs. There are also new interesting directions like fitting parameters (e.g., gamma) directly, which might be interesting here. For more information, see [1]_. We will revisit this implementation.
    
    Note also that coefficients are only interpretable in the linear case, and therefore this class automatically prevents access in other cases. Given a linear kernel, however, this class will also automatically compute the patterns used for classification, available as pattern_. For more information, see [2]_.
    
    Note also that this class will rarely be public-facing, as at runtime it is wrapped in a Classifier class which handles OvR/OvO modes. By default, OvR will make predictions based on maximum decision values, whereas OvO will make voting-based decisions between all classifiers.
    
    References
    ----------
    .. [1] Luo, L., Yang, Q., Peng, H., Wang, Y., & Chen, Z. (2025). MaxMin-L2-SVC-NCH: A novel approach for support vector classifier training and parameter selection. arXiv. 10.48550/arXiv.2307.07343
    .. [2] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.D., Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96-110. 10.1016/j.neuroimage.2013.10.067
    
    Examples
    --------
    First, let's look at a case where we have one feature that has two classes.
    
    >>> import torch
    >>> from mvpy.estimators import SVC
    >>> from sklearn.datasets import make_circles
    >>> X, y = make_circles(noise = 0.3)
    >>> X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    >>> clf = SVC(kernel = 'rbf').fit(X, y)
    >>> y_h = clf.predict(X)
    >>> mv.math.accuracy(y_h.squeeze(), y)
    tensor(0.6700)
    
    Second, let's look at a case where we have one feature that has three classes.
    
    >>> import torch
    >>> from mvpy.estimators import SVC
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y = True)
    >>> X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    >>> clf = SVC(kernel = 'rbf').fit(X, y)
    >>> y_h = clf.predict(X)
    >>> mv.math.accuracy(y_h.squeeze(), y)
    tensor(0.9733)
    
    Third, let's look at a case where we have two features with a variable number of classes.
    
    >>> import torch
    >>> from mvpy.estimators import SVC
    >>> from sklearn.datasets import make_classification
    >>> X0, y0 = make_classification(n_classes = 3, n_informative = 6)
    >>> X1, y1 = make_classification(n_classes = 4, n_informative = 8)
    >>> X = torch.from_numpy(np.concatenate((X0, X1), axis = -1)).float()
    >>> y = torch.from_numpy(np.stack((y0, y1), axis = -1)).float()
    >>> clf = SVC(kernel = 'rbf').fit(X, y)
    >>> y_h = clf.predict(X)
    >>> mv.math.accuracy(y_h.T, y.T)
    tensor([1.000, 0.9800])
    """
    
    def __init__(self, method: str = 'OvR', C: float = 1.0, kernel: str = 'linear', gamma: Union[str, float] = 'scale', coef0: float = 0.0, degree: float = 3.0, tol: float = 1e-3, lr: float = 1e-3, max_iter: int = 1000) -> sklearn.base.BaseEstimator:
        """Obtain a support vector classifier.
        
        Parameters
        ----------
        method : str, default='OvR'
            For multiclass problems, which method should we use? One-versus-one (OvO) or one-versus-rest (OvR)?
        C : float, default=1.0
            Regularisation strength is inversely related to C.
        kernel : str, default='linear'
            Which kernel function should we use (linear, poly, rbf, sigmoid)?
        gamma : Union[str, float], default='scale'
            What gamma to use for poly, rbf and sigmoid. Available methods are scale or auto, or positive float.
        coef0 : float, default=0.0
            What offset to use for poly and sigmoid.
        degree : float, default=3.0
            What degree polynomial to use (if any).
        tol : float, default=1e-3
            Tolerance over maximum update step (i.e., when maximal gradient < tol, early stopping is triggered).
        lr : float, default=1e-3
            The learning rate.
        max_iter : int, default=1000
            The maximum number of iterations to perform while fitting, or -1 to disable.
        """
        
        self.method = method
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.tol = tol
        self.lr = lr
        self.max_iter = max_iter
    
    def _get_estimator(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> sklearn.base.BaseEstimator:
        """Obtain the wrapper and estimator for this SVC.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            Input data of shape (n_samples, n_channels).
        y : Union[np.ndarray, torch.Tensor]
            Input labels of shape (n_samples[, n_features]).
        
        Returns
        -------
        clf : Classifier
            The classifier.
        """
        
        # setup arguments
        arguments = []
        kwarguments = dict(
            C = self.C,
            kernel = self.kernel,
            gamma = self.gamma,
            coef0 = self.coef0,
            degree = self.degree,
            tol = self.tol,
            lr = self.lr,
            max_iter = self.max_iter
        )
        
        # make estimator
        if isinstance(X, torch.Tensor) & isinstance(y, torch.Tensor):
            return _Classifier_torch(
                _SVC_torch,
                method = self.method,
                arguments = arguments,
                kwarguments = kwarguments
            )
        elif isinstance(X, np.ndarray) & isinstance(y, np.ndarray):
            return _Classifier_numpy(
                _SVC_numpy,
                method = self.method,
                arguments = arguments,
                kwarguments = kwarguments
            )
        
        raise TypeError(f'`X` and `y` must be either torch.Tensor or np.ndarray, but got {type(X)} and {type(y)}.')
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> sklearn.base.BaseEstimator:
        """Fit the estimator.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The features of shape (n_samples, n_channels).
        y : Union[np.ndarray, torch.Tensor]
            The targets of shape (n_samples[, n_features]).
        
        Returns
        -------
        clf : Classifier
            The classifier.
        """

        return self._get_estimator(X, y).fit(X, y)
    
    def decision_function(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The features (n_samples, n_channels).

        Returns
        -------
        df : Union[np.ndarray, torch.Tensor]
            The predictions of shape (n_samples, n_classes).
        """

        raise NotImplementedError('This method is not implemented in the base class.')

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The features (n_samples, n_channels).
        
        Returns
        -------
        y_h : Union[np.ndarray, torch.Tensor]
            The predictions of shape (n_samples, n_features).
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The features (n_samples, n_channels).

        Returns
        -------
        df : Union[np.ndarray, torch.Tensor]
            The predictions of shape (n_samples, n_classes).
        """

        raise NotImplementedError('This method is not implemented in the base class.')
    
    def to_torch(self) -> sklearn.base.BaseEstimator:
        """Obtain the estimator with torch as backend.
        
        Returns
        -------
        clf : _Classifier_torch
            The estimator.
        """
        
        return self._get_estimator(torch.tensor([1.0]), torch.tensor([1.0]))
    
    def to_numpy(self) -> sklearn.base.BaseEstimator:
        """Obtain the estimator with numpy as backend.
        
        Returns
        -------
        clf : _Classifier_numpy
            The estimator.
        """
        
        return self._get_estimator(np.array([1.0]), np.array([1.0]))
    
    def clone(self) -> "SVC":
        """Clone this class.
        
        Returns
        -------
        clf : SVC
            The cloned object.
        """
        
        return SVC(
            method = self.method,
            C = self.C,
            kernel = self.kernel,
            gamma = self.gamma,
            coef0 = self.coef0,
            degree = self.degree,
            tol = self.tol,
            lr = self.lr,
            max_iter = self.max_iter
        )
    
    def copy(self) -> "SVC":
        """Clone this class.
        
        Returns
        -------
        clf : SVC
            The cloned object.
        """
        
        return self.clone()