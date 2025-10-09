'''
A collection of estimators for fitting cross-validated ridge regressions.
'''

import numpy as np
import torch
import sklearn
import scipy
import warnings

from ..math import kernel_linear, kernel_rbf, kernel_poly, kernel_sigmoid

from typing import Union, Optional, Any

KERNELS = dict(
    linear = kernel_linear,
    rbf = kernel_rbf,
    poly = kernel_poly,
    sigmoid = kernel_sigmoid
)

class _KernelRidgeCV_numpy(sklearn.base.BaseEstimator):
    """Obtain a new KernelRidgeCV using the numpy backend.
    
    Parameters
    ----------
    alphas : Union[np.ndarray, list, float, int], default=1.0
        Alpha penalties to test.
    kernel : str, default='linear'
        Kernel function to use (linear, poly, rbf, sigmoid).
    gamma : Union[float, str], default='auto'
        Gamma to use in kernel computation (or auto or scale).
    coef0 : float, default=1.0
        Coefficient zero to use in kernel computation.
    degree : float, default=3.0
        Degree of kernel to use.
    alpha_per_target : bool, default=False
        Should we fit one alpha per target?
    
    Attributes
    ----------
    alphas : Union[np.ndarray, list, float, int], default=1.0
        Alpha penalties to test.
    kernel : str, default='linear'
        Kernel function to use (linear, poly, rbf, sigmoid).
    gamma : Union[float, str], default='auto'
        Gamma to use in kernel computation (or auto or scale).
    coef0 : float, default=1.0
        Coefficient zero to use in kernel computation.
    degree : float, default=3.0
        Degree of kernel to use.
    alpha_per_target : bool, default=False
        Should we fit one alpha per target?
    X_ : np.ndarray
        Training data X of shape ``(n_samples, n_channels)``.
    A_dual_ : np.ndarray
        Chosen dual alpha of shape ``(n_samples, n_features)``.
    alpha_ : Union[float, np.ndarray]
        Chosen alpha penalties.
    coef_ : Optional[np.ndarray]
        If :py:attr:`~mvpy.estimators.KernelRidgeCV.kernel` is ``linear``, coefficients of shape ``(n_channels, n_features)``.
    """
    
    def __init__(self, alphas: Union[np.ndarray, list, float, int] = 1.0, kernel: str = 'linear', gamma: Union[float, str] = 'auto', coef0: float = 1.0, degree: float = 3.0, alpha_per_target: bool = False):
        """Obtain a new KernelRidgeCV using the numpy backend.
        
        Parameters
        ----------
        alphas : Union[np.ndarray, list, float, int], default=1.0
            Alpha penalties to test.
        kernel : str, default='linear'
            Kernel function to use (linear, poly, rbf, sigmoid).
        gamma : Union[float, str], default='auto'
            Gamma to use in kernel computation (or auto or scale).
        coef0 : float, default=1.0
            Coefficient zero to use in kernel computation.
        degree : float, default=3.0
            Degree of kernel to use.
        alpha_per_target : bool, default=False
            Should we fit one alpha per target?
        """
        
        # check alphas
        if isinstance(alphas, (int, float)):
            alphas = np.array([alphas])
        elif isinstance(alphas, list):
            alphas = np.array(alphas)
        
        self.alphas = alphas

        # check kernel 
        if kernel not in KERNELS:
            raise ValueError(f"Unknown kernel `{kernel}`. Expected one of {list(KERNELS.keys())}.")
        self.kernel = kernel
        self.kernel_ = KERNELS[kernel]

        # setup opts
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.alpha_per_target = alpha_per_target

        # setup placeholders
        self.X_ = None
        self.A_dual_ = None
        self.alpha_ = None
        self.coef_ = None
        self.intercept_ = None

    def compute_gamma_(self, X: np.ndarray) -> float:
        """Compute gamma.
        
        Parameters
        ----------
        X : np.ndarray
            The features of shape ``(n_samples, n_channels)``.
        
        Returns
        -------
        gamma : float
            Gamma parameter.
        """
        
        # check if we need to compute it
        if isinstance(self.gamma, str):
            # check method
            if self.gamma == 'scale':
                return (1.0 / (X.shape[1] * X.var()))
            elif self.gamma == 'auto':
                return (1.0 / X.shape[1])

            raise ValueError(f'Gamma method `{self.gamma}` unknown. Must be positive float or [\'scale\', \'auto\'].')

        # otherwise, use float
        gamma = float(self.gamma)
        
        # check gamma
        if gamma < 0:
            raise ValueError(f'Gamma must be positive float, but got {gamma}.')

        return gamma

    def fit(self, X: np.ndarray, y: np.ndarray) -> sklearn.base.BaseEstimator:
        """Fit the estimator.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape ``(n_samples, n_channels)``.
        y : np.ndarray
            Input features of shape ``(n_samples, n_features)``.
        
        Returns
        -------
        estimator : _KernelRidgeCV_numpy
            The fitted estimator.
        """
        
        # check shape of y
        if y.ndim == 1:
            y = y[:, None]
        
        # force contiguous data in X (speeds up kernels)
        X = np.ascontiguousarray(X)
                
        # compute gamma
        self.gamma_ = self.compute_gamma_(X)
        
        # compute gram matrix
        K = self.kernel_(X, X, self.gamma_, self.coef0, self.degree)
        
        # setup diagonal idx
        d = np.arange(K.shape[0])
        
        # setup dual
        A_duals = np.zeros((self.alphas.shape[0], K.shape[0], y.shape[1]))
        cv_loo = np.zeros((self.alphas.shape[0], K.shape[0], y.shape[1]))
        
        # loop over alphas
        for a_i, alpha in enumerate(self.alphas):
            # add alpha
            K[d,d] += alpha
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                
                try:
                    # solve efficiently using cholesky
                    A_duals[a_i] = scipy.linalg.solve(K, y, assume_a = "pos", overwrite_a = False)
                except np.linalg.LinAlgError:
                    # fall back to least squares
                    A_duals[a_i] = scipy.linalg.lstsq(K, y)[0]
            
            # compute (K + alphaI)^-1
            # technically, the better solution here would be
            # to use a cholesky -- but this runs into the same
            # speed issues described elsewhere. this seems
            # like a good enough compromise
            P = scipy.linalg.pinv(K)
            
            # subtract alpha
            K[d,d] -= alpha
            
            # compute H diagonal
            H_diag = (K @ P).diagonal()[:,None]
            
            # compute LOO squared errors
            cv_loo[a_i] = ((y - K @ A_duals[a_i]) / (1.0 - H_diag)) ** 2
        
        # check alpha kind
        if self.alpha_per_target:
            # select alpha per target
            best = cv_loo.mean(1).argmin(axis = 0)
            idx = np.arange(y.shape[1])
            A_dual = A_duals[best,:,idx]
        else:
            # select alpha overall
            best = cv_loo.mean((1, 2)).argmin()
            A_dual = A_duals[best]
        
        # set alpha
        alpha = self.alphas[best]
        
        # cache data
        self.X_ = X
        self.A_dual_ = A_dual
        self.alpha_ = alpha

        # if linear, expose coefficients
        if self.kernel == 'linear':
            self.coef_ = self.X_.T @ self.A_dual_
            self.intercept_ = None
        else:
            self.coef_ = None
            self.intercept_ = None
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions from the estimator.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape ``(n_samples, n_channels)``.
        
        Returns
        -------
        y_h : np.ndarray
            Predicted output features of shape ``(n_samples, n_features)``.
        """
        
        # check model fit
        if self.A_dual_ is None:
            raise RuntimeError("The KernelRidgeCV has not been fitted yet.")
        
        # check linear case:
        if self.kernel == 'linear':
            # here we can simply one-shot from
            # channel-transformed coefficients
            y_h = (X @ self.coef_)
        else:
            # otherwise, compute kernel
            K = self.kernel_(X, self.X_, self.gamma_, self.coef0, self.degree)
            y_h = K @ self.A_dual_
        
        # check resulting shape
        if y_h.shape[-1] == 1:
            return y_h.squeeze(-1)

        return y_h

    def clone(self) -> sklearn.base.BaseEstimator:
        """Obtain a clone of this estimator.
        
        Returns
        -------
        estimator : _KernelRidgeCV_numpy
            The cloned estimator.
        """
        
        return _KernelRidgeCV_numpy(
            alphas = self.alphas,
            kernel = self.kernel,
            gamma = self.gamma,
            coef0 = self.coef0,
            degree = self.degree,
            alpha_per_target = self.alpha_per_target,
        )

class _KernelRidgeCV_torch(sklearn.base.BaseEstimator):
    """Obtain a new KernelRidgeCV using the torch backend.
    
    Parameters
    ----------
    alphas : Union[torch.Tensor, list, float, int], default=1.0
        Alpha penalties to test.
    kernel : str, default='linear'
        Kernel function to use (linear, poly, rbf, sigmoid).
    gamma : Union[float, str], default='auto'
        Gamma to use in kernel computation (or auto or scale).
    coef0 : float, default=1.0
        Coefficient zero to use in kernel computation.
    degree : float, default=3.0
        Degree of kernel to use.
    alpha_per_target : bool, default=False
        Should we fit one alpha per target?
    
    Attributes
    ----------
    alphas : Union[torch.Tensor, list, float, int], default=1.0
        Alpha penalties to test.
    kernel : str, default='linear'
        Kernel function to use (linear, poly, rbf, sigmoid).
    gamma : Union[float, str], default='auto'
        Gamma to use in kernel computation (or auto or scale).
    coef0 : float, default=1.0
        Coefficient zero to use in kernel computation.
    degree : float, default=3.0
        Degree of kernel to use.
    alpha_per_target : bool, default=False
        Should we fit one alpha per target?
    X_ : torch.Tensor
        Training data X of shape ``(n_samples, n_channels)``.
    A_dual_ : torch.Tensor
        Chosen dual alpha of shape ``(n_samples, n_features)``.
    alpha_ : Union[float, torch.Tensor]
        Chosen alpha penalties.
    coef_ : Optional[torch.Tensor]
        If :py:attr:`~mvpy.estimators.KernelRidgeCV.kernel` is ``linear``, coefficients of shape ``(n_channels, n_features)``.
    """
    
    def __init__(self, alphas: Union[torch.Tensor, list, float, int] = 1.0, kernel: str = 'linear', gamma: Union[float, str] = 'auto', coef0: float = 1.0, degree: float = 3.0, alpha_per_target: bool = False):
        """Obtain a new KernelRidgeCV using the torch backend.
        
        Parameters
        ----------
        alphas : Union[torch.Tensor, list, float, int], default=1.0
            Alpha penalties to test.
        kernel : str, default='linear'
            Kernel function to use (linear, poly, rbf, sigmoid).
        gamma : Union[float, str], default='auto'
            Gamma to use in kernel computation (or auto or scale).
        coef0 : float, default=1.0
            Coefficient zero to use in kernel computation.
        degree : float, default=3.0
            Degree of kernel to use.
        alpha_per_target : bool, default=False
            Should we fit one alpha per target?
        """
        
        # check alphas
        if isinstance(alphas, (int, float)):
            alphas = torch.tensor([alphas])
        elif isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        self.alphas = alphas

        # check kernel 
        if kernel not in KERNELS:
            raise ValueError(f"Unknown kernel `{kernel}`. Expected one of {list(KERNELS.keys())}.")
        self.kernel = kernel
        self.kernel_ = KERNELS[kernel]

        # setup opts
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.alpha_per_target = alpha_per_target

        # setup placeholders
        self.X_ = None
        self.A_dual_ = None
        self.alpha_ = None
        self.coef_ = None
        self.intercept_ = None

    def compute_gamma_(self, X: torch.Tensor) -> float:
        """Compute gamma.
        
        Parameters
        ----------
        X : torch.Tensor
            The features of shape ``(n_samples, n_channels)``.
        
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
                return (1.0 / X.shape[1])

            raise ValueError(f'Gamma method `{self.gamma}` unknown. Must be positive float or [\'scale\', \'auto\'].')

        # otherwise, use float
        gamma = float(self.gamma)
        
        # check gamma
        if gamma < 0:
            raise ValueError(f'Gamma must be positive float, but got {gamma}.')

        return gamma

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> sklearn.base.BaseEstimator:
        """Fit the estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        y : torch.Tensor
            Input features of shape ``(n_samples, n_features)``.
        
        Returns
        -------
        estimator : _KernelRidgeCV_torch
            The fitted estimator.
        """
        
        # check shape of y
        if y.ndim == 1:
            y = y[:, None]
        
        # check device
        device = X.device
        
        if self.alphas.device != device:
            self.alphas = self.alphas.to(device)
        
        # compute gamma
        self.gamma_ = self.compute_gamma_(X)
        
        # compute gram matrix
        K = self.kernel_(X, X, self.gamma_, self.coef0, self.degree)
        
        # setup diagonal idx
        d = torch.arange(K.shape[0], device = device)
        
        # setup dual
        A_duals = torch.zeros((self.alphas.shape[0], K.shape[0], y.shape[1]), dtype = X.dtype, device = device)
        cv_loo = torch.zeros_like(A_duals)
        
        # loop over alphas
        for a_i, alpha in enumerate(self.alphas):
            # add alpha
            K[d,d] += alpha
            
            try:
                # torch.linalg.solve will happily solve
                # even when matrix is singular - this can
                # really hurt model performance and we may
                # want to consider checking cholesky first
                # and then going for torch.linalg.solve_ex
                # this presents a good compromise of speed
                # and stability
                _, info = torch.linalg.cholesky_ex(K)
                assert ~(info > 0)
                
                A_duals[a_i], info = torch.linalg.solve_ex(K, y)
                assert ~(info > 0)
            except:
                # fall back to least squares
                A_duals[a_i] = torch.linalg.lstsq(K, y).solution
            
            # compute (K + alphaI)^-1
            # technically, the better solution here would be
            # to use a cholesky -- but this runs into the same
            # speed issues described elsewhere. this seems
            # like a good enough compromise
            P = torch.linalg.pinv(K)
            
            # subtract alpha
            K[d,d] -= alpha
            
            # compute H diagonal
            H_diag = (K @ P).diagonal()[:,None]
            
            # compute LOO squared errors
            cv_loo[a_i] = ((y - K @ A_duals[a_i]) / (1.0 - H_diag)) ** 2
        
        # check alpha kind
        if self.alpha_per_target:
            # select alpha per target
            best = cv_loo.mean(1).argmin(dim = 0)
            idx = torch.arange(y.shape[1], device = device)
            A_dual = A_duals[best,:,idx]
        else:
            # select alpha overall
            best = cv_loo.mean((1, 2)).argmin()
            A_dual = A_duals[best]
                
        # set alpha
        alpha = self.alphas[best]
        
        # cache data
        self.X_ = X
        self.A_dual_ = A_dual
        self.alpha_ = alpha

        # if linear, expose coefficients
        if self.kernel == 'linear':
            self.coef_ = self.X_.t() @ self.A_dual_
            self.intercept_ = None
        else:
            self.coef_ = None
            self.intercept_ = None
        
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions from the estimator.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        
        Returns
        -------
        y_h : torch.Tensor
            Predicted output features of shape ``(n_samples, n_features)``.
        """
        
        # check model fit
        if self.A_dual_ is None:
            raise RuntimeError("The KernelRidgeCV has not been fitted yet.")
        
        # check linear case:
        if self.kernel == 'linear':
            # here we can simply one-shot from
            # channel-transformed coefficients
            y_h = (X @ self.coef_)
        else:
            # otherwise, compute kernel
            K = self.kernel_(X, self.X_, self.gamma_, self.coef0, self.degree)
            y_h = K @ self.A_dual_
        
        # check resulting shape
        if y_h.shape[-1] == 1:
            return y_h.squeeze(-1)

        return y_h

    def clone(self) -> sklearn.base.BaseEstimator:
        """Obtain a clone of this estimator.
        
        Returns
        -------
        estimator : _KernelRidgeCV_torch
            The cloned estimator.
        """
        
        return _KernelRidgeCV_torch(
            alphas = self.alphas,
            kernel = self.kernel,
            gamma = self.gamma,
            coef0 = self.coef0,
            degree = self.degree,
            alpha_per_target = self.alpha_per_target,
        )

class KernelRidgeCV(sklearn.base.BaseEstimator):
    """Implements a kernel ridge regression with cross-validation.
    
    Kernel ridge regression maps input data :math:`X` to output data :math:`y`
    through coefficients :math:`\\beta`:
    
    .. math::
        y = \\beta\\kappa + \\varepsilon
    
    where :math:`\\kappa` is some gram matrix of :math:`X` and solves for the model 
    :math:`\\beta` through:
    
    .. math::

        \\arg\\min_\\beta \\frac{1}{2}\\lvert\\lvert y - \\kappa\\beta\\rvert\\rvert_2^2 + \\frac{\\alpha_\\beta}{2}\\lvert\\lvert\\beta\\rvert\\rvert_\\kappa^2
    
    where :math:`\\alpha_\\beta` are penalties to test in LOO-CV which has a 
    convenient closed-form solution here:
    
    .. math::

        \\arg\\min_{\\alpha_\\beta} \\frac{1}{N} \\sum_{i = 1}^{N} \\left(\\frac{y - \\kappa\\beta_\\alpha}{1 - H_{\\alpha,ii}}\\right) \\qquad\\textrm{where}\\qquad
        H_{\\alpha,ii} = \\textrm{diag}\\left(\\kappa\\cdot\\left(\\kappa + \\alpha_\\beta I\\right)^{-1}\\right)
    
    In other words, this solves a ridge regression in the parameter space defined by 
    the kernel function :math:`\\kappa(X, X)`. This is convenient because, just like
    :py:class:`~mvpy.estimators.SVC`, it allows for non-parametric estimation. For
    example, :py:attr:`~mvpy.estimators.KernelRidgeCV.kernel` ``rbf`` may capture
    non-linearities in data that :py:class:`~mvpy.estimators.RidgeCV` cannot account
    for. The closed-form LOO-CV formula is evaluated at all values of 
    :py:attr:`~mvpy.estimatos.KernelRidgeCV.alphas` and the penalty minimising the
    mean-squared loss is automatically chosen. This is convenient because it is 
    faster than performing inner cross-validation to fine-tune penalties.
    
    As such, :py:class:`~mvpy.estimators.KernelRidgeCV` mirrors :py:class:`~mvpy.estimators.SVC`
    in its application of the kernel trick and the associated benefits. The key 
    difference here is that :py:class:`~mvpy.estimators.KernelRidgeCV` is fit
    using L2 regularised squared error, whereas :py:class:`~mvpy.estimators.SVC`
    is fit through sequential minimal optimisation or gradient ascent over hinge
    losses. In practice, this means that :py:class:`~mvpy.estimators.KernelRidgeCV` 
    is much faster--particularly when multiple values of :py:attr:`~mvpy.estimators.KernelRidgeCV.alphas` 
    are specified--but produces less sparse solutions that are not margin-based.
    
    For more information on kernel ridge regression, see [1]_ [2]_.
    
    Parameters
    ----------
    alphas : np.ndarray | torch.tensor | List | float | int, default=1.0
        Alpha penalties to test.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='linear'
        Kernel function to use.
    gamma : {float, 'auto', 'scale'}, default='auto'
        Gamma to use in kernel computation.
    coef0 : float, default=1.0
        Coefficient zero to use in kernel computation.
    degree : float, default=3.0
        Degree of kernel to use.
    alpha_per_target : bool, default=False
        Should we fit one alpha per target?
    
    Attributes
    ----------
    alphas : np.ndarray | torch.tensor | List | float | int, default=1.0
        Alpha penalties to test.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='linear'
        Kernel function to use.
    gamma : {float, 'auto', 'scale'}, default='auto'
        Gamma to use in kernel computation.
    coef0 : float, default=1.0
        Coefficient zero to use in kernel computation.
    degree : float, default=3.0
        Degree of kernel to use.
    alpha_per_target : bool, default=False
        Should we fit one alpha per target?
    X_ : np.ndarray | torch.Tensor
        Training data X of shape ``(n_samples, n_channels)``.
    A_dual_ : np.ndarray | torch.Tensor
        Chosen dual alpha of shape ``(n_samples, n_features)``.
    alpha_ : float | np.ndarray | torch.Tensor
        Chosen alpha penalties.
    coef_ : Optional[np.ndarray | torch.Tensor]
        If :py:attr:`~mvpy.estimators.KernelRidgeCV.kernel` is ``linear``, coefficients of shape ``(n_channels, n_features)``.
    
    See also
    --------
    mvpy.estimators.RidgeCV : Alternative ridge regression without kernel functions.
    mvpy.math.kernel_linear, mvpy.math.kernel_poly, mvpy.math.kernel_rbf, mvpy.math.kernel_sigmoid : Available kernel functions.
    
    Notes
    -----
    Coefficients :py:attr:`~mvpy.estimators.KernelRidgeCV.coef_` are available 
    only when :py:attr:`~mvpy.estimators.KernelRidgeCV.kernel` is ``linear``
    where primal weights can be computed from dual solutions:
    
    .. math::

        w = X^T\\beta
    
    For other kernel functions, coefficients are not interpretable and,
    therefore, not computed here.
    
    .. warning::
        For small values of :py:attr:`~mvpy.estimators.KernelRidgeCV.alphas`, kernel
        matrices may no longer be positive semidefinite. This means that, in many
        cases, model fitting may have to resort to least squares solutions, which
        can decrease through-put by an order of magnitude (or more). This issue
        is particularly prevalent in the numpy backend. Please consider this when
        choosing penalties.
    
    .. warning::
        This issue can also appear independently of :py:attr:`~mvpy.estimators.KernelRidgeCV.alphas`.
        For example, the gram matrix given :math:`X\sim\mathcal{N}(0, 1)` will already
        be rank-deficient if :math:`n\_samples\\geq n\_channels`. As is the case in ``sklearn``,
        this will lead to poor solving speed in the numpy backend. The torch backend
        is more robust to this. Please consider this when investigating your data
        prior to model fitting.
    
    References
    ----------
    .. [1] Murphy, K.P. (2012). Machine learning: A probabilistic perspective. MIT Press.
    .. [2] Nadaraya, E.A. (1964). On estimating regression. Theory of Probability and Its Applications, 9, 141-142. 10.1137/1109020
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import KernelRidgeCV
    >>> ß = torch.normal(0, 1, size = (5,))
    >>> X = torch.normal(0, 1, size = (240, 5))
    >>> y = ß @ X.T + torch.normal(0, 0.5, size = (X.shape[0],))
    >>> model = KernelRidgeCV().fit(X, y)
    >>> model.coef_
    """
    
    def __new__(self, alphas: Union[np.ndarray, torch.Tensor, list, float, int] = 1, kernel: str = 'linear', gamma: Union[float, str] = 'auto', coef0: float = 1.0, degree: float = 3.0, alpha_per_target: bool = False):
        """Obtain a new KernelRidgeCV estimator.
        
        Parameters
        ----------
        alphas : np.ndarray | torch.tensor | List | float | int, default=1.0
            Alpha penalties to test.
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='linear'
            Kernel function to use.
        gamma : {float, 'auto', 'scale'}, default='auto'
            Gamma to use in kernel computation.
        coef0 : float, default=1.0
            Coefficient zero to use in kernel computation.
        degree : float, default=3.0
            Degree of kernel to use.
        fit_intercept : bool, default=True
            Should we fit an intercept?
        centre_kernel : bool, default=True
            Should we centre the kernel?
        alpha_per_target : bool, default=False
            Should we fit one alpha per target?
        solver : {'auto', 'svd', 'cholesky'}, default='auto'
            Which solver to use?
        """
        
        # check alphas
        if isinstance(alphas, int) | isinstance(alphas, float):
            alphas = torch.tensor([alphas])
        
        if isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        # check model type
        if isinstance(alphas, torch.Tensor):
            return _KernelRidgeCV_torch(alphas = alphas, kernel = kernel, gamma = gamma, coef0 = coef0, degree = degree, alpha_per_target = alpha_per_target)
        elif isinstance(alphas, np.ndarray):
            return _KernelRidgeCV_numpy(alphas = alphas, kernel = kernel, gamma = gamma, coef0 = coef0, degree = degree, alpha_per_target = alpha_per_target)
        
        raise ValueError(f'Alphas should be of type np.ndarray or torch.tensor, but got {type(alphas)}.')
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> "KernelRidgeCV":
        """Fit the estimator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        y : np.ndarray | torch.Tensor
            Input features of shape ``(n_samples, n_features)``.
        
        Returns
        -------
        estimator : KernelRidgeCV
            The fitted estimator.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Make predictions from the estimator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        
        Returns
        -------
        y_h : np.ndarray | torch.Tensor
            Predicted output features of shape ``(n_samples, n_features)``.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def clone(self) -> "KernelRidgeCV":
        """Make a clone of this class.
        
        Returns
        -------
        estimator : KernelRidgeCV
            A clone of this class.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')