'''
A collection of estimators for fitting cross-validated ridge regressions.
'''

import numpy as np
import torch
import sklearn

from typing import Union, Dict, Tuple, Optional, Any

from sklearn.linear_model import RidgeCV as _RidgeCV_numpy

from .. import metrics

class _RidgeCV_torch(sklearn.base.BaseEstimator):
    """Implements RidgeCV using torch as our backend.
    
    Attributes
    ----------
    alpha_ : torch.Tensor
        The penalties used for estimation.
    intercept_ : torch.Tensor
        The intercepts.
    coef_ : torch.Tensor
        The coefficients.
    metric_ : mvpy.metrics.r2
        The default metric to use.
    """
    
    def __init__(self, alphas: Union[torch.Tensor, list, float, int] = 1, fit_intercept: bool = True, alpha_per_target: bool = False):
        """Obtain a RidgeCV estimator.
        
        Parameters
        ----------
        alphas : Union[torch.Tensor, list, float, int], default=1
            Penalties to use for estimation.
        fit_intercept : bool, default=True
            Whether to fit an intercept.
        alpha_per_target : bool, default=False
            Whether to use a different penalty for each target.
        """
        
        # check alphas
        if isinstance(alphas, int) | isinstance(alphas, float):
            alphas = torch.tensor([alphas])
        
        if isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.alpha_per_target = alpha_per_target
        
        self.alpha_ = None
        self.coef_ = None
        self.intercept_ = None
        self.metric_ = metrics.r2
    
    def _preprocess(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess the data.

        Parameters
        ----------
        X : torch.Tensor
            The features.
        y : torch.Tensor
            The targets.

        Returns
        -------
        X : torch.Tensor
            The preprocessed features.
        y : torch.Tensor
            The preprocessed targets.
        X_offset : torch.Tensor
            Offsets in X
        y_offset : torch.Tensor
            Offsets in y
        """

        if self.fit_intercept:
            # find and demean
            X_offset = X.mean(0)
            X -= X_offset
            
            # same for y
            y_offset = y.mean(0)
            y -= y_offset
        else:
            # otherwise, just zero out
            X_offset = torch.zeros(X.shape[1], dtype = X.dtype, device = X.device)
            y_offset = torch.zeros(y.shape[1], dtype = X.dtype, device = X.device)
        
        return (X, y, X_offset, y_offset)
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the RidgeCV model.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape ``(n_samples, n_features)``.
        y : torch.Tensor
            Output data of shape ``(n_samples[, n_targets])``.
        """
        
        # check shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f'`X` and `y` must have the same number of samples, ' +
                f'but got {X.shape[0]} and {y.shape[0]}'
            )

        # check y shape
        if len(y.shape) == 1:
            y = y[:, None]
        
        # get dims
        n, n_x = X.shape
        _, n_y = y.shape
        
        # preprocess
        X, y = X.clone(), y.clone() # make sure we don't have in-place changes
        X, y, X_offset, y_offset = self._preprocess(X, y)
        self.alphas = self.alphas.to(X.dtype).to(X.device)
        
        n_alphas = self.alphas.numel()
        n_targets = y.shape[-1]
        
        # disable gradients and check optimal decomposition structure
        with torch.no_grad():
            ones = torch.ones(n, dtype=X.dtype, device=X.device)
            normalized_ones = ones / torch.linalg.vector_norm(ones)

            if n <= n_x:
                # if we have fewer samples than features, use gram formulation
                gram = X @ X.mT

                if self.fit_intercept:
                    # add unregularized intercept direction
                    gram.add_(ones[:,None] * ones[None,:])

                # decomposition
                eigenvalues, Q = torch.linalg.eigh(gram)
                Qty = Q.mT @ y
                Q2 = Q.square()

                if self.fit_intercept:
                    # identify intercept dimension in eigenvectors
                    alignment = torch.abs(Q.mT @ normalized_ones)
                    intercept_dim = alignment.argmax()
                
                inverse = (
                    eigenvalues[None,:] + self.alphas[:,None]
                ).reciprocal() # (n_alphas, n_components)

                if self.fit_intercept:
                    # remove intercept direction from inverse G
                    inverse[:, intercept_dim] = 0.0

                # compute duals
                duals = torch.matmul(
                    Q,
                    inverse[:,:,None] * Qty[None,:,:],
                ) # (n_alphas, n_samples, n_targets)

                # compute diagonal
                diagonal = (Q2 @ inverse.mT).mT # (n_alphas, n_samples)

                # compute loo residuals
                loo_errors = duals / diagonal[:,:,None]
                losses = loo_errors.square().mean(1)
                
                if self.alpha_per_target:
                    # select best alpha per target
                    best = losses.argmin(0)
                    selected_alphas = self.alphas[best]

                    inverse = (
                        eigenvalues[:,None]
                        + selected_alphas[None,:]
                    ).reciprocal()

                    if self.fit_intercept:
                        # remove intercept direction
                        inverse[intercept_dim, :] = 0.0

                    duals = Q @ (inverse * Qty)
                else:
                    # select best alpha overall
                    best = losses.mean(1).argmin()
                    selected_alphas = self.alphas[best]

                    inverse = (
                        eigenvalues + selected_alphas
                    ).reciprocal()

                    if self.fit_intercept:
                        # remove intercept direction
                        inverse[intercept_dim] = 0.0

                    duals = Q @ (inverse[:, None] * Qty)

                # prepare coefficients
                coefs = duals.mT @ X
            else:
                # otherwise, use SVD approach
                if self.fit_intercept:
                    # add intercept to matrix
                    X_decomp = torch.cat(
                        (X, ones[:, None]),
                        dim=1,
                    )
                else:
                    X_decomp = X

                # decomposition
                U, singular_values, _ = torch.linalg.svd(
                    X_decomp, full_matrices = False,
                )

                singular_values_sq = singular_values.square()
                Uy = U.mT @ y
                U2 = U.square()

                if self.fit_intercept:
                    alignment = torch.abs(U.mT @ normalized_ones)
                    intercept_dim = alignment.argmax()

                alpha_inv = self.alphas.reciprocal()

                weights = (
                    singular_values_sq[None,:]
                    + self.alphas[:,None]
                ).reciprocal() - alpha_inv[:,None] # (n_alphas, rank)

                if self.fit_intercept:
                    # cancel regularisation for intercept dimension
                    weights[:, intercept_dim] = -alpha_inv

                duals = torch.matmul(
                    U,
                    weights[:,:,None] * Uy[None,:,:],
                )
                duals.add_(alpha_inv[:,None,None] * y[None,:,:])

                diagonal = (U2 @ weights.mT).mT
                diagonal.add_(alpha_inv[:,None])

                loo_errors = duals / diagonal[:,:,None]
                losses = loo_errors.square().mean(1)

                if self.alpha_per_target:
                    # select best alpha per target
                    best = losses.argmin(0)
                    selected_alphas = self.alphas[best]

                    selected_inv = selected_alphas.reciprocal()

                    weights = (
                        singular_values_sq[:,None]
                        + selected_alphas[None,:]
                    ).reciprocal() - selected_inv[None,:]

                    if self.fit_intercept:
                        # cancel regularisation for intercept
                        weights[intercept_dim,:] = -selected_inv

                    duals = U @ (weights * Uy)
                    duals.add_(y * selected_inv[None,:])
                else:
                    # select best alpha overall
                    best = losses.mean(1).argmin()
                    selected_alphas = self.alphas[best]

                    selected_inv = selected_alphas.reciprocal()

                    weights = (
                        singular_values_sq + selected_alphas
                    ).reciprocal() - selected_inv

                    if self.fit_intercept:
                        # cancel intercept regularisation
                        weights[intercept_dim] = -selected_inv

                    duals = U @ (weights[:,None] * Uy)
                    duals.add_(y, alpha=selected_inv)

                # prepare coefficients
                coefs = duals.mT @ X
            
        # set alphas and coefs
        self.coef_ = coefs
        self.alpha_ = selected_alphas
        
        # compute intercept
        if self.fit_intercept:
            Xoff_coef = X_offset[None,:] @ self.coef_.mT
            self.intercept_ = y_offset - Xoff_coef
        else:
            self.intercept_ = 0.0
        
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions from fitted model.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        
        Returns
        -------
        y : torch.Tensor
            Output data of shape ``(n_samples, n_features)``.
        """
        
        # make sure model has been fit
        if (self.coef_ is None) | (self.intercept_ is None):
            raise ValueError('Model has not been fit yet.')

        return X @ self.coef_.mT + self.intercept_
    
    def score(self, X: torch.Tensor, y: torch.Tensor, metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Make predictions from :math:`X` and score against :math:`y`.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        y : torch.Tensor
            Output data of shape ``(n_samples, n_features)``.
        metric : Optional[Metric], default=None
            Metric or tuple of metrics to compute.  If ``None``, defaults to :py:attr:`~mvpy.estimators.RidgeCV.metric_`.
        
        Returns
        -------
        score : torch.Tensor | Dict[str, torch.Tensor]
            Scores of shape ``(n_features,)`` or, for multiple metrics, a dictionary of metric names and scores of shape ``(n_features,)``.
        
        .. warning::
            If multiple values are supplied for ``metric``, this function will
            output a dictionary of ``{Metric.name: score, ...}`` rather than
            a stacked array. This is to provide consistency across cases where
            metrics may or may not differ in their output shapes.
        """
        
        # check metric
        if metric is None:
            metric = self.metric_
        
        return metrics.score(self, metric, X, y)

    def clone(self):
        """Make a clone of this class.
        
        Returns
        -------
        RidgeCV
            A clone of this class.
        """
        
        return _RidgeCV_torch(
            alphas = self.alphas, 
            fit_intercept = self.fit_intercept, 
            alpha_per_target = self.alpha_per_target
        )

class RidgeCV(sklearn.base.BaseEstimator):
    """Implements ridge regression with cross-validation.
    
    Ridge regression maps input data :math:`X` to output data :math:`y` 
    through coefficients :math:`\\beta`:
    
    .. math::

        y = \\beta X + \\varepsilon
    
    and solves for the model :math:`\\beta` through:
    
    .. math::
    
        \\arg\\min_\\beta \\sum_i (y_i - \\beta^T X_i)^2 + \\alpha_\\beta\\lvert\\lvert\\beta\\rvert\\rvert^2
    
    where :math:`\\alpha_\\beta` are penalties to test in LOO-CV which 
    has a convenient closed-form solution here:
    
    .. math::

        \\arg\\min_{\\alpha_\\beta} \\frac{1}{N}\\sum_{i = 1}^{N} \\left(\\frac{y - \\beta_\\alpha X}{1 - H_{\\alpha,ii}}\\right)\\qquad
        \\textrm{where}\\qquad
        H_{\\alpha,ii} = \\textrm{diag}\\left(X(X^T X + \\alpha I)^{-1}X^T\\right)
    
    As such, this will automatically evaluate the LOO-CV of all values of 
    :py:attr:`~mvpy.estimators.RidgeCV.alphas` and chose the penalty that
    minimises the mean-squared loss. This is convenient because it is much
    faster than performing inner cross-validation to fine-tune penalties.
    
    For more information on ridge regression, see [1]_. Note that this implementation 
    will automatically chose either the gram matrix formulation of the problem if 
    ``n_samples`` is smaller than ``n_features`` or SVD formulation otherwise for 
    optimal speed.
    
    Parameters
    ----------
    alphas : np.ndarray | torch.Tensor | List | float | int, default=1
        Penalties to use for estimation.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    alpha_per_target : bool, default=True
        Whether to use a different penalty for each target.
    
    Attributes
    ----------
    alpha_ : np.ndarray | torch.Tensor
        The penalties used for estimation.
    intercept_ : np.ndarray | torch.Tensor
        The intercepts of shape ``(n_features,)``.
    coef_ : np.ndarray | torch.Tensor
        The coefficients of shape ``(n_channels, n_features)``.
    metric_ : mvpy.metrics.r2
        The default metric to use.
    
    Notes
    -----
    If data are supplied as numpy, this class will fall back to :py:class:`sklearn.linear_model.RidgeCV`. See [2]_.
    
    References
    ----------
    .. [1] McDonald, G.C. (2009). Ridge regression. Wiley Interdisciplinary Reviews: Computational Statistics, 1, 93-100. doi.org/10.1002/wics.14
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
    
    Examples
    --------
    >>> import torch
    >>> from mvpy.estimators import RidgeCV
    >>> ß = torch.normal(0, 1, size = (5,))
    >>> X = torch.normal(0, 1, size = (240, 5))
    >>> y = ß @ X.T + torch.normal(0, 0.5, size = (X.shape[0],))
    >>> model = RidgeCV().fit(X, y)
    >>> model.coef_
    """
    
    def __new__(self, alphas: Union[np.ndarray, torch.Tensor, list, float, int] = 1, fit_intercept: bool = True, alpha_per_target: bool = False):
        """Obtain a RidgeCV estimator.
        
        Parameters
        ----------
        alphas : np.ndarray | torch.Tensor | List | float | int, default=1
            Penalties to use for estimation.
        fit_intercept : bool, default=True
            Whether to fit an intercept.
        alpha_per_target : bool, default=False
            Whether to use a different penalty for each target.
        """
        
        # check alphas
        if isinstance(alphas, int) | isinstance(alphas, float):
            alphas = torch.tensor([alphas])
        
        if isinstance(alphas, list):
            alphas = torch.tensor(alphas)
        
        # check model type
        if isinstance(alphas, torch.Tensor):
            return _RidgeCV_torch(
                alphas = alphas, 
                fit_intercept = fit_intercept, 
                alpha_per_target = alpha_per_target
            )
        elif isinstance(alphas, np.ndarray):
            return _RidgeCV_numpy(
                alphas = alphas, 
                fit_intercept = fit_intercept, 
                alpha_per_target = alpha_per_target
            )
        
        raise ValueError(
            f'Alphas should be of type np.ndarray or torch.tensor, ' + 
            f'but got {type(alphas)}.'
        )
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> "RidgeCV":
        """Fit the estimator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        y : np.ndarray | torch.Tensor
            Output data of shape ``(n_samples, n_features)``.
        
        Returns
        -------
        ridge : RidgeCV
            The fitted ridge estimator.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Predict from the estimator.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        
        Returns
        -------
        y_h : np.ndarray | torch.Tensor
            Predicted data of shape ``(n_samples, n_features)``.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def score(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None) -> Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
        """Make predictions from :math:`X` and score against :math:`y`.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape ``(n_samples, n_channels)``.
        y : torch.Tensor
            Output data of shape ``(n_samples, n_features)``.
        metric : Optional[Metric], default=None
            Metric or tuple of metrics to compute.  If ``None``, defaults to :py:attr:`~mvpy.estimators.RidgeCV.metric_`.
        
        Returns
        -------
        score : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
            Scores of shape ``(n_features,)`` or, for multiple metrics, a dictionary of metric names and scores of shape ``(n_features,)``.
        
        .. warning::
            If multiple values are supplied for ``metric``, this function will
            output a dictionary of ``{Metric.name: score, ...}`` rather than
            a stacked array. This is to provide consistency across cases where
            metrics may or may not differ in their output shapes.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')
    
    def clone(self) -> "RidgeCV":
        """Make a clone of this class.
        
        Returns
        -------
        ridge : RidgeCV
            A clone of this class.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')