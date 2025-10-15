'''
A collection of estimators for ReceptiveField modeling (mTRF + SR).
'''

import numpy as np
import torch
import sklearn
import scipy
import math

from .ridgecv import _RidgeCV_numpy, _RidgeCV_torch
from ..crossvalidation import KFold
from ..math import pearsonr
from ..utilities import compile
from .. import metrics

from typing import Union, Any, Optional, List, Dict, Tuple

@compile.numpy()
def _edge_correct_numpy(XX_eij: np.ndarray, X_e: np.ndarray, ch_i: int, ch_j: int, s_min: int, s_max: int):
    """Perform in-place edge correction.
    
    Parameters
    ----------
    XX_eij : np.ndarray
        Auto-correlation entry of shape (n_features * n_trf, n_features * n_trf).
    X_e : np.ndarray
        Input data at n_epoch of shape (n_features, n_timepoints).
    ch_i : int
        Current channel i.
    ch_j : int
        Current channel j.
    s_min : int
        Minimum sample.
    s_max : int
        Maximum sample.
    """
    
    if s_max > 0:
        tail = _toeplitz_dot_numpy(X_e[ch_i,-1:-s_max:-1], X_e[ch_j,-1:-s_max:-1])
        if s_min > 0:
            tail = tail[s_min - 1:,s_min - 1:]
        XX_eij[max(-s_min + 1, 0):,max(-s_min + 1, 0):] -= tail
    if s_min < 0:
        head = _toeplitz_dot_numpy(X_e[ch_i,:-s_min], X_e[ch_j,:-s_min])[::-1, ::-1]
        if s_max < 0:
            head = head[:s_max,:s_max]
        XX_eij[:-s_min,:-s_min] -= head

@compile.numpy()
def _toeplitz_dot_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute toeplitz dot.
    
    Parameters
    ----------
    a : np.ndarray
        Matrix a.
    b : np.ndarray
        Matrix b.
    
    Returns
    -------
    out : np.ndarray
        Output matrix.
    """
    
    out = np.outer(a, b)
    for ii in range(1, len(a)):
        out[ii,ii:] += out[ii-1,ii-1:-1]
        out[ii+1:,ii] += out[ii:-1,ii-1]
    return out

class _ReceptiveField_numpy(sklearn.base.BaseEstimator):
    """Obtain a new receptive field estimator using numpy backend.
    
    Parameters
    ----------
    t_min : float
        Minimum time point to fit (unlike TimeDelayed, this is relative to y).
    t_max : float
        Maximum time point to fit (unlike TimeDelayed, this is relative to y). Must be greater than t_min.
    fs : int
        Sampling frequency.
    alpha : Union[float, np.ndarray, List], default=1.0
        Alpha penalties as float or of shape (n_penalties,). If not float, cross-validation will be employed (see reg_cv).
    reg_type : Union[str, List], default='ridge'
        Type of regularisation to employ (either 'ridge' or 'laplacian' or tuple describing (`time`, `features`)).
    reg_cv : Any, default=5
        If alpha is list or array, what cross-validation scheme should we use? Integers are interpeted as n_splits for KFold crossvalidation. String input 'LOO' will use RidgeCV to solve LOO over alphas (only available for reg_type='ridge'). Alternatively, a cross-validator that exposes a split()-method can be supplied.
    patterns : bool, default=False
        Should we estimate the patterns from coefficients and data (useful only for stimulus reconstruction, not mTRF)?
    fit_intercept : bool, default=True
        Should we fit an intercept for this model?
    edge_correction : bool, default=True
        Should we apply edge corrections to auto-correlations?
    
    Attributes
    ----------
    t_min : float
        Minimum time point to fit (unlike TimeDelayed, this is relative to y).
    t_max : float
        Maximum time point to fit (unlike TimeDelayed, this is relative to y). Must be greater than t_min.
    fs : int
        Sampling frequency.
    alpha : Union[float, np.ndarray, List], default=1.0
        Alpha penalties as float or of shape (n_penalties,). If not float, cross-validation will be employed (see reg_cv).
    reg_type : Union[str, List], default='ridge'
        Type of regularisation to employ (either 'ridge' or 'laplacian' or tuple describing (`time`, `features`)).
    reg_cv : Any, default=5
        If alpha is list or array, what cross-validation scheme should we use? Integers are interpeted as n_splits for KFold crossvalidation. String input 'LOO' will use RidgeCV to solve LOO over alphas (only available for reg_type='ridge'). Alternatively, a cross-validator that exposes a split()-method can be supplied.
    patterns : bool, default=False
        Should we estimate the patterns from coefficients and data (useful only for stimulus reconstruction, not mTRF)?
    fit_intercept : bool, default=True
        Should we fit an intercept for this model?
    edge_correction : bool, default=True
        Should we apply edge corrections to auto-correlations?
    s_min : int
        t_min converted to samples.
    s_max : int
        t_max converted to samples.
    window : np.ndarray
        The TRF window ranging from s_min-s_max of shape (n_trf,).
    n_features_ : int
        Number of features in X.
    n_channels_ : int
        Number of channels in y.
    n_trf_ : int
        Number of timepoints in the estimated response functions.
    cov_ : np.ndarray
        Covariance from auto-correlations of shape (n_samples, n_features * n_trf, n_features * n_trf).
    coef_ : np.ndarray
        Estimated coefficients of shape (n_channels, n_features, n_trf).
    pattern_ : np.ndarray
        If computed, estimated pattern of shape (n_channels, n_features, n_trf).
    intercept_ : Union[float, np.ndarray]
        Estimated intercepts of shape (n_channels,) or float.
    metric_ : mvpy.metrics.r2
        The default metric to use.
    """
    
    def __init__(self, t_min: float, t_max: float, fs: int, alpha: Union[float, np.ndarray, List] = 1.0, reg_type: Union[str, List] = 'ridge', reg_cv: Any = 5, patterns: bool = False, fit_intercept: bool = True, edge_correction: bool = True):
        """Obtain a new receptive field estimator.
        
        Parameters
        ----------
        t_min : float
            Minimum time point to fit (unlike TimeDelayed, this is relative to y).
        t_max : float
            Maximum time point to fit (unlike TimeDelayed, this is relative to y). Must be greater than t_min.
        fs : int
            Sampling frequency.
        alpha : Union[float, np.ndarray, List], default=1.0
            Alpha penalties as float or of shape (n_penalties,). If not float, cross-validation will be employed (see reg_cv).
        reg_type : Union[str, List], default='ridge'
            Type of regularisation to employ (either 'ridge' or 'laplacian' or tuple describing (`time`, `features`)).
        reg_cv : Any, default=5
            If alpha is list or array, what cross-validation scheme should we use? Integers are interpeted as n_splits for KFold crossvalidation. String input 'LOO' will use RidgeCV to solve LOO over alphas (only available for reg_type='ridge'). Alternatively, a cross-validator that exposes a split()-method can be supplied.
        patterns : bool, default=False
            Should we estimate the patterns from coefficients and data (useful only for stimulus reconstruction, not mTRF)?
        fit_intercept : bool, default=True
            Should we fit an intercept for this model?
        edge_correction : bool, default=True
            Should we apply edge corrections to auto-correlations?
        """
        
        # setup parameters
        self.t_min = t_min
        self.t_max = t_max
        self.fs = fs
        self.alpha = alpha
        self.reg_type = reg_type
        self.reg_cv = reg_cv
        self.patterns = patterns
        self.fit_intercept = fit_intercept
        self.edge_correction = edge_correction
        
        # check t_min and t_max
        if t_max < t_min:
            raise ValueError(f't_min must be smaller than t_max, but got {t_min} > {t_max}.')
        
        # check regularisation opts
        if (reg_type != 'ridge') and (reg_cv == 'LOO'):
            raise ValueError(f'reg_cv=`LOO` is available only for reg_type=`ridge`, but got {reg_type}.')

        # check reg_cv
        if (hasattr(reg_cv, 'split') == False) and ((isinstance(reg_cv, int) or isinstance(reg_cv, str)) == False):
            raise ValueError(
                f'`reg_cv` must either be of type int specifying number of folds to use, ' + 
                f'str (\'LOO\') for RidgeCV solving, ' + 
                f'or expose a `split` method for cross-validation.'
            )
        
        # check alpha
        if isinstance(self.alpha, list):
            self.alpha = np.array(self.alpha)
        
        if isinstance(self.alpha, np.ndarray):
            if self.alpha.shape[0] == 1:
                self.alpha = float(self.alpha[0])
        
        if (isinstance(self.alpha, np.ndarray) == False) and (isinstance(self.alpha, float) == False):
            raise ValueError(f'Unknown `alpha` type, must be either float, List, or np.ndarray, but got {type(self.alpha)}.')
        
        # setup internals
        self.s_min = int(t_min * fs)
        self.s_max = int(t_max * fs) + 1
        self.window = np.arange(self.s_min, self.s_max)
        self.n_features_ = None
        self.n_channels_ = None
        self.n_trf = None
        self.cov_ = None
        self.coef_ = None
        self.intercept_ = None
        self.pattern_ = None
        self.metric_ = metrics.r2
    
    def _compute_corrs(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute the auto- and cross-correlation matrices.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features, n_timepoints).
        y : np.ndarray
            Input data of shape (n_samples, n_channels, n_timepoints).
        
        Returns
        -------
        X_XT : np.ndarray
            Auto-correlation matrix of shape (n_samples, n_features * n_trf, n_features * n_trf).
        X_y : np.ndarray
            Cross-correlation matrix of shape (n_samples, n_trf, n_channels, n_features).
        X_offset : Union[float, np.ndarray]
            If computed, means of X of shape (n_features,).
        y_offset : Union[float, np.ndarray]
            If comptued, means of y of shape (n_channels,).
        """
        
        # checks
        assert len(X.shape) == 3
        assert len(y.shape) == 3
        assert (X.shape[0] == y.shape[0]) & (X.shape[-1] == y.shape[-1])
        
        # get dims
        n_samples, n_features, n_timepoints = X.shape
        _, n_channels, _ = y.shape
        
        # get min and max samples
        s_min = self.s_min
        s_max = self.s_max
        
        # determine length of TRF
        len_trf = s_max - s_min
        
        # determine length of FFT (next power of 2)
        n_fft = int(
            2 ** math.ceil(
                math.log2(2 * X.shape[-1] - 1)
            )
        )
        
        # record relevant dims
        self.n_features_ = n_features
        self.n_channels_ = n_channels
        self.n_trf_ = len_trf
        
        # if desired, compute offsets
        if self.fit_intercept:
            X_offset = X.mean(axis = (0, 2), keepdims = True)
            X = X - X_offset
            
            y_offset = y.mean(axis = (0, 2), keepdims = True)
            y = y - y_offset
        else:
            X_offset = y_offset = 0.0
        
        # setup toeplitz matix
        ij = np.empty((len_trf, len_trf), int)
        for ii in range(len_trf):
            ij[ii, ii:] = np.arange(len_trf - ii)
            ij[ii + 1:, ii] = np.arange(n_fft - 1, n_fft - len_trf + ii, -1)
        
        # compute FFTs
        F_X = np.fft.rfft(X, n = n_fft, axis = -1)
        F_Z = F_X.conj()
        F_y = np.fft.rfft(y, n = n_fft, axis = -1)
        
        # compute auto-correlation of X
        n = np.arange(X.shape[1])
        nx, ny = np.meshgrid(n, n)
        cx, cy = np.triu_indices(X.shape[1], k = 0)
        indc_i, indc_j = nx.flatten()[cx], ny.T.flatten()[cy]
        XX = np.fft.irfft(
            F_X[:,indc_i,:] * F_Z[:,indc_j,:],
            n = n_fft, axis = -1
        ) # (n_samples, n_triu_chs, n_timepoints)
        
        # compute cross-correlation of X and y
        Xy = np.fft.irfft(
            F_y[:,None,:,:] * F_Z[:,:,None,:],
            n = n_fft, axis = -1
        ) # (n_samples, n_channels, n_features, n_timepoints)
        
        # setup estimation
        X_XT = np.zeros((X.shape[0], X.shape[1] * len_trf, X.shape[1] * len_trf))
        X_y = np.zeros((X.shape[0], len_trf, X.shape[1], y.shape[1]))
        
        # loop over epochs
        for n_epoch in range(X.shape[0]):
            # set counter for n_triu_chs
            eij = 0
            
            # loop over channel pairs
            for ch_i in range(X.shape[1]):
                for ch_j in range(ch_i, X.shape[1]):
                    # select with toeplitz indexer
                    XX_eij = XX[n_epoch][eij][ij]
                    
                    # correct edges
                    if self.edge_correction:
                        _edge_correct_numpy(XX_eij, X[n_epoch], ch_i, ch_j, self.s_min, self.s_max)
                    
                    # sum auto-correlation
                    X_XT[
                        n_epoch,
                        ch_i * len_trf:(ch_i + 1) * len_trf,
                        ch_j * len_trf:(ch_j + 1) * len_trf
                    ] += XX_eij
                    
                    # if not diagonal, add mirror triangle
                    if ch_i != ch_j:
                        X_XT[
                            n_epoch,
                            ch_j * len_trf:(ch_j + 1) * len_trf,
                            ch_i * len_trf:(ch_i + 1) * len_trf
                        ] += XX_eij
                    
                    # move triangle counter
                    eij += 1
                
                # sum cross-correlation
                if s_min < 0 and s_max >= 0:
                    X_y[n_epoch,:-s_min,ch_i] += Xy[n_epoch,ch_i,:,s_min:].T
                    X_y[n_epoch,len_trf - s_max:,ch_i] += Xy[n_epoch,ch_i,:,:s_max].T
                else:
                    X_y[n_epoch,:,ch_i] += Xy[n_epoch,ch_i,:,s_min:s_max].T
        
        return X_XT, X_y, X_offset, y_offset
    
    def _compute_reg_neighbours(self, n_features: int, n_trf: int, reg_type: Union[str, List]) -> np.ndarray:
        """Compute regularisation matrix.
        
        Parameters
        ----------
        n_features : int
            Number of features in X.
        n_trf : int
            Number of timepoints we estimate.
        reg_type : Union[str, List]
            Regularisation type (either 'ridge' or 'laplacian' or tuple thereof).
        
        Returns
        -------
        reg : np.ndarray
            Regularisation matrix.
        """
        
        # check reg_type
        if isinstance(reg_type, str):
            reg_type = (reg_type,) * 2
        
        if len(reg_type) != 2:
            raise ValueError(f'reg_type must be of length two, but got {len(reg_type)}.')
        
        for r in reg_type:
            if r not in ['ridge', 'laplacian']:
                raise ValueError(f'Unknown reg_type. Expected ridge or laplacian, but got {r}.')
        
        # setup regularisation
        reg_time = (reg_type[0] == 'laplacian') and (n_trf > 1)
        reg_fts = (reg_type[1] == 'laplacian') and (n_features > 1)
        
        # handle identity case
        if not reg_time and not reg_fts:
            return np.eye(n_features * n_trf)
        
        # handle time laplacian
        if reg_time:
            reg = np.eye(n_trf)
            stride = n_trf + 1
            reg.flat[1::stride] += -1
            reg.flat[n_trf::stride] += -1
            reg.flat[n_trf + 1:-n_trf - 1:stride] += 1
            args = [reg] * n_features
            reg = scipy.linalg.block_diag(*args)
        else:
            reg = np.zeros((n_trf * n_features,) * 2)
        
        # handle feature laplacian
        if reg_fts:
            block = n_trf * n_trf
            row_offset = block * n_features
            stride = n_trf * n_features + 1
            reg.flat[n_trf:-row_offset:stride] += -1
            reg.flat[n_trf + row_offset :: stride] += 1
            reg.flat[row_offset:-n_trf:stride] += -1
            reg.flat[: -(n_trf + row_offset) : stride] += 1
        
        return reg
    
    def _fit_corrs(self, X_XT: np.ndarray, X_y: np.ndarray, alpha: Optional[float] = None) -> tuple[np.ndarray, Optional[float]]:
        """Fit the model.
        
        Parameters
        ----------
        X_XT : np.ndarray
            Auto-correlation matrix of shape (n_samples, n_features * n_trf, n_features * n_trf).
        X_y : np.ndarray
            Cross-correlation matrix of shape (n_samples, n_trf, n_channels, n_features).
        alpha : Optional[float], default=None
            Regularisation strength to apply. If None, perform LOO.

        Returns
        -------
        coef_ : np.ndarray
            Estiamted coefficients of shape (n_channels, n_features, n_trf).
        alpha_ : Optional[float]
            If computed, best alpha value used in RidgeCV.
        """
        
        # check dims
        if None in [self.n_features_, self.n_channels_, self.n_trf_]:
            raise ValueError(f'_fit_corrs cannot be called before _compute_corrs.')
        
        # sum over available epochs
        X_XT = X_XT.sum(0)
        X_y = X_y.sum(0)
        
        # reshape X_y
        X_y = np.transpose(X_y, (2, 1, 0))
        X_y = X_y.reshape(self.n_channels_, self.n_features_ * self.n_trf_)
        X_y = X_y.T
        
        if (self.reg_type == 'ridge') and (self.reg_cv == 'LOO') and (alpha is None):
            # solve via RidgeCV
            est = _RidgeCV_numpy(alphas = self.alpha, fit_intercept = True).fit(X_XT, X_y)
            w = est.coef_.reshape((self.n_channels_, self.n_features_, self.n_trf_))
            alpha = est.alpha_
            return w, alpha
        else:
            # compute and apply regularisation
            reg = self._compute_reg_neighbours(self.n_features_, self.n_trf_, self.reg_type)
            mat = X_XT + alpha * reg
            
            # solve
            try:
                # try solving equation
                w = scipy.linalg.solve(mat, X_y, overwrite_a = False, assume_a = 'pos')
            except np.linalg.LinAlgError:
                # fall back to least squares solution if singular
                w = scipy.linalg.lstsq(mat, X_y, lapack_driver = 'gelsy')[0]
            
            # reshape appropriately
            w = w.T.reshape((self.n_channels_, self.n_features_, self.n_trf_))
        
        return w
    
    def _fit_intercept(self, X_offset: Union[np.ndarray, float], y_offset: Union[np.ndarray, float], coef_: np.ndarray) -> Union[float, np.ndarray]:
        """Fit the intercept.
        
        Parameters
        ----------
        X_offset : Union[float, np.ndarray]
            Means of X of shape (n_features,) or float.
        y_offset : Union[float, np.ndarray]
            Means of y of shape (n_channels,) or float.
        coef_ : np.ndarray
            Estimated coefficients of shape (n_channels, n_features, n_trf).
        
        Returns
        -------
        intercept : Union[float, np.ndarray]
            Model intercept, if desired.
        """
        
        # check if we need to fit
        if self.fit_intercept:
            # fit sklearn-style
            return y_offset.squeeze() - np.dot(X_offset.squeeze(), coef_.sum(-1).T)
        
        # otherwise, just return zero
        return 0.0

    def _fit_pattern(self, X: np.ndarray, y: np.ndarray, coef_: np.ndarray) -> Optional[np.ndarray]:
        """Fit the patterns.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features, n_timepoints).
        y : np.ndarray
            Input data of shape (n_samples, n_channels, n_timepoints).
        coef_ : np.ndarray
            Coefficients of shape (n_channels, n_features, n_trf).
        
        Returns
        -------
        pattern : Optional[np.ndarray]
            If computed, patterns of shape (n_channels, n_features, n_trf).
        """
        
        if self.patterns:
            # get covariance of X
            S_X = self.cov_.sum(0) / float(X.shape[-1] * X.shape[0] - 1)
            
            # get precision of y
            if y.shape[1] > 1:
                # demean
                z = y.swapaxes(1, 2).reshape((-1, y.shape[1]))
                z = (z - z.mean(0, keepdims = True))
                
                # covariance
                cov = np.cov(z.T)
                
                # compute precision
                try:
                    # try scipy first for speed
                    P_y = scipy.linalg.pinv(cov)
                except:
                    # fall back to numpy
                    P_y = np.linalg.pinv(cov)
                
                # pattern
                pattern = S_X @ (coef_.reshape((self.n_features_ * self.n_trf_, self.n_channels_)) @ P_y)
            else:
                # norm
                P_y = 1.0 / float(X.shape[-1] * X.shape[0] - 1)
                
                # pattern
                pattern = S_X @ (self.coef_.reshape((self.n_features_ * self.n_trf_, self.n_channels_)) * P_y)
            
            # reshape patterns
            return pattern.reshape(self.coef_.shape)
        
        return None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ReceptiveField_numpy":
        """Fit the estimator, optionally with cross-validation over penalties.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features, n_timepoints).
        y : np.ndarray
            Input data of shape (n_samples, n_channels, n_timepoints).
        
        Returns
        -------
        rf : _ReceptiveField_numpy
            The fit estimator.
        """
        
        # check X
        if len(X.shape) not in [2, 3]:
            raise ValueError(f'`X` must be of shape (n_samples[, n_features], n_timepoints), but got {X.shape}.')
        
        if len(X.shape) == 2:
            X = X[:,None,:]
        
        # check y
        if len(y.shape) not in [2, 3]:
            raise ValueError(f'`y` must be of shape (n_samples[, n_channels], n_timepoints), but got {y.shape}.')
        
        if len(y.shape) == 2:
            y = y[:,None,:]
        
        # check shape match
        if (X.shape[0] != y.shape[0]) or (X.shape[-1] != y.shape[-1]):
            raise ValueError(
                f'`X` (n_samples[, n_features], n_timepoints) and ' +
                f'`y` (n_samples[, n_channels], n_timepoints) ' + 
                f'must match in (n_samples, n_timepoints), but got ' + 
                f'X={(X.shape[0], X.shape[-1])} and ' + 
                f'y={(y.shape[0], y.shape[-1])}.'
            )
        
        # compute auto- and cross-correlations
        X_XT, X_y, X_offset, y_offset = self._compute_corrs(X, y)
        self.cov_ = X_XT
        
        # check LOO case
        if self.reg_cv == 'LOO':
            # run directly and save
            self.coef_, self.alpha_ = self._fit_corrs(X_XT, X_y)
            self.intercept_ = self._fit_intercept(X_offset, y_offset, self.coef_)
            self.pattern_ = self._fit_pattern(X, y, self.coef_)
            
            return self

        # check if we need CV
        if isinstance(self.alpha, float) == False:
            # check if we have a crossvalidator provided in args
            if hasattr(self.reg_cv, 'split'):
                cv = self.reg_cv
            else:
                cv = KFold(n_splits = self.reg_cv).to_numpy()
            
            # setup data container
            oos = []
            
            # loop over alphas
            for a_i, alpha in enumerate(self.alpha):
                oos_i = []
                
                # loop over cv
                for f_i, (train, test) in enumerate(cv.split(X_XT, X_y)):
                    # fit coefficients
                    coef_ = self._fit_corrs(X_XT[train], X_y[train], alpha = alpha)
                    intercept_ = self._fit_intercept(X_offset, y_offset, coef_)
                    
                    # get predictions
                    y_h = self._predict(X[test], coef_, intercept_)
                    
                    # score predictions
                    oos_i.append(
                        pearsonr(
                            y_h.T, y[test].T
                        ).mean()
                    )
                
                # add score
                score_i = np.mean(oos_i)
                oos.append(score_i)

            # find best
            oos = np.array(oos)
            best = np.argmax(oos)
            self.alpha_ = self.alpha[best]
            
            # fit full best model
            self.coef_ = self._fit_corrs(X_XT, X_y, alpha = self.alpha_)
            self.intercept_ = self._fit_intercept(X_offset, y_offset, self.coef_)
            self.pattern_ = self._fit_pattern(X, y, self.coef_)
            
            return self
        
        # otherwise, simply fit single model
        self.alpha_ = self.alpha
        self.coef_ = self._fit_corrs(X_XT, X_y, alpha = self.alpha)
        self.intercept_ = self._fit_intercept(X_offset, y_offset, self.coef_)
        self.pattern_ = self._fit_pattern(X, y, self.coef_)
        
        return self

    def _predict(self, X: np.ndarray, coef_: np.ndarray, intercept_: Union[float, np.ndarray]) -> np.ndarray:
        """Make predictions from supplied model.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features, n_timepoints).
        coef_ : np.ndarray
            Model coefficients of shape (n_channels, n_features, n_trf).
        intercept_ : Union[float, np.ndarray]
            If computed, intercepts of shape (n_channels,) or float.
        
        Returns
        -------
        y_h : np.ndarray
            Predicted responses of shape (n_samples, n_channels, n_timepoints).
        """
        
        # check X
        if len(X.shape) == 2:
            X = X[:,None,:]
        
        # setup safe fft length
        n_fft = 2 ** math.ceil(math.log2(2 * X.shape[-1] - 1))
        
        # take FFTs
        F_x = np.fft.rfft(X, n = n_fft, axis = -1)
        F_w = np.fft.rfft(coef_, n = n_fft, axis = -1)
        
        # setup outputs
        y = np.zeros((X.shape[0], coef_.shape[0], X.shape[-1]))
        
        # loop over features
        for i in range(coef_.shape[0]):
            # do the frequency-domain convolution
            x_y = np.fft.irfft(F_x * F_w[i,None], n = n_fft, axis = -1).sum(1)
            
            # account for the fact that part of our coefficients may be negative
            x_y = x_y[...,max(-self.s_min, 0):X.shape[-1]+max(-self.s_min, 0)]
            
            # account for the fact that our coefficients may not start at t_min=0, but at e.g. t_min=0.2
            y[:,i,max(self.s_min, 0):] += x_y[...,:x_y.shape[-1]-max(self.s_min, 0)]
        
        # add intercept
        if isinstance(intercept_, float):
            y += intercept_
        else:
            if len(intercept_.shape) == 1:
                y += intercept_[None,:,None]
            elif len(intercept_.shape) == 2:
                y += intercept_[...,None]
            else:
                y += intercept_
        
        return y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions from model.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features, n_timepoints).
        
        Returns
        -------
        y_h : np.ndarray
            Predicted responses of shape (n_samples, n_channels, n_timepoints).
        """
        
        return self._predict(X, self.coef_, self.intercept_)

    def score(self, X: np.ndarray, y: np.ndarray, metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None) -> Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
        """Make predictions from :math:`X` and score against :math:`y`.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape ``(n_samples, n_features, n_timepoints)``.
        y : np.ndarray
            Output data of shape ``(n_samples, n_channels, n_timepoints)``.
        metric : Optional[Metric], default=None
            Metric or tuple of metrics to compute.  If ``None``, defaults to :py:attr:`~mvpy.estimators.ReceptiveField.metric_`.
        
        Returns
        -------
        score : np.ndarray | Dict[str, np.ndarray]
            Scores of shape ``(n_channels, n_timepoints)`` or, for multiple metrics, a dictionary of metric names and scores of shape ``(n_channels, n_timepoints)``.
        
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
    
    def clone(self) -> "_ReceptiveField_numpy":
        """Clone this class.
        
        Returns
        -------
        rf : _ReceptiveField_numpy
            A clone of this class.
        """
        
        return _ReceptiveField_numpy(
            self.t_min,
            self.t_max,
            self.fs,
            alpha = self.alpha,
            reg_type = self.reg_type,
            reg_cv = self.reg_cv,
            patterns = self.patterns,
            fit_intercept = self.fit_intercept,
            edge_correction = self.edge_correction
        )

@compile.torch()
def _edge_correct_torch(XX_eij: torch.Tensor, X_e: torch.Tensor, ch_i: torch.Tensor, ch_j: torch.Tensor, s_min: int, s_max: int):
    """Perform in-place edge correction.
    
    Parameters
    ----------
    XX_eij : torch.Tensor
        Auto-correlation entry of shape (n_features * n_trf, n_features * n_trf).
    X_e : torch.Tensor
        Input data at n_epoch (with temporal dim flipped) of shape (n_features, n_timepoints).
    ch_i : torch.Tensor
        Current channels i.
    ch_j : torch.Tensor
        Current channels j.
    s_min : int
        Minimum sample.
    s_max : int
        Maximum sample.
    """
    
    if s_max > 0:
        tail = _toeplitz_dot_torch(X_e[ch_i,:s_max - 1], X_e[ch_j,:s_max - 1])
        if s_min > 0:
            tail = tail[:,s_min - 1:,s_min - 1:]
        XX_eij[:,max(-s_min + 1, 0):,max(-s_min + 1, 0):] -= tail
    if s_min < 0:
        head = _toeplitz_dot_torch(X_e[ch_i,s_min:], X_e[ch_j,s_min:])
        if s_max < 0:
            head = head[:,:s_max,:s_max]
        XX_eij[:,:-s_min,:-s_min] -= head

@compile.torch()
def _toeplitz_dot_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute toeplitz dot.
    
    Parameters
    ----------
    a : torch.Tensor
        Matrix a.
    b : torch.Tensor
        Matrix b.
    
    Returns
    -------
    out : torch.Tensor
        Output matrix.
    """
    
    out = torch.bmm(a.unsqueeze(2), b.unsqueeze(1))
    for ii in range(1, a.shape[1]):
        out[:,ii,ii:] += out[:,ii-1,ii-1:-1]
        out[:,ii+1:,ii] += out[:,ii:-1,ii-1]
    return out

class _ReceptiveField_torch(sklearn.base.BaseEstimator):
    """Obtain a new receptive field estimator using torch backend.
    
    Parameters
    ----------
    t_min : float
        Minimum time point to fit (unlike TimeDelayed, this is relative to y).
    t_max : float
        Maximum time point to fit (unlike TimeDelayed, this is relative to y). Must be greater than t_min.
    fs : int
        Sampling frequency.
    alpha : Union[float, torch.Tensor, List], default=1.0
        Alpha penalties as float or of shape (n_penalties,). If not float, cross-validation will be employed (see reg_cv).
    reg_type : Union[str, List], default='ridge'
        Type of regularisation to employ (either 'ridge' or 'laplacian' or tuple describing (`time`, `features`)).
    reg_cv : Any, default=5
        If alpha is list or array, what cross-validation scheme should we use? Integers are interpeted as n_splits for KFold crossvalidation. String input 'LOO' will use RidgeCV to solve LOO over alphas (only available for reg_type='ridge'). Alternatively, a cross-validator that exposes a split()-method can be supplied.
    patterns : bool, default=False
        Should we estimate the patterns from coefficients and data (useful only for stimulus reconstruction, not mTRF)?
    fit_intercept : bool, default=True
        Should we fit an intercept for this model?
    edge_correction : bool, default=True
        Should we apply edge corrections to auto-correlations?
    
    Attributes
    ----------
    t_min : float
        Minimum time point to fit (unlike TimeDelayed, this is relative to y).
    t_max : float
        Maximum time point to fit (unlike TimeDelayed, this is relative to y). Must be greater than t_min.
    fs : int
        Sampling frequency.
    alpha : Union[float, torch.Tensor, List], default=1.0
        Alpha penalties as float or of shape (n_penalties,). If not float, cross-validation will be employed (see reg_cv).
    reg_type : Union[str, List], default='ridge'
        Type of regularisation to employ (either 'ridge' or 'laplacian' or tuple describing (`time`, `features`)).
    reg_cv : Any, default=5
        If alpha is list or array, what cross-validation scheme should we use? Integers are interpeted as n_splits for KFold crossvalidation. String input 'LOO' will use RidgeCV to solve LOO over alphas (only available for reg_type='ridge'). Alternatively, a cross-validator that exposes a split()-method can be supplied.
    patterns : bool, default=False
        Should we estimate the patterns from coefficients and data (useful only for stimulus reconstruction, not mTRF)?
    fit_intercept : bool, default=True
        Should we fit an intercept for this model?
    edge_correction : bool, default=True
        Should we apply edge corrections to auto-correlations?
    s_min : int
        t_min converted to samples.
    s_max : int
        t_max converted to samples.
    window : torch.Tensor
        The TRF window ranging from s_min-s_max of shape (n_trf,).
    n_features_ : int
        Number of features in X.
    n_channels_ : int
        Number of channels in y.
    n_trf_ : int
        Number of timepoints in the estimated response functions.
    cov_ : torch.Tensor
        Covariance from auto-correlations of shape (n_samples, n_features * n_trf, n_features * n_trf).
    coef_ : torch.Tensor
        Estimated coefficients of shape (n_channels, n_features, n_trf).
    pattern_ : torch.Tensor
        If computed, estimated pattern of shape (n_channels, n_features, n_trf).
    intercept_ : Union[float, torch.Tensor]
        Estimated intercepts of shape (n_channels,) or float.
    metric_ : mvpy.metrics.r2
        The default metric to use.
    """
    
    def __init__(self, t_min: float, t_max: float, fs: int, alpha: Union[float, torch.Tensor, List] = 1.0, reg_type: Union[str, List] = 'ridge', reg_cv: Any = 5, patterns: bool = False, fit_intercept: bool = True, edge_correction: bool = True):
        """Obtain a new receptive field estimator.
        
        Parameters
        ----------
        t_min : float
            Minimum time point to fit (unlike TimeDelayed, this is relative to y).
        t_max : float
            Maximum time point to fit (unlike TimeDelayed, this is relative to y). Must be greater than t_min.
        fs : int
            Sampling frequency.
        alpha : Union[float, torch.Tensor, List], default=1.0
            Alpha penalties as float or of shape (n_penalties,). If not float, cross-validation will be employed (see reg_cv).
        reg_type : Union[str, List], default='ridge'
            Type of regularisation to employ (either 'ridge' or 'laplacian' or tuple describing (`time`, `features`)).
        reg_cv : Any, default=5
            If alpha is list or array, what cross-validation scheme should we use? Integers are interpeted as n_splits for KFold crossvalidation. String input 'LOO' will use RidgeCV to solve LOO over alphas (only available for reg_type='ridge'). Alternatively, a cross-validator that exposes a split()-method can be supplied.
        patterns : bool, default=False
            Should we estimate the patterns from coefficients and data (useful only for stimulus reconstruction, not mTRF)?
        fit_intercept : bool, default=True
            Should we fit an intercept for this model?
        edge_correction : bool, default=True
            Should we apply edge corrections to auto-correlations?
        """
        
        # setup parameters
        self.t_min = t_min
        self.t_max = t_max
        self.fs = fs
        self.alpha = alpha
        self.reg_type = reg_type
        self.reg_cv = reg_cv
        self.patterns = patterns
        self.fit_intercept = fit_intercept
        self.edge_correction = edge_correction
        
        # check t_min and t_max
        if t_max < t_min:
            raise ValueError(f't_min must be smaller than t_max, but got {t_min} > {t_max}.')
        
        # check regularisation opts
        if (reg_type != 'ridge') and (reg_cv == 'LOO'):
            raise ValueError(f'reg_cv=`LOO` is available only for reg_type=`ridge`, but got {reg_type}.')

        # check reg_cv
        if (hasattr(reg_cv, 'split') == False) and ((isinstance(reg_cv, int) or isinstance(reg_cv, str)) == False):
            raise ValueError(
                f'`reg_cv` must either be of type int specifying number of folds to use, ' + 
                f'str (\'LOO\') for RidgeCV solving, ' + 
                f'or expose a `split` method for cross-validation.'
            )
        
        # check alpha
        if isinstance(self.alpha, list):
            self.alpha = torch.tensor(self.alpha)
        
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.shape[0] == 1:
                self.alpha = float(self.alpha.cpu().item())
        
        if (isinstance(self.alpha, torch.Tensor) == False) and (isinstance(self.alpha, float) == False):
            raise ValueError(f'Unknown `alpha` type, must be either float, List, or torch.Tensor, but got {type(self.alpha)}.')
        
        # setup internals
        self.s_min = int(t_min * fs)
        self.s_max = int(t_max * fs) + 1
        self.window = torch.arange(self.s_min, self.s_max)
        self.n_features_ = None
        self.n_channels_ = None
        self.n_trf = None
        self.cov_ = None
        self.coef_ = None
        self.intercept_ = None
        self.pattern_ = None
        self.metric_ = metrics.r2
    
    def _compute_corrs(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Union[float, torch.Tensor], Union[float, torch.Tensor]]:
        """Compute the auto- and cross-correlation matrices.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_features, n_timepoints).
        y : torch.Tensor
            Input data of shape (n_samples, n_channels, n_timepoints).
        
        Returns
        -------
        X_XT : torch.Tensor
            Auto-correlation matrix of shape (n_samples, n_features * n_trf, n_features * n_trf).
        X_y : torch.Tensor
            Cross-correlation matrix of shape (n_samples, n_trf, n_channels, n_features).
        X_offset : Union[float, torch.Tensor]
            If computed, means of X of shape (n_features,).
        y_offset : Union[float, torch.Tensor]
            If comptued, means of y of shape (n_channels,).
        """
        
        # checks
        assert len(X.shape) == 3
        assert len(y.shape) == 3
        assert (X.shape[0] == y.shape[0]) & (X.shape[-1] == y.shape[-1])
        
        # get dims
        n_samples, n_features, n_timepoints = X.shape
        _, n_channels, _ = y.shape
        
        # get min and max samples
        s_min = self.s_min
        s_max = self.s_max
        
        # determine length of TRF
        len_trf = s_max - s_min
        
        # determine length of FFT (next power of 2)
        n_fft = int(
            2 ** math.ceil(
                math.log2(2 * X.shape[-1] - 1)
            )
        )
        
        # record relevant dims
        self.n_features_ = n_features
        self.n_channels_ = n_channels
        self.n_trf_ = len_trf
        
        # if desired, compute offsets
        if self.fit_intercept:
            X_offset = X.mean((0, 2), keepdim = True)
            X = X - X_offset
            
            y_offset = y.mean((0, 2), keepdim = True)
            y = y - y_offset
        else:
            X_offset = y_offset = 0.0
        
        # setup toeplitz matix
        ij = torch.zeros((len_trf, len_trf), device = X.device).long()
        for ii in range(len_trf):
            ij[ii, ii:] = torch.arange(len_trf - ii, device = X.device)
            ij[ii + 1:, ii] = torch.arange(n_fft - 1, n_fft - len_trf + ii, -1, device = X.device)
        
        # for efficiency, compute flipped X
        X_f = X.flip(-1)
        
        # compute FFTs
        F_X = torch.fft.rfft(X, n = n_fft, dim = -1)
        F_Z = F_X.conj()
        F_y = torch.fft.rfft(y, n = n_fft, dim = -1)
        
        # compute auto-correlation of X
        n = torch.arange(X.shape[1], device = X.device)
        nx, ny = torch.meshgrid(n, n, indexing = 'ij')
        cx, cy = torch.triu_indices(X.shape[1], X.shape[1], offset = 0, device = X.device)
        indc_i, indc_j = nx.t().flatten()[cx], ny.flatten()[cy]
        XX = torch.fft.irfft(
            F_X[:,indc_i,:] * F_Z[:,indc_j,:],
            n = n_fft, dim = -1
        ) # (n_samples, n_triu_chs, n_timepoints)
        
        # compute cross-correlation of X and y
        Xy = torch.fft.irfft(
            F_y[:,None,:,:] * F_Z[:,:,None,:],
            n = n_fft, dim = -1
        ) # (n_samples, n_channels, n_features, n_timepoints)
        
        # expand toeplitz for feature pairs
        ij_f = ij.view(-1)
        ij_e = ij_f.unsqueeze(0).expand(XX.shape[1], -1)
        
        # setup estimation
        X_XT = torch.zeros((X.shape[0], X.shape[1] * len_trf, X.shape[1] * len_trf), dtype = X.dtype, device = X.device)
        X_y = torch.zeros((X.shape[0], len_trf, X.shape[1], y.shape[1]), dtype = X.dtype, device = X.device)
        
        # loop over epochs
        for n_epoch in range(X.shape[0]):
            # set counter for n_triu_chs
            eij = 0
            
            # select XX data as blocked view
            XX_n = XX[n_epoch]
            XX_b = XX_n.gather(1, ij_e).view(XX.shape[1], len_trf, len_trf) # (n_triu_pairs, n_trf, n_trf)
            
            # if required, apply blocked edge correction
            if self.edge_correction:
                _edge_correct_torch(XX_b, X_f[n_epoch], indc_i, indc_j, self.s_min, self.s_max)
            
            # loop over channel pairs
            for ch_i in range(X.shape[1]):                
                for ch_j in range(ch_i, X.shape[1]):
                    # select chunk
                    XX_eij = XX_b[eij]

                    # sum auto-correlation
                    X_XT[
                        n_epoch,
                        ch_i * len_trf:(ch_i + 1) * len_trf,
                        ch_j * len_trf:(ch_j + 1) * len_trf
                    ] += XX_eij
                    
                    # if not diagonal, add mirror triangle
                    if ch_i != ch_j:
                        X_XT[
                            n_epoch,
                            ch_j * len_trf:(ch_j + 1) * len_trf,
                            ch_i * len_trf:(ch_i + 1) * len_trf
                        ] += XX_eij
                    
                    # move triangle counter
                    eij += 1
                
                # sum cross-correlation
                if s_min < 0 and s_max >= 0:
                    X_y[n_epoch,:-s_min,ch_i] += Xy[n_epoch,ch_i,:,s_min:].T
                    X_y[n_epoch,len_trf - s_max:,ch_i] += Xy[n_epoch,ch_i,:,:s_max].T
                else:
                    X_y[n_epoch,:,ch_i] += Xy[n_epoch,ch_i,:,s_min:s_max].T
        
        return X_XT, X_y, X_offset, y_offset
    
    def _compute_reg_neighbours(self, n_features: int, n_trf: int, reg_type: Union[str, List]) -> torch.Tensor:
        """Compute regularisation matrix.
        
        Parameters
        ----------
        n_features : int
            Number of features in X.
        n_trf : int
            Number of timepoints we estimate.
        reg_type : Union[str, List]
            Regularisation type (either 'ridge' or 'laplacian' or tuple thereof).
        
        Returns
        -------
        reg : torch.Tensor
            Regularisation matrix.
        """
        
        # check reg_type
        if isinstance(reg_type, str):
            reg_type = (reg_type,) * 2
        
        if len(reg_type) != 2:
            raise ValueError(f'reg_type must be of length two, but got {len(reg_type)}.')
        
        for r in reg_type:
            if r not in ['ridge', 'laplacian']:
                raise ValueError(f'Unknown reg_type. Expected ridge or laplacian, but got {r}.')
        
        # setup regularisation
        reg_time = (reg_type[0] == 'laplacian') and (n_trf > 1)
        reg_fts = (reg_type[1] == 'laplacian') and (n_features > 1)
        
        # handle identity case
        if not reg_time and not reg_fts:
            return torch.eye(n_features * n_trf)
        
        # handle time laplacian
        if reg_time:
            reg = torch.eye(n_trf).view(-1)
            stride = n_trf + 1
            reg[1::stride] += -1
            reg[n_trf::stride] += -1
            reg[n_trf + 1:-n_trf - 1:stride] += 1
            reg = reg.view(n_trf, n_trf)
            args = [reg] * n_features
            reg = torch.block_diag(*args)
        else:
            reg = torch.zeros((n_trf * n_features,) * 2)
        
        # handle feature laplacian
        if reg_fts:
            block = n_trf * n_trf
            row_offset = block * n_features
            stride = n_trf * n_features + 1
            shape = reg.shape
            reg = reg.view(-1)
            reg[n_trf:-row_offset:stride] += -1
            reg[n_trf + row_offset :: stride] += 1
            reg[row_offset:-n_trf:stride] += -1
            reg[: -(n_trf + row_offset) : stride] += 1
            reg = reg.view(shape)
        
        return reg
    
    def _fit_corrs(self, X_XT: torch.Tensor, X_y: torch.Tensor, alpha: Optional[float] = None) -> tuple[torch.Tensor, Optional[float]]:
        """Fit the model.
        
        Parameters
        ----------
        X_XT : torch.Tensor
            Auto-correlation matrix of shape (n_samples, n_features * n_trf, n_features * n_trf).
        X_y : torch.Tensor
            Cross-correlation matrix of shape (n_samples, n_trf, n_channels, n_features).
        alpha : Optional[float], default=None
            Regularisation strength to apply. If None, perform LOO.

        Returns
        -------
        coef_ : torch.Tensor
            Estiamted coefficients of shape (n_channels, n_features, n_trf).
        alpha_ : Optional[float]
            If computed, best alpha value used in RidgeCV.
        """
        
        # check dims
        if None in [self.n_features_, self.n_channels_, self.n_trf_]:
            raise ValueError(f'_fit_corrs cannot be called before _compute_corrs.')
        
        # sum over available epochs
        X_XT = X_XT.sum(0)
        X_y = X_y.sum(0)
        
        # reshape X_y
        X_y = torch.permute(X_y, (2, 1, 0))
        X_y = X_y.reshape(self.n_channels_, self.n_features_ * self.n_trf_)
        X_y = X_y.t()
        
        if (self.reg_type == 'ridge') and (self.reg_cv == 'LOO') and (alpha is None):
            # solve via RidgeCV
            est = _RidgeCV_torch(alphas = self.alpha, fit_intercept = True).fit(X_XT, X_y)
            w = est.coef_.reshape((self.n_channels_, self.n_features_, self.n_trf_))
            alpha = est.alpha_
            return w, alpha
        else:
            # compute and apply regularisation
            reg = self._compute_reg_neighbours(self.n_features_, self.n_trf_, self.reg_type).to(X_XT.device)
            mat = X_XT + alpha * reg
            
            # solve
            try:
                # try solving equation
                w = torch.linalg.solve(mat, X_y, left = True)
            except:
                try:
                    # fall back to least squares solution if singular, using GELS (GPU-compatible)
                    w = torch.linalg.lstsq(mat, X_y, driver = 'gels')[0]
                except:
                    # fall back to least squares solution if singular, using GELSY (not GPU-compatible)
                    w = torch.linalg.lstsq(mat, X_y, driver = 'gelsy')[0]
            
            # reshape appropriately
            w = w.T.reshape((self.n_channels_, self.n_features_, self.n_trf_))
        
        return w
    
    def _fit_intercept(self, X_offset: Union[torch.Tensor, float], y_offset: Union[torch.Tensor, float], coef_: torch.Tensor) -> Union[float, torch.Tensor]:
        """Fit the intercept.
        
        Parameters
        ----------
        X_offset : Union[float, torch.Tensor]
            Means of X of shape (n_features,) or float.
        y_offset : Union[float, torch.Tensor]
            Means of y of shape (n_channels,) or float.
        coef_ : torch.Tensor
            Estimated coefficients of shape (n_channels, n_features, n_trf).
        
        Returns
        -------
        intercept : Union[float, torch.Tensor]
            Model intercept, if desired.
        """
        
        # check if we need to fit
        if self.fit_intercept:
            # check shape of offsets
            X_off_sq = X_offset.squeeze()
            if len(X_off_sq.shape) < 1:
                X_off_sq = X_off_sq.unsqueeze(0)
            
            # fit sklearn-style
            return y_offset.squeeze() - X_off_sq @ coef_.sum(-1).T
        
        # otherwise, just return zero
        return 0.0

    def _fit_pattern(self, X: torch.Tensor, y: torch.Tensor, coef_: torch.Tensor) -> Optional[torch.Tensor]:
        """Fit the patterns.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_features, n_timepoints).
        y : torch.Tensor
            Input data of shape (n_samples, n_channels, n_timepoints).
        coef_ : torch.Tensor
            Coefficients of shape (n_channels, n_features, n_trf).
        
        Returns
        -------
        pattern : Optional[torch.Tensor]
            If computed, patterns of shape (n_channels, n_features, n_trf).
        """
        
        if self.patterns:
            # get covariance of X
            S_X = self.cov_.sum(0) / float(X.shape[-1] * X.shape[0] - 1)
            
            # get precision of y
            if y.shape[1] > 1:
                # demean
                z = y.swapaxes(1, 2).reshape((-1, y.shape[1]))
                z = (z - z.mean(0, keepdim = True))
                
                # precision
                P_y = torch.linalg.pinv(torch.cov(z.T))
                
                # pattern
                pattern = S_X @ (coef_.reshape((self.n_features_ * self.n_trf_, self.n_channels_)) @ P_y)
            else:
                # norm
                P_y = 1.0 / float(X.shape[-1] * X.shape[0] - 1)
                
                # pattern
                pattern = S_X @ (self.coef_.reshape((self.n_features_ * self.n_trf_, self.n_channels_)) * P_y)
            
            # reshape patterns
            return pattern.reshape(self.coef_.shape)
        
        return None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "_ReceptiveField_torch":
        """Fit the estimator, optionally with cross-validation over penalties.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_features, n_timepoints).
        y : torch.Tensor
            Input data of shape (n_samples, n_channels, n_timepoints).
        
        Returns
        -------
        rf : _ReceptiveField_torch
            The fit estimator.
        """
        
        # check X
        if len(X.shape) not in [2, 3]:
            raise ValueError(f'`X` must be of shape (n_samples[, n_features], n_timepoints), but got {X.shape}.')
        
        if len(X.shape) == 2:
            X = X[:,None,:]
        
        # check y
        if len(y.shape) not in [2, 3]:
            raise ValueError(f'`y` must be of shape (n_samples[, n_channels], n_timepoints), but got {y.shape}.')
        
        if len(y.shape) == 2:
            y = y[:,None,:]
        
        # check shape match
        if (X.shape[0] != y.shape[0]) or (X.shape[-1] != y.shape[-1]):
            raise ValueError(
                f'`X` (n_samples[, n_features], n_timepoints) and ' +
                f'`y` (n_samples[, n_channels], n_timepoints) ' + 
                f'must match in (n_samples, n_timepoints), but got ' + 
                f'X={(X.shape[0], X.shape[-1])} and ' + 
                f'y={(y.shape[0], y.shape[-1])}.'
            )
        
        # compute auto- and cross-correlations
        X_XT, X_y, X_offset, y_offset = self._compute_corrs(X, y)
        self.cov_ = X_XT
        
        # check LOO case
        if self.reg_cv == 'LOO':
            # run directly and save
            self.coef_, self.alpha_ = self._fit_corrs(X_XT, X_y)
            self.intercept_ = self._fit_intercept(X_offset, y_offset, self.coef_)
            self.pattern_ = self._fit_pattern(X, y, self.coef_)
            
            return self

        # check if we need CV
        if isinstance(self.alpha, float) == False:
            # check device of alpha
            if self.alpha.device != X.device:
                self.alpha = self.alpha.to(X.device)
            
            # check if we have a crossvalidator provided in args
            if hasattr(self.reg_cv, 'split'):
                cv = self.reg_cv
            else:
                cv = KFold(n_splits = self.reg_cv).to_torch()
            
            # setup data container
            oos = []
            
            # loop over alphas
            for a_i, alpha in enumerate(self.alpha):
                oos_i = []
                
                # loop over cv
                for f_i, (train, test) in enumerate(cv.split(X_XT, X_y)):
                    # fit coefficients
                    coef_ = self._fit_corrs(X_XT[train], X_y[train], alpha = alpha)
                    intercept_ = self._fit_intercept(X_offset, y_offset, coef_)
                    
                    # get predictions
                    y_h = self._predict(X[test], coef_, intercept_)
                    
                    # score predictions
                    oos_i.append(
                        pearsonr(
                            y_h.permute(2, 1, 0), y[test].permute(2, 1, 0)
                        ).mean().cpu().item()
                    )
                
                # add score
                score_i = torch.mean(torch.tensor(oos_i)).item()
                oos.append(score_i)

            # find best
            oos = torch.tensor(oos, device = X.device)
            best = torch.argmax(oos)
            self.alpha_ = self.alpha[best]
            
            # fit full best model
            self.coef_ = self._fit_corrs(X_XT, X_y, alpha = self.alpha_)
            self.intercept_ = self._fit_intercept(X_offset, y_offset, self.coef_)
            self.pattern_ = self._fit_pattern(X, y, self.coef_)
            
            return self
        
        # otherwise, simply fit single model
        self.alpha_ = self.alpha
        self.coef_ = self._fit_corrs(X_XT, X_y, alpha = self.alpha)
        self.intercept_ = self._fit_intercept(X_offset, y_offset, self.coef_)
        self.pattern_ = self._fit_pattern(X, y, self.coef_)
        
        return self

    def _predict(self, X: torch.Tensor, coef_: torch.Tensor, intercept_: Union[float, torch.Tensor]) -> torch.Tensor:
        """Make predictions from supplied model.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_features, n_timepoints).
        coef_ : torch.Tensor
            Model coefficients of shape (n_channels, n_features, n_trf).
        intercept_ : Union[float, torch.Tensor]
            If computed, intercepts of shape (n_channels,) or float.
        
        Returns
        -------
        y_h : torch.Tensor
            Predicted responses of shape (n_samples, n_channels, n_timepoints).
        """
        
        # check X
        if len(X.shape) == 2:
            X = X[:,None,:]
        
        # setup safe fft length
        n_fft = 2 ** math.ceil(math.log2(2 * X.shape[-1] - 1))
        
        # take FFTs
        F_x = torch.fft.rfft(X, n = n_fft, dim = -1)
        F_w = torch.fft.rfft(coef_, n = n_fft, dim = -1)
        
        # setup outputs
        y = torch.zeros((X.shape[0], coef_.shape[0], X.shape[-1]), device = X.device)
        
        # loop over features
        for i in range(coef_.shape[0]):
            # do the frequency-domain convolution
            x_y = torch.fft.irfft(F_x * F_w[i,None], n = n_fft, dim = -1).sum(1)
            
            # account for the fact that part of our coefficients may be negative
            x_y = x_y[...,max(-self.s_min, 0):X.shape[-1]+max(-self.s_min, 0)]
            
            # account for the fact that our coefficients may not start at t_min=0, but at e.g. t_min=0.2
            y[:,i,max(self.s_min, 0):] += x_y[...,:x_y.shape[-1]-max(self.s_min, 0)]
        
        # add intercept
        if isinstance(intercept_, float):
            y += intercept_
        else:
            if len(intercept_.shape) == 1:
                y += intercept_[None,:,None]
            elif len(intercept_.shape) == 2:
                y += intercept_[...,None]
            else:
                y += intercept_
        
        return y

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions from model.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_features, n_timepoints).
        
        Returns
        -------
        y_h : torch.Tensor
            Predicted responses of shape (n_samples, n_channels, n_timepoints).
        """
        
        return self._predict(X, self.coef_, self.intercept_)

    def score(self, X: torch.Tensor, y: torch.Tensor, metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Make predictions from :math:`X` and score against :math:`y`.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape ``(n_samples, n_features, n_timepoints)``.
        y : torch.Tensor
            Output data of shape ``(n_samples, n_channels, n_timepoints)``.
        metric : Optional[Metric], default=None
            Metric or tuple of metrics to compute.  If ``None``, defaults to :py:attr:`~mvpy.estimators.ReceptiveField.metric_`.
        
        Returns
        -------
        score : torch.Tensor | Dict[str, torch.Tensor]
            Scores of shape ``(n_channels, n_timepoints)`` or, for multiple metrics, a dictionary of metric names and scores of shape ``(n_channels, n_timepoints)``.
        
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
    
    def clone(self) -> "_ReceptiveField_torch":
        """Clone this class.
        
        Returns
        -------
        rf : _ReceptiveField_torch
            A clone of this class.
        """
        
        return _ReceptiveField_torch(
            self.t_min,
            self.t_max,
            self.fs,
            alpha = self.alpha,
            reg_type = self.reg_type,
            reg_cv = self.reg_cv,
            patterns = self.patterns,
            fit_intercept = self.fit_intercept,
            edge_correction = self.edge_correction
        )

class ReceptiveField(sklearn.base.BaseEstimator):
    """Implements receptive field estimation (for multivariate temporal response functions or stimulus reconstruction).
    
    Generally, mTRF models are described by:
    
    .. math::
        r(t,n) = \\sum_{\\tau} w(\\tau, n) s(t - \\tau) + \\varepsilon
    
    where :math:`r(t,n)` is the reconstructed signal at timepoint :math:`t` for channel :math:`n`, :math:`s(t)` 
    is the stimulus at time :math:`t`, :math:`w(\\tau, n)` is the weight at time delay :math:`\\tau` for channel 
    :math:`n`, and :math:`\\varepsilon` is the error.
    
    SR models are estimated as:
    
    .. math::
        s(t) = \\sum_{n}\\sum_{\\tau} r(t + \\tau, n) g(\\tau, n)
    
    where :math:`s(t)` is the reconstructed stimulus at time :math:`t`, :math:`r(t,n)` is the neural response
    at :math:`t` and lagged by :math:`\\tau` for channel :math:`n`, :math:`g(\\tau, n)` is the weight at 
    time delay :math:`\\tau` for channel :math:`n`.
    
    For more information on mTRF or SR models, see [1]_.
    
    Consequently, this class fundamentally solves the same problem as :py:class:`~mvpy.estimators.TimeDelayed`.
    However, unlike :py:class:`~mvpy.estimators.TimeDelayed`, this approach avoids creating and solving the
    full time-delayed design and outcome matrix. Instead, this approach uses the fact that we are fundamentally
    interested in (de-)convolution, which can be solved efficiently through estimation of auto- and cross-
    correlations in the Fourier domain. For more information on this approach, see [2]_ [3]_ [4]_.
    
    Solving this in the Fourier domain can be extremely beneficial when the number of predictors ``n_features``
    is small, but scales poorly for a higher number of ``n_features`` unless 
    :py:attr:`~mvpy.estimators.ReceptiveField.edge_correction` is explicitly disabled. Generally, we would
    recommend testing both :py:class:`~mvpy.estimators.ReceptiveField` and :py:class:`~mvpy.estimators.TimeDelayed`
    on a realistic subset of the data before deciding for one of the two approaches.
    
    Like :py:class:`~mvpy.estimators.TimeDelayed`, this class will automatically perform inner cross-validation
    if multiple values of :py:attr:`~mvpy.estimators.ReceptiveField.alpha` are supplied.
    
    Parameters
    ----------
    t_min : float
        Minimum time point to fit (unlike TimeDelayed, this is relative to y).
    t_max : float
        Maximum time point to fit (unlike TimeDelayed, this is relative to y). Must be greater than t_min.
    fs : int
        Sampling frequency.
    alpha : int | float | np.ndarray | torch.Tensor | List, default=1.0
        Alpha penalties as float or of shape (n_penalties,). If not float, cross-validation will be employed (see :py:attr:`~mvpy.estimators.ReceptiveField.reg_cv`).
    reg_type : {'ridge', 'laplacian', List}, default='ridge'
        Type of regularisation to employ (either 'ridge' or 'laplacian' or tuple describing ``(time, features)``).
    reg_cv : {int, 'LOO', mvpy.crossvalidation.KFold}, default=5
        If alpha is list or array, what cross-validation scheme should we use? Integers are interpeted as :py:attr:`~mvpy.crossvalidation.KFold.n_splits` for :py:class:`~mvpy.crossvalidation.KFold`. String input ``'LOO'`` will use :py:class:`~mvpy.estimators.RidgeCV` to solve LOO-CV over alphas, but is available only for :py:attr:`~mvpy.estimators.ReceptiveField.reg_type` ``'ridge'``. Alternatively, a cross-validator that exposes a :py:meth:`~mvpy.crossvalidation.KFold.split` method can be supplied.
    patterns : bool, default=False
        Should we estimate the patterns from coefficients and data (useful only for stimulus reconstruction, not mTRF)?
    fit_intercept : bool, default=True
        Should we fit an intercept for this model?
    edge_correction : bool, default=True
        Should we apply edge corrections to auto-correlations?
    
    Attributes
    ----------
    t_min : float
        Minimum time point to fit (unlike TimeDelayed, this is relative to y).
    t_max : float
        Maximum time point to fit (unlike TimeDelayed, this is relative to y). Must be greater than t_min.
    fs : int
        Sampling frequency.
    alpha : int | float | np.ndarray | torch.Tensor | List, default=1.0
        Alpha penalties as float or of shape ``(n_penalties,)``. If not float, cross-validation will be employed (see :py:attr:`~mvpy.estimators.ReceptiveField.reg_cv`).
    reg_type : {'ridge', 'laplacian', List}, default='ridge'
        Type of regularisation to employ (either 'ridge' or 'laplacian' or tuple describing ``(time, features)``).
    reg_cv : {int, 'LOO', mvpy.crossvalidation.KFold}, default=5
        If alpha is list or array, what cross-validation scheme should we use? Integers are interpeted as :py:attr:`~mvpy.crossvalidation.KFold.n_splits` for :py:class:`~mvpy.crossvalidation.KFold`. String input ``'LOO'`` will use :py:class:`~mvpy.estimators.RidgeCV` to solve LOO-CV over alphas, but is available only for :py:attr:`~mvpy.estimators.ReceptiveField.reg_type` ``'ridge'``. Alternatively, a cross-validator that exposes a :py:meth:`~mvpy.crossvalidation.KFold.split` method can be supplied.
    patterns : bool, default=False
        Should we estimate the patterns from coefficients and data (useful only for stimulus reconstruction, not mTRF)?
    fit_intercept : bool, default=True
        Should we fit an intercept for this model?
    edge_correction : bool, default=True
        Should we apply edge corrections to auto-correlations?
    s_min : int
        t_min converted to samples.
    s_max : int
        t_max converted to samples.
    window : np.ndarray | torch.Tensor
        The TRF window ranging from s_min-s_max of shape ``(n_trf,)``.
    n_features_ : int
        Number of features in :math:`X`.
    n_channels_ : int
        Number of channels in :math:`y`.
    n_trf_ : int
        Number of timepoints in the estimated response functions.
    cov_ : np.ndarray | torch.Tensor
        Covariance from auto-correlations of shape ``(n_samples, n_features * n_trf, n_features * n_trf)``.
    coef_ : np.ndarray | torch.Tensor
        Estimated coefficients of shape ``(n_channels, n_features, n_trf)``.
    pattern_ : np.ndarray | torch.Tensor
        If computed, estimated pattern of shape ``(n_channels, n_features, n_trf)``.
    intercept_ : float | np.ndarray | torch.Tensor
        Estimated intercepts of shape ``(n_channels,)`` or ``float``.
    metric_ : mvpy.metrics.r2
        The default metric to use.
    
    See also
    --------
    mvpy.estimators.TimeDelayed : An alternative mTRF/SR estimator that solves the time-expanded design matrix.
    mvpy.crossvalidation.KFold, mvpy.crossvalidation.RepeatedKFold, mvpy.crossvalidation.StratifiedKFold, mvpy.crossvalidation.RepeatedStratifiedKFold : Cross-validation classes for automatically testing multiple values of :py:attr:`~mvpy.estimators.ReceptiveField.alpha`.
    
    Notes
    -----
    For SR models it is recommended to also set :py:attr:`~mvpy.estimators.ReceptiveField.patterns` 
    to ``True`` to estimate not only the coefficients but also the patterns that were actually used for 
    reconstructing stimuli. For more information, see [5]_.
    
    .. warning::
        Unlike :py:class:`~mvpy.estimators.TimeDelayed`, this class expects :py:attr:`~mvpy.estimators.ReceptiveField.t_min` 
        and :py:attr:`~mvpy.estimators.ReceptiveField.t_max` to be causal in :math:`y`. Consequently,
        positive values mean :math:`X(t)` asserts influence over :math:`y(t + \\tau)`. This is in 
        line with MNE's behaviour. 
    
    References
    ----------
    .. [1] Crosse, M.J., Di Liberto, G.M., Bednar, A., & Lalor, E.C. (2016). The multivariate temporal response function (mTRF) toolbox: A MATLAB toolbox for relating neural signals to continuous stimuli. Frontiers in Human Neuroscience, 10, 604. 10.3389/fnhum.2016.00604
    .. [2] Willmore, B., & Smyth, D. (2009). Methods for first-order kernel estimation: Simple-cell receptive fields from responses to natural scenes. Network: Computation in Neural Systems, 14, 553-577. 10.1088/0954-898X_14_3_309
    .. [3] Theunissen, F.E., David, S.V., Singh, N.C., Hsu, A., Vinje, W.E., & Gallant, J.L. (2001). Estimating spatio-temporal receptive fields of auditory and visual neurons from their responses to natural stimuli. Network: Computation in Neural Systems, 12, 289-316. 10.1080/net.12.3.289.316
    .. [4] https://mne.tools/stable/generated/mne.decoding.ReceptiveField.html
    .. [5] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.D., Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96-110. 10.1016/j.neuroimage.2013.10.067
    
    Examples
    --------
    For mTRF estimation, we can do:
    
    >>> import torch
    >>> from mvpy.estimators import ReceptiveField
    >>> ß = torch.tensor([1., 2., 3., 2., 1.])
    >>> X = torch.normal(0, 1, (100, 1, 50))
    >>> y = torch.nn.functional.conv1d(X, ß[None,None,:], padding = 'same')
    >>> y = y + torch.normal(0, 1, y.shape)
    >>> trf = ReceptiveField(-2, 2, 1, alpha = 1e-5)
    >>> trf.fit(X, y).coef_
    tensor([[[0.9912, 2.0055, 2.9974, 1.9930, 0.9842]]])
    
    For stimulus reconstruction, we can do:
    
    >>> import torch
    >>> from mvpy.estimators import ReceptiveField
    >>> ß = torch.tensor([1., 2., 3., 2., 1.])
    >>> X = torch.arange(50)[None,None,:] * torch.ones((100, 1, 50))
    >>> y = torch.nn.functional.conv1d(X, ß[None,None,:], padding = 'same')
    >>> y = y + torch.normal(0, 1, y.shape)
    >>> X, y = y, X
    >>> sr = ReceptiveField(-2, 2, 1, alpha = 1e-3, patterns = True).fit(X, y)
    >>> sr.predict(X).mean(0)[0,:]
    tensor([ 0.2148,  0.7017,  1.4021,  2.3925,  3.5046,  4.4022,  5.4741,  6.4759,
             7.5530,  8.4915,  9.6014,  10.5186, 11.5872, 12.6197, 13.5862, 14.6769,
             15.6523, 16.6765, 17.6622, 18.7172, 19.7117, 20.7994, 21.7023, 22.7885,
             23.8434, 24.7849, 25.8697, 26.8705, 27.8523, 28.9028, 29.9428, 30.9342,
             31.9401, 32.9729, 33.9704, 34.9847, 36.0325, 37.0251, 38.0297, 39.0678,
             40.0847, 41.0827, 42.1410, 43.0924, 44.2115, 45.1548, 41.9511, 45.9482,
             32.2861, 76.4690])
    """
    
    def __new__(self, t_min: float, t_max: float, fs: int, alpha: Union[int, float, np.ndarray, torch.Tensor, List] = 1.0, reg_type: Union[str, List] = 'ridge', reg_cv: Any = 5, patterns: bool = False, fit_intercept: bool = True, edge_correction: bool = True):
        """Obtain a new receptive field estimator.
        
        Parameters
        ----------
        t_min : float
            Minimum time point to fit (unlike TimeDelayed, this is relative to y).
        t_max : float
            Maximum time point to fit (unlike TimeDelayed, this is relative to y). Must be greater than t_min.
        fs : int
            Sampling frequency.
        alpha : int | float | np.ndarray | torch.Tensor | List, default=1.0
            Alpha penalties as float or of shape (n_penalties,). If not float, cross-validation will be employed (see :py:attr:`~mvpy.estimators.ReceptiveField.reg_cv`).
        reg_type : {'ridge', 'laplacian', List}, default='ridge'
            Type of regularisation to employ (either 'ridge' or 'laplacian' or tuple describing ``(time, features)``).
        reg_cv : {int, 'LOO', mvpy.crossvalidation.KFold}, default=5
            If alpha is list or array, what cross-validation scheme should we use? Integers are interpeted as :py:attr:`~mvpy.crossvalidation.KFold.n_splits` for :py:class:`~mvpy.crossvalidation.KFold`. String input ``'LOO'`` will use :py:class:`~mvpy.estimators.RidgeCV` to solve LOO-CV over alphas, but is available only for :py:attr:`~mvpy.estimators.ReceptiveField.reg_type` ``'ridge'``. Alternatively, a cross-validator that exposes a :py:meth:`~mvpy.crossvalidation.KFold.split` method can be supplied.
        patterns : bool, default=False
            Should we estimate the patterns from coefficients and data (useful only for stimulus reconstruction, not mTRF)?
        fit_intercept : bool, default=True
            Should we fit an intercept for this model?
        edge_correction : bool, default=True
            Should we apply edge corrections to auto-correlations?
        """
        
        # convert alpha if required
        if isinstance(alpha, float) or isinstance(alpha, int):
            alpha = torch.tensor([float(alpha)])
        
        if isinstance(alpha, list):
            alpha = torch.tensor(alpha)
        
        # setup parameters
        args = [t_min, t_max, fs]
        kwargs = dict(
            alpha = alpha, 
            reg_type = reg_type, 
            reg_cv = reg_cv, 
            patterns = patterns, 
            fit_intercept = fit_intercept, 
            edge_correction = edge_correction
        )
        
        # check instance type
        if isinstance(alpha, torch.Tensor):
            return _ReceptiveField_torch(*args, **kwargs)
        elif isinstance(alpha, np.ndarray):
            return _ReceptiveField_numpy(*args, **kwargs)

        raise ValueError(f'Unknown type of alpha. Expected int, float, list, np.ndarray, or torch.Tensor, but got {type(alpha)}.')
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> "ReceptiveField":
        """Fit the estimator, optionally with cross-validation over penalties.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of shape ``(n_samples, n_features, n_timepoints)``.
        y : np.ndarray | torch.Tensor
            Input data of shape ``(n_samples, n_channels, n_timepoints)``.
        
        Returns
        -------
        rf : mvpy.estimators._ReceptiveField_numpy | mvpy.estimators._ReceptiveField_torch
            The fitted ReceptiveField estimator.
        """
        
        raise NotImplementedError(f'Method not implemented in base class.')
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Make predictions from model.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of shape ``(n_samples, n_features, n_timepoints)``.
        
        Returns
        -------
        y_h : np.ndarray | torch.Tensor
            Predicted responses of shape ``(n_samples, n_channels, n_timepoints)``.
        """
        
        raise NotImplementedError(f'Method not implemented in base class.')
    
    def score(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], metric: Optional[Union[metrics.Metric, Tuple[metrics.Metric]]] = None) -> Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
        """Make predictions from :math:`X` and score against :math:`y`.
        
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input data of shape ``(n_samples, n_features, n_timepoints)``.
        y : np.ndarray | torch.Tensor
            Output data of shape ``(n_samples, n_channels, n_timepoints)``.
        metric : Optional[Metric | Tuple[Metric]], default=None
            Metric or tuple of metrics to compute.  If ``None``, defaults to :py:attr:`~mvpy.estimators.ReceptiveField.metric_`.
        
        Returns
        -------
        score : np.ndarray | torch.Tensor | Dict[str, np.ndarray] | Dict[str, torch.Tensor]
            Scores of shape ``(n_channels, n_timepoints)`` or, for multiple metrics, a dictionary of metric names and scores of shape ``(n_channels, n_timepoints)``.
        
        .. warning::
            If multiple values are supplied for ``metric``, this function will
            output a dictionary of ``{Metric.name: score, ...}`` rather than
            a stacked array. This is to provide consistency across cases where
            metrics may or may not differ in their output shapes.
        """
        
        raise NotImplementedError(f'Method not implemented in base class.')
    
    def clone(self) -> "ReceptiveField":
        """Clone this class.
        
        Returns
        -------
        rf : ReceptiveField
            The cloned object.
        """
        
        raise NotImplementedError('This method is not implemented in the base class.')