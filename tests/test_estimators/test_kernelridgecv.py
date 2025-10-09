'''
A series of unit tests for mvpy.estimators.RidgeCV
'''

import pytest
import warnings

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import mvpy as mv

import numpy as np
import torch
import scipy

from sklearn.kernel_ridge import KernelRidge

from typing import Any, Dict

# setup threshold for correlation
_THRESHOLD = 0.75

'''
Test KernelRidgeCV estimators
'''

@pytest.mark.parametrize(
    "opts",
    [
        (dict(kernel = 'linear')),
        (dict(kernel = 'poly')),
        (dict(kernel = 'rbf')),
        (dict(kernel = 'sigmoid')),
    ],
    ids = [
        'auto_linear',
        'auto_poly',
        'auto_rbf',
        'auto_sigmoid'
    ]
)
def test_KernelRidgeCV_compare_numpy_torch_sklearn(opts: Dict[str, Any]):
    '''
    Make sure that KernelRidgeCV estimators are consistent between numpy, torch and sklearn.
    '''
    
    # create some data
    ß = np.random.normal(0, 1, size = (128,))
    
    X_np = np.random.normal(0, 1, size = (60, 128))
    y_np = ß @ X_np.T
    
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    y_tr = torch.from_numpy(y_np).to(torch.float32)
    
    a_np = np.logspace(-5, 5, 20)
    a_tr = torch.from_numpy(a_np).to(torch.float32)
    
    # run tests
    ridge_np = mv.estimators.KernelRidgeCV(alphas = a_np, **opts).fit(X_np, y_np)
    ridge_tr = mv.estimators.KernelRidgeCV(alphas = a_tr, **opts).fit(X_tr, y_tr)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        ridge_sk = KernelRidge(alpha = ridge_np.alpha_, **opts).fit(X_np, y_np)
    
    h_np = ridge_np.predict(X_np)
    h_tr = ridge_tr.predict(X_tr)
    h_sk = ridge_sk.predict(X_np)
    
    r_np = mv.math.pearsonr(h_np.T, y_np.T)
    r_tr = mv.math.pearsonr(h_tr.t(), y_tr.t())
    r_npsk = mv.math.pearsonr(h_np.T, h_sk.T)
    r_trsk = mv.math.pearsonr(h_tr.cpu().numpy().T, h_sk.T)
    
    assert (r_np > _THRESHOLD).all(), f"Pearson r: {r_np} < {_THRESHOLD:.2f}"
    assert (r_tr > _THRESHOLD).all(), f"Pearson r: {r_tr} < {_THRESHOLD:.2f}"
    assert (r_npsk > _THRESHOLD).all(), f"Pearson r: {r_npsk} < {_THRESHOLD:.2f}"
    assert (r_trsk > _THRESHOLD).all(), f"Pearson r: {r_trsk} < {_THRESHOLD:.2f}"

@pytest.mark.parametrize(
    "opts",
    [
        (dict(kernel = 'linear')),
        (dict(kernel = 'poly')),
        (dict(kernel = 'rbf')),
        (dict(kernel = 'sigmoid')),
    ],
    ids = [
        'auto_linear',
        'auto_poly',
        'auto_rbf',
        'auto_sigmoid'
    ]
)
def test_RidgeCV_compare_numpy_torch_sklearn_multi_target(opts: Dict[str, Any]):
    '''
    Make sure that KernelRidgeCV estimators are consistent between numpy, torch and sklearn with multiple targets.
    '''
    
    # create some data
    ß = np.random.normal(0, 1, size = (128, 10))
    
    X_np = np.random.normal(0, 1, size = (60, 128))
    y_np = np.array([ß[:,i] @ X_np.T for i in range(ß.shape[1])]).T
    
    X_tr = torch.from_numpy(X_np).to(torch.float32)
    y_tr = torch.from_numpy(y_np).to(torch.float32)
    
    a_np = np.logspace(-5, 5, 20)
    a_tr = torch.from_numpy(a_np).to(torch.float32)
    
    # run tests
    ridge_np = mv.estimators.KernelRidgeCV(alphas = a_np, **opts).fit(X_np, y_np)
    ridge_tr = mv.estimators.KernelRidgeCV(alphas = a_tr, **opts).fit(X_tr, y_tr)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        ridge_sk = KernelRidge(alpha = ridge_np.alpha_, **opts).fit(X_np, y_np)
    
    h_np = ridge_np.predict(X_np)
    h_tr = ridge_tr.predict(X_tr)
    h_sk = ridge_sk.predict(X_np)
    
    r_np = mv.math.pearsonr(h_np.T, y_np.T)
    r_tr = mv.math.pearsonr(h_tr.t(), y_tr.t())
    r_npsk = mv.math.pearsonr(h_np.T, h_sk.T)
    r_trsk = mv.math.pearsonr(h_tr.cpu().numpy().T, h_sk.T)
    
    assert (r_np > _THRESHOLD).all(), f"Pearson r: {r_np} < {_THRESHOLD:.2f}"
    assert (r_tr > _THRESHOLD).all(), f"Pearson r: {r_tr} < {_THRESHOLD:.2f}"
    assert (r_npsk > _THRESHOLD).all(), f"Pearson r: {r_npsk} < {_THRESHOLD:.2f}"
    assert (r_trsk > _THRESHOLD).all(), f"Pearson r: {r_trsk} < {_THRESHOLD:.2f}"

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])