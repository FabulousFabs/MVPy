import numpy as np
import torch
import pytest

from sklearn.linear_model import RidgeCV as RidgeCV_sk
from mvpy.estimators import RidgeCV

from tests._common.assertions import assert_allclose, assert_equalshape, assert_equaldevice
from tests._common.backends import Backend, ALL_BACKENDS, TORCH_BACKENDS
from tests._common.dtypes import to_numpy

from typing import Callable

@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
    indirect = True,
)
@pytest.mark.parametrize(
    "n_samples,n_features",
    [
        pytest.param(10, 20, id = "gram"),
        pytest.param(20, 10, id = "svd"),
    ],
)
@pytest.mark.parametrize(
    "n_targets",
    [
        pytest.param(1, id = "single-target"),
        pytest.param(3, id = "multi-target"),
    ],
)
@pytest.mark.parametrize(
    "fit_intercept",
    [
        pytest.param(True, id = "with_intercept"),
        pytest.param(False, id = "without_intercept")
    ]
)
def test_ridgecv_matches_sklearn(backend: Backend, regression_problem: Callable, n_samples: int, n_features: int, n_targets: int, fit_intercept: bool) -> None:
    """Test whether mvpy.estimators.RidgeCV matches sklearn.linear_model.RidgeCV in fitted values and coefficients.
    """
    
    # obtain data
    X_np, y_np, ß_np = regression_problem(
        n_samples = n_samples,
        n_features = n_features,
        n_targets = n_targets,
    )
    
    # transform data
    X_ts, y_ts, ß_ts = backend.asarrays(X_np, y_np, ß_np, dtype = X_np.dtype)

    # setup alphas
    alphas = tuple(np.logspace(-5, 5, 10))
    alpha_per_target = n_targets > 1

    # setup reference model (sklearn)
    reference = RidgeCV_sk(
        alphas = np.asarray(alphas),
        fit_intercept = fit_intercept,
        alpha_per_target = alpha_per_target,
    ).fit(X_np, y_np)

    # setup test model
    model = RidgeCV(
        alphas = backend.asarray(alphas),
        fit_intercept = fit_intercept,
        alpha_per_target = alpha_per_target,
    ).fit(X_ts, y_ts)

    # obtain model predictions
    observed_yh = to_numpy(
        model.predict(X_ts)
    ).squeeze()
    expected_yh = reference.predict(X_np).squeeze()

    # obtain model coefficients
    observed_ß = to_numpy(model.coef_).squeeze()
    expected_ß = model.coef_.squeeze()
    
    # test output shapes
    assert_equalshape(observed_yh, expected_yh)
    
    # test output values
    assert_allclose(observed_yh, expected_yh, rtol = 1e-5, atol = 1e-7)
    
    # test coefficient shapes
    assert_equalshape(observed_ß, expected_ß)
    
    # test coefficient values
    assert_allclose(observed_ß, expected_ß, rtol = 1e-5, atol = 1e-7)

@pytest.mark.parametrize(
    "backend",
    TORCH_BACKENDS,
    indirect = True,
)
def test_ridgecv_requires_matching_shapes(backend: Backend, regression_problem: Callable) -> None:
    """Test whether mvpy.estimators.RidgeCV throws an appropriate error when n_samples are not constant between X and y during .fit().
    """
    
    # obtain data
    X_np, y_np, ß_np = regression_problem(
        n_samples = 120,
        n_features = 10,
        n_targets = 1,
    )
    
    # transform data
    X_ts, y_ts, ß_ts = backend.asarrays(X_np, y_np, ß_np, dtype = X_np.dtype)

    # setup alphas
    alphas = tuple(np.logspace(-5, 5, 10))
    alpha_per_target = False
    
    # setup test model
    model = RidgeCV(
        alphas = backend.asarray(alphas),
        fit_intercept = True,
        alpha_per_target = alpha_per_target,
    )
    
    # test for value errors
    with pytest.raises(ValueError):
        model.fit(X_ts[:80], y_ts)

@pytest.mark.parametrize(
    "backend",
    TORCH_BACKENDS,
    indirect = True
)
def test_ridgecv_requires_fit_before_predict(backend: Backend, regression_problem: Callable) -> None:
    """Test whether mvpy.estimators.RidgeCV throws an appropriate error when trying to predict from an unfitted model.
    """
    
    # obtain data
    X_np, y_np, ß_np = regression_problem(
        n_samples = 120,
        n_features = 10,
        n_targets = 1,
    )
    
    # transform data
    X_ts, y_ts, ß_ts = backend.asarrays(X_np, y_np, ß_np, dtype = X_np.dtype)
    
    # setup alphas
    alphas = tuple(np.logspace(-5, 5, 10))
    alphas = backend.asarray(alphas)
    
    # setup test model
    model = RidgeCV(alphas = alphas)
    
    # test for value errors
    with pytest.raises(ValueError):
        model.predict(X_ts)

@pytest.mark.parametrize(
    "backend",
    TORCH_BACKENDS,
    indirect = True
)
def test_ridgecv_does_not_mutate_input(backend: Backend, regression_problem: Callable) -> None:
    """Test whether mvpy.estimators.RidgeCV mutates input data.
    """
    
    # obtain data
    X_np, y_np, ß_np = regression_problem(
        n_samples = 120,
        n_features = 10,
        n_targets = 1,
    )
    
    # transform data
    X_ts, y_ts, ß_ts = backend.asarrays(X_np, y_np, ß_np, dtype = X_np.dtype)
    
    # setup alphas
    alphas = tuple(np.logspace(-5, 5, 10))
    alphas = backend.asarray(alphas)
    
    # setup test model
    model = RidgeCV(
        alphas = alphas,
        fit_intercept = True
    )
    
    # save and fit
    X_bf, y_bf = X_ts.clone(), y_ts.clone()
    model.fit(X_ts, y_ts)
    
    # check whether mutation occurred
    assert_allclose(X_bf, X_ts, rtol = 1e-5, atol = 1e-7)
    assert_allclose(y_bf, y_ts, rtol = 1e-5, atol = 1e-7)

@pytest.mark.parametrize(
    "backend",
    TORCH_BACKENDS,
    indirect = True
)
def test_ridgecv_preserves_device(backend: Backend, regression_problem: Callable) -> None:
    """Test whether mvpy.estimators.RidgeCV preserves device through-out fitting.
    """
    
    # obtain data
    X_np, y_np, ß_np = regression_problem(
        n_samples = 120,
        n_features = 10,
        n_targets = 1,
    )
    
    # transform data
    X_ts, y_ts, ß_ts = backend.asarrays(X_np, y_np, ß_np, dtype = X_np.dtype)
    
    # setup alphas
    alphas = tuple(np.logspace(-5, 5, 10))
    alphas = backend.asarray(alphas)
    
    # setup test model
    model = RidgeCV(alphas = alphas).fit(X_ts, y_ts)
    
    # make predictions
    y_hs = model.predict(X_ts)
    
    # check device
    assert_equaldevice(y_hs, y_ts)