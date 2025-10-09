[![Unit Tests](https://github.com/FabulousFabs/MVPy/workflows/Unit%20tests/badge.svg)](https://github.com/FabulousFabs/MVPy/actions)
[![Unit Tests](https://github.com/FabulousFabs/MVPy/workflows/Documentation/badge.svg)](https://github.com/FabulousFabs/MVPy/actions)


[<img src="./docs/images/mvpy-large.png" width="500px">](http://mvpy.tools/)

**MVPy** is a python package for **m**ulti**v**ariate **p**attern analysis in neuroscience, with direct GPU support and easy-to-use coding patterns.

The project started in 2024 when we found that our planned analyses were prohibitively expensive to run in existing neuroscience frameworks and we needed to find a way to put intensive multivariate computations on GPUs.

It is currently maintained by me, but new contributors are extremely welcome. 

**[Find the documentation here](http://mvpy.tools)**.

## How do I get started?
You can install MVPy directly from GitHub:

```bash
pip install git+https://github.com/FabulousFabs/MVPy.git
```

and get started immediately with familiar sklearn workflows:

```python
import torch
from mvpy.dataset import make_meeg_continuous
from mvpy.estimators import Scaler, TimeDelayed
from mvpy.crossvalidation import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

# create dataset
device = 'cuda'
X, y = make_meeg_continuous(device = device)

# setup validation scheme
kfold = RepeatedKFold(
    n_splits = 5,
    n_repeats = 5
).to_torch()

# setup pipeline for estimation of multivariate temporal response functions
trf = make_pipeline(
    Scaler().to_torch(),
    TimeDelayed(
        -10.0, 10.0, 1.0, 
        alphas = torch.logspace(-5, 5, 20, device = device)
    )
)

# out-of-sample predictive accuracy
oos_r = cross_val_score(trf, kfold)
```

Find examples and more information here in the **[documentation](http://mvpy.tools)**.

## Is it really worth it?
Consider a toy example where we have MEG data `y` of shape `(n_trials, n_channels, n_timepoints)` and a set of regressors `X` of shape `(n_trials, n_features, n_timepoints)` that describe properties of acoustic stimuli (envelopes, edges, etc.). We want to estimate repeated cross-validation accuracy to allow for isolation of variance explained by each predictor through model comparisons:

```python
from mvpy.dataset import make_meeg_continuous

X, y = make_meeg_continuous(device = 'cuda')
print(X.shape, y.shape)
```