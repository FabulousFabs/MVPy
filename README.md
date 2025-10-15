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
from mvpy.preprocessing import Scaler
from mvpy.estimators import TimeDelayed
from mvpy.crossvalidation import cross_val_score
from sklearn.pipeline import make_pipeline

# create dataset
device = 'cuda'
fs = 200
X, y = make_meeg_continuous(fs = fs, device = device)

# setup pipeline for estimation of multivariate temporal response functions
trf = make_pipeline(
    Scaler().to_torch(),
    TimeDelayed(
        -1.0, 0.0, fs, 
        alphas = torch.logspace(-5, 5, 20, device = device)
    )
)

# out-of-sample predictive accuracy
validator, scores = cross_val_score(trf, X, y, verbose = True)
```

Find examples and more information here in the **[documentation](http://mvpy.tools)**.

## Is it really worth it?
...