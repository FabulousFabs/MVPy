'''
A collection of classes and functions to automatically perform cross-validation.
'''

import torch
import numpy as np
import sklearn

from .kfold import _KFold_numpy, _KFold_torch
from .repeatedkfold import _RepeatedKFold_numpy, _RepeatedKFold_torch
from .stratifiedkfold import _StratifiedKFold_numpy, _StratifiedKFold_torch
from .repeatedstratifiedkfold import _RepeatedStratifiedKFold_numpy, _RepeatedStratifiedKFold_torch

from typing import Optional, Union
from collections.abc import Generator

class Validator(sklearn.base.BaseEstimator):
    """
    """
    
    def __init__(self):
        pass
    
    