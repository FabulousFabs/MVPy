'''

'''

import sklearn
import numpy as np
import torch

from typing import Union

_ESTIMATORS = ['LedoitWolf']

class _LedoitWolf_numpy(sklearn.base.BaseEstimator):
    '''
    '''
    
    def __init__(self):
        '''
        '''
        
        pass
    
    def fit(self, X: np.ndarray):
        '''
        '''
        
        return self
    
    def transform(self, X: np.ndarray):
        '''
        '''
        
        return self
    
    def fit_transform(self, X: np.ndarray):
        '''
        '''
        
        return self

class _LedoitWolf_torch(sklearn.base.BaseEstimator):
    '''
    '''
    
    def __init__(self):
        '''
        '''
        
        pass
    
    def fit(self, X: torch.Tensor):
        '''
        '''
        
        return self
    
    def transform(self, X: torch.Tensor):
        '''
        '''
        
        return self
    
    def fit_transform(self, X: torch.Tensor):
        '''
        '''
        
        return self.fit(X).transform(X)

class Covariance(sklearn.base.BaseEstimator):
    '''
    '''
    
    def __init__(self, method: str = _ESTIMATORS[0]):
        '''
        '''
        
        if method not in _ESTIMATORS:
            raise ValueError(f'Unknown covariance estimation method {method}. Available methods: {_ESTIMATORS}.')

        self.method = method
    
    def fit(self, X: Union[np.ndarray, torch.Tensor]):
        '''
        '''
        
        if isinstance(X, torch.Tensor) & self.method == 'LedoitWolf':
            return _LedoitWolf_torch().fit(X)
        elif isinstance(X, np.ndarray) & self.method == 'LedoitWolf':
            return _LedoitWolf_numpy().fit(X)
        
        return self