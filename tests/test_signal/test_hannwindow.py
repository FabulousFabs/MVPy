'''
A series of unit tests for mvpy.signal.hann_window
'''

import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import mvpy as mv

import numpy as np
import torch
import scipy
import sklearn

from typing import Any

# setup tolerance for np.allclose
# we are a bit liberabl here because
# scipy computes only with two digits
# whereas we use the exact value
_ALLCLOSE_RTOL = 5e-2
_ALLCLOSE_ATOL = 5e-2

'''
Test hann_window functions
'''

def test_hann_window_numpy():
    '''
    Make sure hann window function is correct in numpy backend.
    '''
    
    assert np.allclose(mv.signal.hann_window(32, backend = 'numpy'), scipy.signal.windows.hann(32), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(mv.signal.hann_window(50, backend = 'numpy'), scipy.signal.windows.hann(50), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

def test_hann_window_torch():
    '''
    Make sure hann window function is correct in torch backend.
    '''
    
    assert np.allclose(mv.signal.hann_window(32, backend = 'torch').cpu().numpy(), scipy.signal.windows.hann(32), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)
    assert np.allclose(mv.signal.hann_window(50, backend = 'torch').cpu().numpy(), scipy.signal.windows.hann(50), rtol = _ALLCLOSE_RTOL, atol = _ALLCLOSE_ATOL)

'''
Allow direct calls
'''

if __name__ == '__main__':
    pytest.main([__file__])