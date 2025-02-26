import numpy as np

def pearsonr_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute pearson correlation.
    
    INPUTS:
        x   -   Vector
        y   -   Vector
    
    OUTPUTS:
        r   -   Correlation
    '''
    
    μ_x, μ_y = x.mean(), y.mean()
    return np.sum((x - μ_x) * (y - μ_y)) / np.sqrt(np.sum((x - μ_x)**2) * np.sum((y - μ_y)**2))

def pearsonr_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute pearson correlations between vectors in x and y.
    
    INPUTS:
        x   -   Matrix (samples x features)
        y   -   Matrix (samples x features)
    
    OUTPUTS:
        r   -   Correlations
    '''
    
    μ_x, μ_y = x.mean(axis = 1, keepdims = True), y.mean(axis = 1, keepdims = True)
    return np.sum((x - μ_x) * (y - μ_y), axis = 1) / np.sqrt(np.sum((x - μ_x)**2, axis = 1) * np.sum((y - μ_y)**2, axis = 1))

def pearsonr_3d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute pearson correlations between vectors in x and y.
    
    INPUTS:
        x   -   Tensor (samples x samples x features)
        y   -   Tensor (samples x samples x features)
    
    OUTPUTS:
        r   -   Correlations
    '''
    
    μ_x, μ_y = x.mean(axis = 2, keepdims = True), y.mean(axis = 2, keepdims = True)
    return np.sum((x - μ_x) * (y - μ_y), axis = 2) / np.sqrt(np.sum((x - μ_x)**2, axis = 2) * np.sum((y - μ_y)**2, axis = 2))

def pearsonr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute 1D, 2D or 3D pearson correlations. Please always
    supply features as the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        r   -   Correlations
    '''
    
    if (1 == np.array([len(x.shape), len(y.shape)])).all(): return pearsonr_1d(x, y)
    elif (2 == np.array([len(x.shape), len(y.shape)])).all(): return pearsonr_2d(x, y)
    elif (3 == np.array([len(x.shape), len(y.shape)])).all(): return pearsonr_3d(x, y)

    raise NotImplementedError