'''
'''

from .accuracy import accuracy
from .cosine import cosine, cosine_d
from .crossvalidated import cv_euclidean, cv_mahalanobis
from .euclidean import euclidean
from .kernel_linear import kernel_linear
from .kernel_rbf import kernel_rbf
from .kernel_poly import kernel_poly
from .kernel_sigmoid import kernel_sigmoid
from .mahalanobis import mahalanobis
from .pearsonr import pearsonr, pearsonr_d
from .r2 import r2
from .rank import rank
from .roc_auc import roc_auc
from .spearmanr import spearmanr, spearmanr_d