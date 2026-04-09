import numpy as np
import torch

from ._common import Normal, Categorical, Constant, BackendCase, BACKENDS

from typing import Dict, Tuple

"""setup benchmarks
"""

benchmarks = {
    "metrics_continuous": {
        "cases": {
            "small_2d": {
                "X": Normal((120, 50)), 
                "y": Normal((120, 50)),
                "Σ": Normal((50, 50))
            },
            "medium_2d": {
                "X": Normal((480, 200)), 
                "y": Normal((480, 200)),
                "Σ": Normal((200, 200))
            },
            "large_2d": {
                "X": Normal((1920, 800)), 
                "y": Normal((1920, 800)),
                "Σ": Normal((800, 800)),
            }
        },
        "backends": BACKENDS
    },
    "metrics_continuous_cv": {
        "cases": {
            "xxsmall_2d": {
                "X": Normal((30, 50)), 
                "y": Normal((30, 50)), 
                "Σ": Normal((50, 50)),
            },
            "xsmall_2d": {
                "X": Normal((60, 200)), 
                "y": Normal((60, 200)), 
                "Σ": Normal((200, 200)),
            },
            "small_2d": {
                "X": Normal((120, 800)), 
                "y": Normal((120, 800)), 
                "Σ": Normal((800, 800)),
            },
        },
        "backends": BACKENDS
    },
    "metrics_categorical": {
        "cases": {
            "small_2d": {
                "y_score": Normal((120, 50)), 
                "y_pred": Categorical((120, 50), dtype_np = np.float32, dtype_tr = torch.float32),
                "y_true": Categorical((120, 50), dtype_np = np.float32, dtype_tr = torch.float32)
            },
            "medium_2d": {
                "y_score": Normal((480, 200)), 
                "y_pred": Categorical((480, 200), dtype_np = np.float32, dtype_tr = torch.float32),
                "y_true": Categorical((480, 200), dtype_np = np.float32, dtype_tr = torch.float32)
            },
            "large_2d": {
                "y_score": Normal((1920, 800)), 
                "y_pred": Categorical((1920, 800), dtype_np = np.float32, dtype_tr = torch.float32),
                "y_true": Categorical((1920, 800), dtype_np = np.float32, dtype_tr = torch.float32)
            }
        },
        "backends": BACKENDS
    },
    "kernels": {
        "cases": {
            "small_2d": {
                "X": Normal((120, 50)), 
                "γ": Constant(1.0),
                "coef0": Constant(0.0),
                "degree": Constant(2.0)
            },
            "medium_2d": {
                "X": Normal((480, 200)), 
                "γ": Constant(1.0),
                "coef0": Constant(0.0),
                "degree": Constant(2.0)
            },
            "large_2d": {
                "X": Normal((1920, 800)), 
                "γ": Constant(1.0),
                "coef0": Constant(0.0),
                "degree": Constant(2.0)
            }
        },
        "backends": BACKENDS
    },
}

"""continuous benchmarks for:
 - mvpy.math.cosine
 - mvpy.math.cosine_d
 - mvpy.math.euclidean
 - mvpy.math.mahalanobis
 - mvpy.math.pearsonr
 - mvpy.math.pearsonr_d
 - mvpy.math.rank
 - mvpy.math.r2
 - mvpy.math.spearmanr
 - mvpy.math.spearmanr_d
"""

from mvpy.math import cosine, cosine_d, euclidean, \
                      mahalanobis, pearsonr, pearsonr_d, \
                      rank, r2, spearmanr, spearmanr_d

class TimeMetricsContinuous(BackendCase):
    name = "metrics_continuous"
    cases = benchmarks[name]['cases']
    backends = benchmarks[name]['backends']
    
    params = [
        list(cases.keys()), backends
    ]
    
    def setup_cache(self):
        return self._build_cache()
    
    def time_cosine(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        cosine(self.data["X"], self.data["y"])
    
    def peakmem_cosine(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        cosine(self.data["X"], self.data["y"])

    def time_cosine_d(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        cosine_d(self.data["X"], self.data["y"])
    
    def peakmem_cosine_d(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        cosine_d(self.data["X"], self.data["y"])
    
    def time_euclidean(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        euclidean(self.data["X"], self.data["y"])
    
    def peakmem_euclidean(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        euclidean(self.data["X"], self.data["y"])

    def time_mahalanobis(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        mahalanobis(self.data["X"], self.data["y"], self.data["Σ"])
    
    def peakmem_mahalanobis(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        mahalanobis(self.data["X"], self.data["y"], self.data["Σ"])
    
    def time_pearsonr(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        pearsonr(self.data["X"], self.data["y"])
    
    def peakmem_pearsonr(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        pearsonr(self.data["X"], self.data["y"])
    
    def time_pearsonr_d(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        pearsonr_d(self.data["X"], self.data["y"])
    
    def peakmem_pearsonr_d(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        pearsonr_d(self.data["X"], self.data["y"])
    
    def time_rank(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        rank(self.data["X"])
    
    def peakmem_rank(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        rank(self.data["X"])
    
    def time_r2(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        r2(self.data["X"], self.data["y"])
    
    def peakmem_r2(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        r2(self.data["X"], self.data["y"])
    
    def time_spearmanr(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        spearmanr(self.data["X"], self.data["y"])
    
    def peakmem_spearmanr(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        spearmanr(self.data["X"], self.data["y"])
    
    def time_spearmanr_d(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        spearmanr_d(self.data["X"], self.data["y"])
    
    def peakmem_spearmanr_d(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        spearmanr_d(self.data["X"], self.data["y"])

"""continuous benchmarks for crossvalidated operators:
 - mvpy.math.cv_euclidean
 - mvpy.math.cv_mahalanobis
"""

from mvpy.math import cv_euclidean, cv_mahalanobis

class TimeMetricsContinuousCV(BackendCase):
    name = "metrics_continuous_cv"
    cases = benchmarks[name]['cases']
    backends = benchmarks[name]['backends']
    
    params = [
        list(cases.keys()), backends
    ]
    
    def setup_cache(self):
        return self._build_cache()
    
    def time_cv_euclidean(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        cv_euclidean(self.data["X"], self.data["y"])
    
    def peakmem_cv_euclidean(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        cv_euclidean(self.data["X"], self.data["y"])

    def time_cv_mahalanobis(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        cv_mahalanobis(self.data["X"], self.data["y"], self.data["Σ"])
    
    def peakmem_cv_mahalanobis(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        cv_mahalanobis(self.data["X"], self.data["y"], self.data["Σ"])

"""categorical benchmarks for:
 - mvpy.math.accuracy
 - mvpy.math.roc_auc
"""

from mvpy.math import accuracy, roc_auc

class TimeMetricsCategorical(BackendCase):
    name = "metrics_categorical"
    cases = benchmarks[name]['cases']
    backends = benchmarks[name]['backends']
    
    params = [
        list(cases.keys()), backends
    ]
    
    def setup_cache(self):
        return self._build_cache()
    
    def time_accuracy(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        accuracy(self.data["y_true"], self.data["y_pred"])
    
    def peakmem_accuracy(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        accuracy(self.data["y_true"], self.data["y_pred"])

    def time_roc_auc(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        roc_auc(self.data["y_true"], self.data["y_score"])
    
    def peakmem_roc_auc(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        roc_auc(self.data["y_true"], self.data["y_score"])


"""kernel benchmarks for:
 - mvpy.math.kernel_linear
 - mvpy.math.kernel_poly
 - mvpy.math.kernel_rbf
 - mvpy.math.kernel_sigmoid
"""

from mvpy.math import kernel_linear, kernel_poly, kernel_rbf, kernel_sigmoid

class TimeKernels(BackendCase):
    name = "kernels"
    cases = benchmarks[name]['cases']
    backends = benchmarks[name]['backends']
    
    params = [
        list(cases.keys()), backends
    ]
    
    def setup_cache(self):
        return self._build_cache()
    
    def time_kernel_linear(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        kernel_linear(self.data["X"], self.data["X"])
    
    def peakmem_kernel_linear(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        kernel_linear(self.data["X"], self.data["X"])

    def time_kernel_poly(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        kernel_poly(self.data["X"], self.data["X"], self.data["γ"], self.data["coef0"], self.data["degree"])
    
    def peakmem_kernel_poly(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        kernel_poly(self.data["X"], self.data["X"], self.data["γ"], self.data["coef0"], self.data["degree"])

    def time_kernel_rbf(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        kernel_rbf(self.data["X"], self.data["X"], self.data["γ"])
    
    def peakmem_kernel_rbf(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        kernel_rbf(self.data["X"], self.data["X"], self.data["γ"])

    def time_kernel_sigmoid(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        kernel_sigmoid(self.data["X"], self.data["X"], self.data["γ"], self.data["coef0"])
    
    def peakmem_kernel_sigmoid(self, cache: Dict, case: Tuple[int], backend: str) -> None:
        kernel_sigmoid(self.data["X"], self.data["X"], self.data["γ"], self.data["coef0"])