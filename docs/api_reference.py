"""Configuration for the API reference documentation."""


def _get_guide(*refs, is_developer=False):
    """Get the rst to refer to user/developer guide.

    `refs` is several references that can be used in the :ref:`...` directive.
    """
    if len(refs) == 1:
        ref_desc = f":ref:`{refs[0]}` section"
    elif len(refs) == 2:
        ref_desc = f":ref:`{refs[0]}` and :ref:`{refs[1]}` sections"
    else:
        ref_desc = ", ".join(f":ref:`{ref}`" for ref in refs[:-1])
        ref_desc += f", and :ref:`{refs[-1]}` sections"

    guide_name = "Developer" if is_developer else "User"
    return f"**{guide_name} guide.** See the {ref_desc} for further details."


def _get_submodule(module_name, submodule_name):
    """Get the submodule docstring and automatically add the hook.

    `module_name` is e.g. `sklearn.feature_extraction`, and `submodule_name` is e.g.
    `image`, so we get the docstring and hook for `sklearn.feature_extraction.image`
    submodule. `module_name` is used to reset the current module because autosummary
    automatically changes the current module.
    """
    lines = [
        f".. automodule:: {module_name}.{submodule_name}",
        f".. currentmodule:: {module_name}",
    ]
    return "\n\n".join(lines)


"""
CONFIGURING API_REFERENCE
=========================

API_REFERENCE maps each module name to a dictionary that consists of the following
components:

short_summary (required)
    The text to be printed on the index page; it has nothing to do the API reference
    page of each module.
description (required, `None` if not needed)
    The additional description for the module to be placed under the module
    docstring, before the sections start.
sections (required)
    A list of sections, each of which consists of:
    - title (required, `None` if not needed): the section title, commonly it should
      not be `None` except for the first section of a module,
    - description (optional): the optional additional description for the section,
    - autosummary (required): an autosummary block, assuming current module is the
      current module name.

Essentially, the rendered page would look like the following:

|---------------------------------------------------------------------------------|
|     {{ module_name }}                                                           |
|     =================                                                           |
|     {{ module_docstring }}                                                      |
|     {{ description }}                                                           |
|                                                                                 |
|     {{ section_title_1 }}   <-------------- Optional if one wants the first     |
|     ---------------------                   section to directly follow          |
|     {{ section_description_1 }}             without a second-level heading.     |
|     {{ section_autosummary_1 }}                                                 |
|                                                                                 |
|     {{ section_title_2 }}                                                       |
|     ---------------------                                                       |
|     {{ section_description_2 }}                                                 |
|     {{ section_autosummary_2 }}                                                 |
|                                                                                 |
|     More sections...                                                            |
|---------------------------------------------------------------------------------|

Hooks will be automatically generated for each module and each section. For a module,
e.g., `sklearn.feature_extraction`, the hook would be `feature_extraction_ref`; for a
section, e.g., "From text" under `sklearn.feature_extraction`, the hook would be
`feature_extraction_ref-from-text`. However, note that a better way is to refer using
the :mod: directive, e.g., :mod:`sklearn.feature_extraction` for the module and
:mod:`sklearn.feature_extraction.text` for the section. Only in case that a section
is not a particular submodule does the hook become useful, e.g., the "Loaders" section
under `sklearn.datasets`.
"""

API_REFERENCE = {
    "mvpy": {
        "short_summary": "Settings and information tools.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "__version__"
                ],
            },
        ],
    },
    "mvpy.crossvalidation": {
        "short_summary": "Classes and utility functions to ease cross-validation.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "cross_val_score",
                    "KFold",
                    "RepeatedKFold",
                    "StratifiedKFold",
                    "RepeatedStratifiedKFold",
                    "Validator"
                ],
            }
        ],
    },
    "mvpy.dataset": {
        "short_summary": "Classes and utility functions for loading test datasets.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "make_meeg_categorical",
                    "make_meeg_colours",
                    "make_meeg_continuous",
                    "make_meeg_discrete",
                    "make_meeg_layout"
                ]
            }
        ]
    },
    "mvpy.estimators": {
        "short_summary": "Classes for multivariate estimators.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "B2B",
                    "Classifier",
                    "Covariance",
                    "KernelRidgeCV",
                    "ReceptiveField",
                    "RidgeClassifier",
                    "RidgeCV",
                    "RidgeDecoder",
                    "RidgeEncoder",
                    "RSA",
                    "Sliding",
                    "SVC",
                    "TimeDelayed"
                ]
            }
        ]
    },
    "mvpy.math": {
        "short_summary": "Functions for computing various mathematical metrics.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "accuracy",
                    "cosine",
                    "cosine_d",
                    "cv_euclidean",
                    "cv_mahalanobis",
                    "euclidean",
                    "kernel_linear",
                    "kernel_poly",
                    "kernel_rbf",
                    "kernel_sigmoid",
                    "mahalanobis",
                    "pearsonr",
                    "pearsonr_d",
                    "r2",
                    "rank",
                    "roc_auc",
                    "spearmanr",
                    "spearmanr_d"
                ]
            }
        ]
    },
    "mvpy.metrics": {
        "short_summary": "Various metrics for evaluating estimators.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "Accuracy",
                    "Metric",
                    "R2",
                    "Roc_auc",
                    "score"
                ]
            }
        ]
    },
    "mvpy.preprocessing": {
        "short_summary": "Classes and functions for preprocessing.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "Clamp",
                    "LabelBinariser",
                    "RobustScaler",
                    "Scaler"
                ]
            }
        ]
    },
    "mvpy.signal": {
        "short_summary": "Classes and functions for signal processing.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "hamming_window",
                    "hann_window",
                    "raised_cosine_window"
                ]
            }
        ]
    },
    "mvpy.utilities": {
        "short_summary": "Various auxiliary functions and classes.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "compile.numpy",
                    "compile.torch",
                    "env.get_var",
                    "env.is_enabled",
                    "Progressbar",
                    "version.compare"
                ]
            }
        ]
    },
}


"""
CONFIGURING DEPRECATED_API_REFERENCE
====================================

DEPRECATED_API_REFERENCE maps each deprecation target version to a corresponding
autosummary block. It will be placed at the bottom of the API index page under the
"Recently deprecated" section. Essentially, the rendered section would look like the
following:

|------------------------------------------|
|     To be removed in {{ version_1 }}     |
|     --------------------------------     |
|     {{ autosummary_1 }}                  |
|                                          |
|     To be removed in {{ version_2 }}     |
|     --------------------------------     |
|     {{ autosummary_2 }}                  |
|                                          |
|     More versions...                     |
|------------------------------------------|

Note that the autosummary here assumes that the current module is `sklearn`, i.e., if
`sklearn.utils.Memory` is deprecated, one should put `utils.Memory` in the "entries"
slot of the autosummary block.

Example:

DEPRECATED_API_REFERENCE = {
    "0.24": [
        "model_selection.fit_grid_point",
        "utils.safe_indexing",
    ],
}
"""

DEPRECATED_API_REFERENCE = {}  # type: ignore[var-annotated]