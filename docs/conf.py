# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MVPy'
copyright = '2025, Fabian Schneider'
author = 'Fabian Schneider'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",           # Automatically document your modules
    "sphinx.ext.napoleon",          # Support for Google and NumPy style docstrings
    "sphinx.ext.autosummary",       # Generate summary tables
    "sphinx.ext.mathjax",           # Support for LaTeX 
    "sphinx_autodoc_typehints",     # Better type hint handling
    "sphinx_gallery.gen_gallery",   # For code galleries
]

templates_path = ['_templates']
exclude_patterns = ['_build', 
                    'Thumbs.db', 
                    '.DS_Store', 
                    '__pycache__',
                    'examples/*.ipynb',
                    'examples/*.py',
                    'examples/*.zip',
                    'examples/*.json',
                    'examples/*.md5']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["navbar-icon-links"],
    "navigation_depth": 5
}
html_static_path = ['_static']

from sphinx_gallery.sorting import FileNameSortKey, ExplicitOrder, FileNameSortKey

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],  # Directory where your .py examples live
    "gallery_dirs": ["examples"],  # Where to generate output .rst and images
    'filename_pattern': r'\.py$',  # Adjust as needed to match the right file types
    'ignore_pattern': r'^\.',  # Ignore dotfiles (like .md5, .json, etc.)
    "subsection_order": ExplicitOrder(
        [
            "../examples/rsa/",
            "../examples/decoders/",
            "../examples/encoders/"
        ]
    ),
    'within_subsection_order': FileNameSortKey,
    "capture_repr": ("_repr_html_",),  # Controls how outputs are captured
}