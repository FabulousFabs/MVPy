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
    "sphinx.ext.autodoc",       # Automatically document your modules
    "sphinx.ext.napoleon",      # Support for Google and NumPy style docstrings
    "sphinx.ext.autosummary",   # Generate summary tables
    "sphinx.ext.mathjax",       # Support for LaTeX 
    "sphinx_autodoc_typehints", # Optional: better type hint handling,
    "nbsphinx",                 # Support for Jupyter Notebooks
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["navbar-icon-links"],
}
html_static_path = ['_static']
