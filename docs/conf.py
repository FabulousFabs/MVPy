import os, sys
# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("sphinxext"))

from pathlib import Path
import jinja2
from github_link import make_linkcode_resolve

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MVPy'
copyright = '2025, Fabian Schneider'
author = 'Fabian Schneider'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",       # Generate summary tables
    "sphinx.ext.autodoc",           # Automatically document your modules
    "sphinx.ext.linkcode",          # Make it so [source] links to GitHub
    "numpydoc",                     # numpy docstring support
    "sphinx.ext.mathjax",           # Support for LaTeX 
    "sphinx_gallery.gen_gallery",   # For code galleries
    "sphinxcontrib.sass",
    "sphinx_remove_toctrees",       # Excessive toctree removal
    "sphinx_design",
    "autoshortsummary",
    "doi_role",
    "dropdown_anchors",
    "override_pst_pagetoc"
]

templates_path = [
    'templates'
]

exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store', 
    '__pycache__',
    'examples/*.ipynb',
    'examples/*.py',
    'examples/*.zip',
    'examples/*.json',
    'examples/*.md5'
]


# -- Configure numpydoc -------------------------------------------------------
# We do not need the table of class members because `sphinxext/override_pst_pagetoc.py`
# will show them in the secondary sidebar
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False

# We want in-page toc of class members instead of a separate page for each entry
numpydoc_class_members_toctree = False

# -- Configure sphinx.ext.linkcode -------------------------------------------
# This allows it to link [source] directly to GitHub.
linkcode_resolve = make_linkcode_resolve(
    "mvpy",
    (
        "https://github.com/FabulousFabs/"
        "mvpy/blob/{revision}/"
        "{package}/{path}#L{lineno}"
    ),
)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "sidebar_includehidden": True,
    "use_edit_page_button": False,
    "external_links": [],
    "icon_links_label": "Icon Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/FabulousFabs/mvpy",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "text": "MVPy",
        "image_light": "images/mvpy-icon.png",
        "image_dark": "images/mvpy-icon.png"
    },
    "navigation_depth": 3,
    "collapse_navigation": False,
    "show_nav_level": 1,
    "show_toc_level": 1,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "header_dropdown_text": "More",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
}

html_static_path = ["images", "css", "js"]

html_title = "MVPy"
html_short_title = "MVPy"

# Additional JS files
html_js_files = [
    "scripts/dropdown.js",
    "scripts/version-switcher.js",
    "scripts/sg_plotly_resize.js",
]

# Compile scss files into css files using sphinxcontrib-sass
sass_src_dir, sass_out_dir = "scss", "css/styles"
sass_targets = {
    f"{file.stem}.scss": f"{file.stem}.css"
    for file in Path(sass_src_dir).glob("*.scss")
}

# Additional CSS files, should be subset of the values of `sass_targets`
html_css_files = ["styles/colors.css", "styles/custom.css"]

def add_js_css_files(app, pagename, templatename, context, doctree):
    """Load additional JS and CSS files only for certain pages.

    Note that `html_js_files` and `html_css_files` are included in all pages and
    should be used for the ones that are used by multiple pages. All page-specific
    JS and CSS files should be added here instead.
    """
    if pagename == "api/index":
        # External: jQuery and DataTables
        app.add_js_file("https://code.jquery.com/jquery-3.7.0.js")
        app.add_js_file("https://cdn.datatables.net/2.0.0/js/dataTables.min.js")
        app.add_css_file(
            "https://cdn.datatables.net/2.0.0/css/dataTables.dataTables.min.css"
        )
        # Internal: API search initialization and styling
        app.add_js_file("scripts/api-search.js")
        app.add_css_file("styles/api-search.css")
    elif pagename == "index":
        app.add_css_file("styles/index.css")
    elif pagename.startswith("modules/generated/"):
        app.add_css_file("styles/api.css")

# -- Options for Gallery output ------------------------------------------------
from sphinx_gallery.sorting import FileNameSortKey, ExplicitOrder, FileNameSortKey
sphinx_gallery_conf = {
    "doc_module": ("mvpy",),
    "backreferences_dir": os.path.join("modules", "generated"),
    "show_memory": False,
    "reference_url": {"sklearn": None},
    "examples_dirs": ["../examples"],  # Directory where your .py examples live
    "gallery_dirs": ["auto_examples"],  # Where to generate output .rst and images
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
    "capture_repr": ("_repr_html_",),  # Controls how outputs are captured,
    "inspect_global_variables": False,
    "remove_config_comments": True,
    "plot_gallery": "True",
    "recommender": {"enable": True, "n_examples": 4, "min_df": 12},
}

# -- Configure autosummary ------------------------------------------------------
autosummary_generate = True

def skip_sklearn_inherited(app, what, name, obj, skip, options):
    mod = getattr(obj, "__module__", "") or ""
    # skip any member coming from sklearn.* (tweak predicate as you like)
    if mod.startswith("sklearn."):
        return True  # skip it
    return skip  # fall back to default decision

def setup(app):
    # triggered just before the HTML for an individual page is created
    app.connect("html-page-context", add_js_css_files)
    
    # skip sklearn methods in autosummary
    app.connect("autodoc-skip-member", skip_sklearn_inherited)

# -- Convert templates to rst ---------------------------------------------------
from api_reference import API_REFERENCE, DEPRECATED_API_REFERENCE

# Define the templates and target files for conversion
# Each entry is in the format (template name, file name, kwargs for rendering)
rst_templates = [
    ("index", "index", {}),
    (
        "api/index",
        "api/index",
        {
            "API_REFERENCE": sorted(API_REFERENCE.items(), key = lambda x: x[0]),
            "DEPRECATED_API_REFERENCE": sorted(
                DEPRECATED_API_REFERENCE.items(), key = lambda x: x[0], reverse = True
            ),
        },
    ),
]

# Convert each module API reference page
for module in API_REFERENCE:
    rst_templates.append(
        (
            "api/module",
            f"api/{module}",
            {"module": module, "module_info": API_REFERENCE[module]},
        )
    )

# Convert the deprecated API reference page (if there exists any)
if DEPRECATED_API_REFERENCE:
    rst_templates.append(
        (
            "api/deprecated",
            "api/deprecated",
            {
                "DEPRECATED_API_REFERENCE": sorted(
                    DEPRECATED_API_REFERENCE.items(), key=lambda x: x[0], reverse=True
                )
            },
        )
    )

for rst_template_name, rst_target_name, kwargs in rst_templates:
    # Read the corresponding template file into jinja2
    with (Path(".") / f"{rst_template_name}.rst.template").open(
        "r", encoding="utf-8"
    ) as f:
        t = jinja2.Template(f.read())

    # Render the template and write to the target
    with (Path(".") / f"{rst_target_name}.rst").open("w", encoding="utf-8") as f:
        f.write(t.render(**kwargs))