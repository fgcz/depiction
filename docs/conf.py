# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "depiction"
copyright = "2024 ETH Zurich"
author = "Leonardo Schwarz"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# napoleon: the codebase writes Google-style docstrings ("Args:", "Returns:"), which plain
# autodoc renders as a docutils indentation error.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.intersphinx",
]

# `_templates` and `_static` are deliberately not configured: both directories were empty, so
# git never tracked them, and `html_static_path` pointing at a directory that does not exist in
# a fresh clone is a warning -- which `nox -s docs` turns into an error. Re-add either setting
# together with the file that justifies it.
# `refactoring/` is repository documentation, not published API docs; the archive under it
# is also kept byte-identical on purpose and must not be reformatted or parsed.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "refactoring"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "bioio": ("https://bioio-devs.github.io/bioio/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
