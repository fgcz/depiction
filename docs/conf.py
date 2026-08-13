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
# `refactoring/` and `test-data.md` are repository documentation, not published API docs.
# Without the exclusion, `nox -s docs` fails: each is a document that no toctree references,
# and that warning is an error under `-W`. Anything else added under `docs/` for readers of the
# repository rather than of the docs site needs the same treatment. Note that the reverse also
# holds: `modules/depiction_io/index.md` globs `*`, so a new page dropped beside it is picked
# up with no toctree edit.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "refactoring", "test-data.md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "bioio": ("https://bioio-devs.github.io/bioio/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
