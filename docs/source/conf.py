# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Imports ---------------------------------------------------------------------------

import os
from datetime import datetime
from distutils.util import strtobool
from os import path

from importlib_metadata import metadata

# -- Run config ------------------------------------------------------------------------


def get_bool_env_var(name, default=False):
    try:
        return bool(strtobool(os.environ[name]))
    except KeyError:
        return default


run_by_github_actions = get_bool_env_var("GITHUB_ACTIONS")
run_by_rtd = get_bool_env_var("READTHEDOCS")
run_by_ci = run_by_github_actions or run_by_rtd or get_bool_env_var("CI")

# -- Path setup ------------------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

PROJECT_ROOT = path.abspath(path.join(path.abspath(path.dirname(__file__)), "..", ".."))


# -- Project information ---------------------------------------------------------------

meta = metadata("ltt")

project = meta["name"]
author = meta["author"]
copyright = f"{datetime.now().year}, {author}"
release = meta["version"]
version = ".".join(release.split(".")[:2])


# -- General configuration -------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []


# -- Config for intersphinx  -----------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.6", None),
}


# -- Options for Latex / MathJax  ------------------------------------------------------

with open("custom_cmds.tex", "r") as fh:
    custom_cmds = fh.read()

latex_elements = {"preamble": custom_cmds}

mathjax_inline = [r"\(" + custom_cmds, r"\)"]
mathjax_display = [r"\[" + custom_cmds, r"\]"]


# -- Options for HTML output -----------------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
