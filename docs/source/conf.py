import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../../src"))

# Import the problematic module
import planai.cli_optimize_prompt

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PlanAI"
copyright = "2024, Niels Provos"
author = "Niels Provos"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxarg.ext",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_default_options = {
    "inherited-members": False,
}

# Define list of undesired members from BaseModel
undesired_members = [
    "model_fields",
    "model_post_init",
    "model_computed_fields",
    "model_config",
    # Add any other Pydantic BaseModel members you want to exclude
]


def should_skip(app, what, name, obj, skip, options):
    if name in undesired_members:
        return True
    # Skip consume_work method specifically for InitialTaskWorker
    if (
        what == "class"
        and name == "consume_work"
        and obj.__qualname__.startswith("InitialTaskWorker")
    ):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", should_skip)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_static_path = ['_static']
