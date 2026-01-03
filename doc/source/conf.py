# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "forest_gen"
copyright = "2025, Tomasz Chady, Jakub Markil, Patryk Olszewski, Oskar Winiarski"
author = "Tomasz Chady, Jakub Markil, Patryk Olszewski, Oskar Winiarski"
release = "0.3.7"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_mock_imports = [
    "isaaclab",
    "isaacsim",
    "pxr",
    "trimesh",
    "gymnasium",
    "matplotlib",
    "matplotlib.pyplot",
    "opensimplex",
    "stripe_kit",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
