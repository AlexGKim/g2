import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# sys.path.insert(0, os.path.abspath('../..'))

print("Python path:", sys.path)
print("Current directory:", os.getcwd())

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'g2'
copyright = '2025, Alex Kim'
author = 'Alex Kim'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
]
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []


autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True,
    'imported-members': True,  # This is key!
    'exclude-members': 'abstractmethod,ABC,quad,dblquad,fft2,ifft2,fftfreq,fftshift,ifftshift,RegularGridInterpolator,dataclass,Any,interp1d,Union,Optional,Tuple,Callable,Dict',
}


# -- Options for EPUB output
epub_show_urls = "footnote"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
