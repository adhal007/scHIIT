import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'SCHIIT Project'
copyright = '2026, Abhilash Dhal'
author = 'Your Name'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',  # For mathematical formulas
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Mock imports for heavy dependencies (speeds up doc building)
autodoc_mock_imports = [
    'tensorflow', 'torch', 'faiss', 'scanpy', 
    'anndata', 'cellxgene_census', 'tiledbsoma'
]

# Only document public APIs
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'special-members': '__init__',
    'inherited-members': False,
    'show-inheritance': True,
}