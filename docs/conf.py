import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'SCHIIT'
copyright = '2025, Abhilash Dhal'
author = 'Abhilash Dhal'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = []  # Empty as in your successful build

# Mock heavy dependencies to avoid installation during doc build
autodoc_mock_imports = [
    'torch', 
    'torchvision',
    'torchaudio',
    'tensorflow',
    'faiss',
    'scanpy', 
    'anndata', 
    'cellxgene_census', 
    'tiledbsoma',
    'sentence_transformers', 
    'transformers',
    'statsmodels',
    'pronto',
]

# Napoleon settings (from your successful build)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings (from your successful build)
add_module_names = True
autodoc_member_order = 'bysource'