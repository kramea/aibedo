# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'AIBEDO'
copyright = '2021, Ramea'
author = 'Ramea'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',  # Supports Google-style docstrings
]


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}
intersphinx_disabled_domains = ['std']

napoleon_use_ivar = True
napoleon_custom_sections = [('Returns', 'params_style')]

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
