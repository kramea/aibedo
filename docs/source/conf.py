# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys
import warnings

from git import Repo

_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_ROOT = os.path.realpath(os.path.join(_PATH_HERE, "..", ".."))
sys.path.insert(0, os.path.abspath(_PATH_ROOT))
sys.path.append(os.path.join(_PATH_ROOT, ".actions"))
sys.path.append(os.path.join(_PATH_ROOT, "examples"))

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
    'nbsphinx',  # notebooks
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    'torchmetrics': ('https://torchmetrics.readthedocs.io/en/latest/', None),
}
intersphinx_disabled_domains = ['std']

napoleon_use_ivar = True
napoleon_custom_sections = [('Returns', 'params_style')]

templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_parsers = {".rst": "restructuredtext", ".txt": "markdown", ".md": "markdown", ".ipynb": "nbsphinx"}

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'


def _get_commit_sha() -> str:
    """Determines the commit sha.
    Returns:
        str: The git commit sha, as a string.
    """
    repo_root = os.path.join(os.path.dirname(__file__), '..', '..')
    repo = Repo(repo_root)
    if repo.is_dirty():
        warning_msg = 'The git repo is dirty. The commit sha for source code links will be incorrect.'
        if os.environ.get('CI', '0') == '0':
            # If developing locally, warn.
            warnings.warn(warning_msg)
        else:
            # If on CI, error.
            raise RuntimeError(warning_msg)
    _commit_sha = repo.commit().hexsha
    return _commit_sha


_COMMIT_SHA = _get_commit_sha()

# Don't show notebook output in the docs
nbsphinx_execute = 'never'

notebook_path = 'kramea/aibedo/blob/' + _COMMIT_SHA + '/{{ env.doc2path(env.docname, base=None) }}'

# Include an "Open in Colab" link at the beginning of all notebooks
nbsphinx_prolog = f"""
.. tip::
    This tutorial is available as a `Jupyter notebook <https://github.com/{notebook_path}>`_.
    ..  image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/{notebook_path}
        :alt: Open in Colab
"""
