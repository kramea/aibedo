# Configuration file for the Sphinx documentation builder.

# -- Project information
import ast
import importlib
import inspect
import os
import sys
import types
import warnings
import sphinx as sphinx
from typing import Dict, Any
from git import Repo

log = sphinx.util.logging.getLogger(__name__)

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
    'sphinx.ext.extlinks',  # For external links
    'sphinx.ext.linkcode',  # For source code links
    'sphinx_copybutton', # For copy button of code blocks
    'sphinx_panels',
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

# ----------------------------- For notebooks ------------------------------------------------
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

def _recursive_getattr(obj: Any, path: str):
    parts = path.split('.')
    try:
        obj = getattr(obj, parts[0])
    except AttributeError:
        return None
    path = '.'.join(parts[1:])
    if path == '':
        return obj
    else:
        return _recursive_getattr(obj, path)


def _determine_lineno_of_attribute(module: types.ModuleType, attribute: str):
    # inspect.getsource() does not work with module-level attributes
    # instead, parse the module manually using ast, and determine where
    # the expression was defined
    source = inspect.getsource(module)
    filename = inspect.getsourcefile(module)
    assert filename is not None, f'filename for module {module} could not be found'
    ast_tree = ast.parse(source, filename)
    for stmt in ast_tree.body:
        if isinstance(stmt, ast.Assign):
            if any(isinstance(x, ast.Name) and x.id == attribute for x in stmt.targets):
                return stmt.lineno
    return None


def linkcode_resolve(domain: str, info: Dict[str, str]):
    """Adds links to the GitHub source code in the API Reference."""
    assert domain == 'py', f'unsupported domain: {domain}'
    module_name = info['module']

    # Get the object and determine the line number
    obj_name_in_module = info['fullname']
    module = importlib.import_module(module_name)
    lineno = _determine_lineno_of_attribute(module, obj_name_in_module)
    if lineno is None:
        obj = _recursive_getattr(module, obj_name_in_module)
        if isinstance(obj, property):
            # For properties, return the getter, where it is documented
            obj = obj.fget
        try:
            _, lineno = inspect.getsourcelines(obj)
        except TypeError:
            # `inspect.getsourcelines` does not work on all object types (e.g. attributes).
            # If it fails, it still might be possible to determine the source line through better parsing
            # in _determine_lineno_of_attribute
            pass
    if lineno is None:
        log.debug(f'Could not determine source line number for {module_name}.{obj_name_in_module}.')
        return None
    # Format the link
    filename = module_name.replace('.', '/')
    commit_sha = _COMMIT_SHA
    return f'https://github.com/kramea/aibedo/blob/{commit_sha}/{filename}.py#L{lineno}'