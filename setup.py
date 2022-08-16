"""AiBEDO package setup."""

import os
import site
import sys
import textwrap

import setuptools
from setuptools import setup
from setuptools.command.develop import develop as develop_orig

_IS_ROOT = os.getuid() == 0
_IS_USER = '--user' in sys.argv[1:]
_IS_VIRTUALENV = 'VIRTUAL_ENV' in os.environ


# From https://stackoverflow.com/questions/51292333/how-to-tell-from-setup-py-if-the-module-is-being-installed-in-editable-mode
class develop(develop_orig):
    """Override the ``develop`` class to error if attempting an editable install as root."""

    def run(self):
        if _IS_ROOT and (not _IS_VIRTUALENV) and (not _IS_USER):
            raise RuntimeError(
                textwrap.dedent("""\
                    When installing in editable mode as root outside of a virtual environment,
                    please specify `--user`. Editable installs as the root user outside of a virtual environment
                    do not work without the `--user` flag. Please instead run something like: `pip install --user -e .`"""
                                ))
        super().run()


# From https://github.com/pypa/pip/issues/7953#issuecomment-645133255
site.ENABLE_USER_SITE = _IS_USER


def package_files(prefix: str, directory: str, extension: str):
    """Get all the files to package."""
    # from https://stackoverflow.com/a/36693250
    paths = []
    for (path, _, filenames) in os.walk(os.path.join(prefix, directory)):
        for filename in filenames:
            if filename.endswith(extension):
                paths.append(os.path.relpath(os.path.join(path, filename), prefix))
    return paths


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Hide the content between <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN --> and
# <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END --> tags in the README
while True:
    start_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->'
    end_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->'
    start = long_description.find(start_tag)
    end = long_description.find(end_tag)
    if start == -1:
        assert end == -1, 'there should be a balanced number of start and ends'
        break
    else:
        assert end != -1, 'there should be a balanced number of start and ends'
        long_description = long_description[:start] + long_description[end + len(end_tag):]

install_requires = [
    'einops',
    'pyyaml>=6.0,<7',
    'matplotlib>=3.4.2',
    'numpy>=1.21.5,<1.23.0',
    'hydra-core',
    'torchmetrics',
    'pytorch==1.9.0',
    'pytorch-lightning>=1.5.8',
]
extra_deps = dict()

extra_deps['base'] = []

extra_deps['dev'] = [
    # Imports for docs builds and running tests
    # Pinning versions strictly to avoid random test failures.
    # Should manually update dependency versions occassionally.
    'pytest==7.1.0',
    'toml==0.10.2',
    'ipython==7.32.0',
    'ipykernel==6.9.2',
    'jupyter==1.0.0',
    'sphinx==4.4.0',
    # embedding md in rst require docutils>=0.17. See
    # https://myst-parser.readthedocs.io/en/latest/sphinx/use.html?highlight=parser#include-markdown-files-into-an-rst-file
    'docutils==0.17.1',
    'sphinx_markdown_tables==0.0.15',
    'sphinx-argparse==0.3.1',
    'sphinxcontrib.katex==0.8.6',
    'sphinxext.opengraph==0.6.1',
    'sphinxemoji==0.2.0',
    'sphinx-copybutton==0.5.0',
    'myst-parser==0.16.1',
    'sphinx_panels==0.6.0',
    'sphinxcontrib-images==0.9.4',
    'pytest_codeblocks==0.16.1',
    'nbsphinx==0.8.8',
    'pandoc==2.2',
    'pypandoc==1.8.1',
    'GitPython==3.1.27',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

setup(name='aibedo',
      version='0.1',
      author='Kalai Ramea',
      author_email='kramea@parc.com',
      description='',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/kramea/aibedo',
      packages=setuptools.find_packages(exclude=['docker*', 'examples*', 'scripts*', 'tests*']),
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ],
      install_requires=install_requires,
      entry_points={
          'console_scripts': [
              'composer = composer.cli.launcher:main',
              'composer_collect_env = composer.utils.collect_env:main',
          ],
      },
      extras_require=extra_deps,
      dependency_links=['https://developer.download.nvidia.com/compute/redist'],
      python_requires='>=3.7',
      ext_package='aibedo',
      cmdclass={'develop': develop})
