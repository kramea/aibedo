# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    ?= -j auto -WT --keep-going
SPHINXBUILD   ?= sphinx-build
SPHINXPROJ    ?= AIBEDO
SOURCEDIR     ?= source
BUILDDIR      ?= build
PYCMD         ?= python

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html-stable:
	# stable differs from `make html` in two ways:
	# 1) The stable logo is used instead of the unstable logo
	# 2) There will not be a link to the stable docs.
	# See conf.py for more details.
	RELEASE=1 make html

.PHONY: help Makefile docset onnx_supported_aten_ops

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile figures onnx_supported_aten_ops
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@echo "Removing everything under 'build' and 'source/generated'.."
	@rm -rf $(BUILDDIR)/html/ $(BUILDDIR)/doctrees $(SOURCEDIR)/generated $(BUILDDIR)/auto_gen_aten_op_list.csv