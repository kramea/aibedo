AIBEDO: a hybrid AI framework to capture the effects of cloud properties on global circulation and regional climate patterns.
=============================================================================================================================



.. note::

   This project is under active development.


Concept
-------

Clouds play a vital role both in modulating Earth's radiation budget and shaping the coupled circulation of the atmosphere and ocean, driving regional changes in temperature and precipitation. The climate response to clouds is one of the largest uncertainties in state-of-the-art Earth System Models (ESMs) when producing decadal climate projections. This limitation becomes apparent when handling scenarios with large changes in cloud properties, e.g., 1) presence of greenhouse gases->loss of clouds or 2) engineered intervention like cloud brightening->increased cloud reflectivity.

Climate intervention techniques—like marine cloud brightening—that need to be fine-tuned spatiotemporally require thousands of hypothetical scenarios to find optimal strategies. Current ESMs need millions of core hours to complete a single simulation. AIBEDO is a hybrid AI model framework developed to resolve the weaknesses of ESMs by generating rapid and robust multi-decadal climate projections. We will demonstrate its utility using marine cloud brightening scenarios—to avoid climate tipping points and produce optimal intervention strategies.


.. toctree::
   :maxdepth: 3

   
   datasets/index
   physics/index
   architecture/index
   dynamics/index
   mcb/index
   code_usage/index
   interface/index
   reports/index


.. tip::
    This tutorial is available as a `Jupyter notebook <https://github.com/{notebook_path}>`_.

    ..  image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/{notebook_path}
        :alt: Open in Colab

.. toctree::
   :hidden:
   :titlesonly:
   :maxdepth: 1
   :caption: Tutorials

   examples/DEMO.ipynb
