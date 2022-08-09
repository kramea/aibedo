.. _aibedo_interface:


Visual Analysis System
======================

Our frontend interactive visual analysis (VA) system lets the climate scientists visualize and directly interact with the trained hybrid AI models. The VA system is developed following the requirements and guidelines expressed by our team of climate experts. The multi-panel design lets the experts load different ESM data, interactively run the trained hybrid model in the backend and visualize the model predictions and inputs using popular geospatial projection schemes. Specific control knobs are provided to facilitate what-if investigation to test different MCB type climate intervention hypothesis directly from our VA system. Users can select a specific geospatial zone and decide to perturb the different input variables using these controls to instantaneously see the predicted outcome of cloud property perturbations. Additional panels for elaborate multivariate visualization using parallel coordinate plots and temporal trend plots are provided to facilitate exploratory data analysis in the VA system. Our system is developed using Plotly's Dash open source platform, which utilizes Flask as the backend engine to run the trained hybrid models.


.. image:: images/aibedo_VA_screenshot.png

   Figure 1 High-level overview of the VA system.