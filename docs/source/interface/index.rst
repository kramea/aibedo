.. _aibedo_interface:


Visual Analysis System
======================

Our frontend interactive visual analysis (VA) system lets the climate scientists visualize and directly interact with the trained hybrid AI models. The VA system is developed following the requirements and guidelines expressed by our team of climate experts. The multi-panel design lets the experts load different ESM data, interactively run the trained hybrid model in the backend and visualize the model predictions and inputs using popular geospatial projection schemes. Specific control knobs are provided to facilitate what-if investigation to test different MCB type climate intervention hypothesis directly from our VA system. Users can select a specific geospatial zone and decide to perturb the different input variables using these controls to instantaneously see the predicted outcome of cloud property perturbations.

Figure 1 shows a high-level overview of the VA system layout. The key functionalities of the system are annotated and explained below:

* C1. General Controls:
   #. Timestep: Select the month of the input data to analyze.
   #. Projection: Projection scheme for input and output panels (i.e., V1 and v2). Default: ``natural earth``

* C2. Model Controls:
   #. Run AiBEDO: Execute the trained hybrid AI model with base input is MCB control (C3) is not turned-on, else run the model with MCB settings and explained below.
   #. Clear Data: Clear the output buffers from memory.
* C3. MCB Controls:
   #. Switch: Turn on to activate MCB experiments. Default: ``Off``
   #. Start Time: MCB start month
   #. End Time: MCB end month
   #. Initialize: Create a fresh copy of data for MCB perturbation.
   #. Select Zone: Predefined MCB sites: ``SEP``, ``NEP``, ``SEA``
   #. Select Variables: Select multiple input variables and perturb their values at MCB sites using the corresponding sliders.





.. figure:: images/aibedo_VA_december_v3.png
   :scale: 28 %
   :alt: map to buried treasure
   
   Figure 1 High-level overview of the VA system.
