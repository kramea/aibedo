.. _aibedo_tutorial:

This section provides directions on:

#. generating preproecessed datasets
#. how to use the derived data
#. methods to train the hybrid model
#. tools for postprocessing the results


Preprocessing Techniques
~~~~~~~~~~~~~~~~~~~~~~~~

Our preprocessing code respository can be found `here <https://github.com/kramea/aibedo/tree/preprocessing_march2022/preprocessing>` that consists of several scripts to preprocess various Earth System Model ensembles. For example, the following code block shows a simple method to preprocess CESM2-WACCM model ensemble:

``import preprocessing

#activity = 'CMIP'
#experiment = 'historical'

activity = 'ScenarioMIP'
experiment = 'ssp585'
institute = 'NCAR'
modelName = 'CESM2-WACCM'

for member in ['r2i1p1f1','r3i1p1f1']:
    #member = 'r1i1p1f1'
    preprocessing.preprocess_input(activity, experiment, modelName, institute, member)
    preprocessing.preprocess_output(activity, experiment, modelName, institute, member)
``

Running this script preprocesses the model data to detrend, deseasonalize and normalize (detailed in Datasets section). We have provided similar high-level scripts for the selected ESM models we are using in our training data. 
