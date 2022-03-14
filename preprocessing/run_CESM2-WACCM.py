import preprocessing

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
