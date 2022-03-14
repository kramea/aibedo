import preprocessing

#activity = 'CMIP'
#experiment = 'historical'
activity = 'ScenarioMIP'
experiment = 'ssp585'
modelName = 'MIROC6'
institute = 'MIROC'

for member in ['r1i1p1f1','r2i1p1f1','r3i1p1f1']:
    #member = 'r1i1p1f1'
    preprocessing.preprocess_input(activity, experiment, modelName, institute, member)
    preprocessing.preprocess_output(activity, experiment, modelName, institute, member)
