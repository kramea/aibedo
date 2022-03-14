import preprocessing

#activity = 'CMIP'
#experiment = 'historical'
activity = 'ScenarioMIP'
experiment = 'ssp585'
modelName = 'MRI-ESM2-0'
institute = 'MRI'

for member in ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1']:
    #member = 'r1i1p1f1'
    preprocessing.preprocess_input(activity, experiment, modelName, institute, member)
    preprocessing.preprocess_output(activity, experiment, modelName, institute, member)
