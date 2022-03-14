import preprocessing

activity = 'CMIP'
experiment = 'historical'
institute = 'MOHC'
modelName = 'UKESM1-0-LL'

for member in ['r1i1p1f2','r2i1p1f2','r3i1p1f2','r4i1p1f2','r5i1p1f3']:
    #member = 'r1i1p1f1'
    preprocessing.preprocess_input(activity, experiment, modelName, institute, member)
    preprocessing.preprocess_output(activity, experiment, modelName, institute, member)
