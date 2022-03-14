import preprocessing

activity = 'CMIP'
experiment = 'historical'
modelName = 'NorESM2-MM'
institute = 'NCC'
member = 'r1i1p1f1'
testvar = 'tas'
preprocessing.test_preprocess(activity, experiment, modelName, institute, member,testvar)