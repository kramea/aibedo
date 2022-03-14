import preprocessing

activity = 'CMIP'
experiment = 'historical'

#activity = 'ScenarioMIP'
#experiment = 'ssp585'

modelName = 'FGOALS-g3'
institute = 'CAS'

#lf_query= "activity_id=='{0}' & institution_id=='{1}' & source_id=='{2}' & table_id=='fx' & experiment_id=='{3}' &  member_id=='{4}' & variable_id==".format('CMIP', institute, modelName,'historical','r1i1p1f2')

for member in ['r3i1p1f1']:
    #member = 'r1i1p1f1'
    preprocessing.preprocess_input(activity, experiment, modelName, institute, member)
    preprocessing.preprocess_output(activity, experiment, modelName, institute, member)
