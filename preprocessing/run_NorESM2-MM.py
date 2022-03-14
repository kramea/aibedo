import preprocessing

#activity = 'CMIP'
#experiment = 'historical'
activity = 'ScenarioMIP'
experiment = 'ssp585'
modelName = 'NorESM2-MM'
institute = 'NCC'
member = 'r1i1p1f1'

lf_query= "activity_id=='{0}' & institution_id=='{1}' & source_id=='{2}' & table_id=='fx' & experiment_id=='{3}' &  member_id=='{4}' & variable_id==".format('CMIP', institute, modelName,'historical','r2i1p1f1')

preprocessing.preprocess_input(activity, experiment, modelName, institute, member,lf_query=lf_query)
preprocessing.preprocess_output(activity, experiment, modelName, institute, member)