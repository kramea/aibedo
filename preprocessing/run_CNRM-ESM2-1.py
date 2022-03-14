import preprocessing

#activity = 'CMIP'
#experiment = 'historical'

activity = 'ScenarioMIP'
experiment = 'ssp585'
institute = 'CNRM-CERFACS'
modelName = 'CNRM-ESM2-1'
lf_query= "activity_id=='{0}' & institution_id=='{1}' & source_id=='{2}' & table_id=='fx' & experiment_id=='{3}' &  member_id=='{4}' & variable_id==".format('CMIP', institute, modelName,'amip','r1i1p1f2')

for member in ['r1i1p1f2', 'r2i1p1f2', 'r3i1p1f2', 'r4i1p1f2', 'r5i1p1f2']:
    #member = 'r1i1p1f1'
    preprocessing.preprocess_input(activity, experiment, modelName, institute, member,lf_query=lf_query)
    preprocessing.preprocess_output(activity, experiment, modelName, institute, member)
