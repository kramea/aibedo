import preprocessing

activity = 'CMIP'
experiment = 'historical'
d_activity = {'historical':'CMIP',
            'ssp585':'ScenarioMIP'}
#activity = 'ScenarioMIP'
#experiment = 'ssp585'
institute = 'BCC'
modelName = 'BCC-CSM2-MR'
lf_query= "activity_id=='{0}' & institution_id=='{1}' & source_id=='{2}' & table_id=='fx' & experiment_id=='{3}' &  member_id=='{4}' & variable_id==".format('GMMIP', institute, modelName,'hist-resIPO','r1i1p1f1')

for experiment in ['ssp585']:
    for member in [ 'r1i1p1f1']:
        activity = d_activity[experiment]
        #member = 'r1i1p1f1'
        preprocessing.preprocess_input(activity, experiment, modelName, institute, member,lf_query=lf_query)
        preprocessing.preprocess_output(activity, experiment, modelName, institute, member)
