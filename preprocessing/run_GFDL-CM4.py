import preprocessing

activity = 'CMIP'
experiment = 'historical'
d_activity = {'historical':'CMIP',
            'ssp585':'ScenarioMIP'}
#activity = 'ScenarioMIP'
#experiment = 'ssp585'
modelName = 'GFDL-ESM4'
institute = 'NOAA-GFDL'

lf_query= "activity_id=='{0}' & institution_id=='{1}' & source_id=='{2}' & table_id=='fx' & experiment_id=='{3}' &  member_id=='{4}' & variable_id==".format('AerChemMIP', institute, modelName,'ssp370SST','r1i1p1f1')

for member in ['r1i1p1f1']:
    for experiment in d_activity:
        activity = d_activity[experiment]
        #member = 'r1i1p1f1'
        preprocessing.preprocess_input(activity, experiment, modelName, institute, member, lf_query=lf_query,load_ps=True)
        preprocessing.preprocess_output(activity, experiment, modelName, institute, member)
