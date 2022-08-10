### Queries AWS to find models with requisite variables
### Also finds a path to a landfraction file
### Then puts the information into a csv

import pandas as pd
import intake

col = intake.open_esm_datastore("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.json")

# list of required variables
required_variables = ['rlut','rsut','rsdt','rsutcs','rlutcs','hfss',
                    'hfls','rlus','rsus','rsds','rlds','ps','pr','tas','psl',
                    'rsuscs','rsdscs','rldscs','evspsbl','hus']

#'clwvi','clivi','cl',
# experiments to query
experiments = ['historical','ssp585']
freq = 'Amon'

query = dict(experiment_id=experiments,
                 table_id=freq,
                 variable_id=required_variables)
# run search
col_subset = col.search(**query)

l_forprocess = []

grouped = col_subset.df.groupby(['source_id','member_id','experiment_id'])
for gr in grouped['variable_id'].nunique().index:   # for each model, member pair
    # get variables
    x = grouped.get_group(gr)['variable_id'].to_list()
    # compare available variables to required variables
    missing = [var for var in required_variables if not var in x]
    # if there are no missing variables, include the model,member pair in the analysis
    if missing == []:
        l_forprocess.append(gr)

# get unique models
l_models = list(set([gr[0] for gr in l_forprocess]))

print(l_models)
excl = []
d_lfquery = {}
# find landfraction file for each model (typically only one, and the experiment id varies)
for model in l_models:
    print(model)
    # find any dataset with the sftlf variable
    query = dict(source_id=model,
             variable_id=['sftlf'])
    lf_col_subset = col.search(**query)
    if len(lf_col_subset.df) > 0:
        # just need the first result
        info = lf_col_subset.df.iloc[0]
        d_lfquery[model] = "source_id=='{0}' & experiment_id=='{1}' & member_id=='{2}' & variable_id=='sftlf' & table_id=='fx'".format(info['source_id'],info['experiment_id'],info['member_id'])
    else:
        print(model + ' has no landfraction data')
        excl.append(model)
for model in excl:
    l_models.remove(model)

# write to a pandas readable csv
print(l_models)
DF = pd.DataFrame([[gr[0],gr[1],gr[2],d_lfquery[gr[0]]] for gr in l_forprocess if gr[0] in l_models],
                columns = ['model','member','experiment','lfquery'])

DF.to_csv('toprocess_no_cl.csv')