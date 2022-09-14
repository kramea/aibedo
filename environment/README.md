# Environment Setup
AiBEDO is developed in Python 3.9.

## Create the conda environment
There are multiple options for which environment to use depending on your use case.
In the following commands, you can use a different environment name,
by changing string that comes after the ``-n`` flag in the ``conda env create ...`` lines

### Train (and evaluate)
#### On GPU
    conda env create -f env_train_gpu.yaml -n aibedo
    conda activate aibedo  # activate the environment called 'aibedo'

#### On CPU
    conda env create -f env_train_cpu.yaml -n aibedo_cpu
    conda activate aibedo_cpu  # activate the environment called 'aibedo_cpu'



### Only evaluate and do inference 
The following environment file is for inference only, i.e. if you want to
evaluate (or predict with) a model that has already been trained.

    conda env create -f env_evaluation.yaml -n aibedo_eval
    conda activate aibedo_eval  # activate the environment called 'aibedo_eval'

## Note for jupyter notebooks: 

You need to choose the above environment (e.g. ``aibedo``) as kernel of the jupyter notebook.
If the environment doesn't show up in the list of possible kernels, please do
    
    python -m ipykernel install --user --name aibedo   # change aibedo with whatever environment name you use 

Then, please refresh the notebook page.
