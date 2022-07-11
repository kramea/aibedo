## Generate predictions

Create a conda environment first:

```
conda create --name snetpred python=3.9
conda activate snetpred
```

Then install all the relevant packages:

```
pip install -r requirements.txt
conda install -c conda-forge cartopy
```

Jupyter notebook and python script can be used to generate predictions. 

Download the model weights and input/output files from S3. 

- Change the paths accordingly
- Change the input time slices as desired




