from boto3 import Session
from _secrets import AWS_ACCESS_KEY, AWS_SECRET_KEY
# from aibedo.constants import CLIMATE_MODELS_ALL

data_dir = '.'
session = Session(
    aws_access_key_id=AWS_ACCESS_KEY,  # put these in _secrets.py (git-ignored by .gitignore)
    aws_secret_access_key=AWS_SECRET_KEY
)

# Change below which ESMs you want to download
CLIMATE_MODELS_ALL = [
    "CESM2",
    "CESM2-FV2",
    "CESM2-WACCM",
    "CESM2-WACCM-FV2",
    "CMCC-CM2-SR5",
    "CanESM5",
    "E3SM-1-1",
    "FGOALS-g3",
    "GFDL-CM4",
    "GFDL-ESM4",
    "GISS-E2-1-H",
    "MIROC-ES2L",
    "MIROC6",
    "MPI-ESM-1-2-HAM",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "SAM0-UNICON",
]

s3 = session.resource('s3')
aibedo_bucket = s3.Bucket('darpa-aibedo')
files = aibedo_bucket.objects.all()
for ESM in CLIMATE_MODELS_ALL:
    files_esm = [f.key for f in files if (ESM in f.key and len(f.key.split('/')) == 1)]
    for f in files_esm:
        print(f"Downloading {f}")
        aibedo_bucket.download_file(f, f"{data_dir}/{f}")
    print(f"-----------------------------> Downloaded {ESM} data.")
