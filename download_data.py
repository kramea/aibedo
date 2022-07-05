from boto3 import Session
from _secrets import AWS_ACCESS_KEY, AWS_SECRET_KEY
from aibedo_salva.constants import CLIMATE_MODELS_ALL

data_dir = 'data'
session = Session(
    aws_access_key_id=AWS_ACCESS_KEY,  # put these in _secrets.py (git-ignored by .gitignore)
    aws_secret_access_key=AWS_SECRET_KEY
)
s3 = session.resource('s3')
aibedo_bucket = s3.Bucket('darpa-aibedo')
files = aibedo_bucket.objects.all()
for ESM in CLIMATE_MODELS_ALL:
    files_esm = [f.key for f in files if (ESM in f.key and len(f.key.split('/')) == 1)]
    for f in files_esm:
        print(f"Downloading {f}")
        aibedo_bucket.download_file(f, f"{data_dir}/{f}")
    print(f"-----------------------------> Downloaded {ESM} data.")
