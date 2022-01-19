
#!/usr/bin/env python3
# coding: utf-8

import os, sys


def _download(url, path):
    import requests

    filename = url.split('/')[-1]
    file_path = os.path.join(path, filename)

    if os.path.exists(file_path):
        return file_path

    print('Downloading ' + url)

    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return file_path


def _unzip(path, zfile):
    import zipfile

    if os.path.exists(path+'data_5_all/'):
        return

    print('Unzip ' + zfile)

    zip_ref = zipfile.ZipFile(zfile, 'r')
    zip_ref.extractall(path)
    zip_ref.close()
    os.unlink(path+zfile)


def download(dir_path):
    url = "http://island.me.berkeley.edu/ugscnn/data/climate_sphere_l5.zip"
    # download stations
    _download(url, dir_path)
    _unzip(dir_path,"climate_sphere_l5.zip")


if len(sys.argv) > 2:
    datapath = sys.argv[1]
else:
    datapath = "./"

download(datapath)
