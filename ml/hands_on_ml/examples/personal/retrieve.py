import os
import tarfile
import urllib.request
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"

def extract_tar(url, path, tar_name):
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, tar_name)
    urllib.request.urlretrieve(url, tgz_path)
    tgz = tarfile.open(tgz_path)
    tgz.extractall(path=path)
    tgz.close()


def load_csv(path, name):
    csv_path = os.path.join(path, name)
    return pd.read_csv(csv_path)