import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
from zlib import crc32

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


def test_set_check(id_, test_ratio=0.2):
                                # Below converts from signed to unsigned
                                # Necessary because lib was changed in python 3.0
    return crc32(np.int64(id_)) & 0xffffffff < (test_ratio * 2**32)


def split_dataset_by_id(data, test_ratio=0.2, id_column="index"):
    if id_column == "index":
        print("WARNING:: You are using the index column as your spliting ID.")
        print("          This is dangerous. Be sure to never remove a column")
        print("          or add a new row without setting its index. Consider")
        print("          using a more stable column for your id")
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

