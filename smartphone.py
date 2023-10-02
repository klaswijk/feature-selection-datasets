from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np


def download_zip(url, path):
    zip_path, _ = urlretrieve(url)
    with ZipFile(zip_path, "r") as f:
        f.extractall(path)


def fetch_smartphone_activity(path, train=True, download=False):
    path = Path(path, "UCI HAR Dataset")
    if not path.exists() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        download_zip(
            "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
            path,
        )
        zippath = path.joinpath("UCI HAR Dataset.zip")
        with ZipFile(zippath, "r") as f:
            f.extractall(path.parent)
        zippath.unlink()

    if train:
        X = np.loadtxt(path.joinpath("train/X_train.txt"))
        y = np.loadtxt(path.joinpath("train/y_train.txt"))
    else:
        X = np.loadtxt(path.joinpath("test/X_test.txt"))
        y = np.loadtxt(path.joinpath("test/y_test.txt"))
    return X, y


X, y = fetch_smartphone_activity("./data", download=True)
print(X.shape, y.shape)
