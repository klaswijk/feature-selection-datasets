from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np


def download_zip(url, path):
    zip_path, _ = urlretrieve(url)
    with ZipFile(zip_path, "r") as f:
        f.extractall(path)


def fetch_smartphone_activity(root, train=True, download=False):
    path = Path(root, "UCI HAR Dataset").expanduser()
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


def fetch_isolet(root, train=True, download=False):
    path = Path(root, "isolet").expanduser()
    if not path.exists() and download:
        import unlzw3

        download_zip("https://archive.ics.uci.edu/static/public/54/isolet.zip", path)
        train = unlzw3.unlzw(path.joinpath("isolet1+2+3+4.data.Z"))
        with open(path.joinpath("isolet1+2+3+4.data"), "wb") as f:
            f.write(train)
        test = unlzw3.unlzw(path.joinpath("isolet5.data.Z"))
        with open(path.joinpath("isolet5.data"), "wb") as f:
            f.write(test)

    if train:
        X = np.genfromtxt(
            path.joinpath("isolet1+2+3+4.data"),
            delimiter=",",
            usecols=range(0, 617),
        )
        y = np.genfromtxt(
            path.joinpath("isolet1+2+3+4.data"),
            delimiter=",",
            usecols=[617],
        )
    else:
        X = np.genfromtxt(
            path.joinpath("isolet5.data"),
            delimiter=",",
            usecols=range(0, 617),
        )
        y = np.genfromtxt(
            path.joinpath("isolet5.data"),
            delimiter=",",
            usecols=[617],
        )
    return X, y


def fetch_mice(root, download=False):
    path = Path(root, "mice").expanduser()
    if not path.exists() and download:
        from pandas import read_excel

        path.parent.mkdir(parents=True, exist_ok=True)
        download_zip(
            "https://archive.ics.uci.edu/static/public/342/mice+protein+expression.zip",
            path,
        )
        xls_path = path.joinpath("Data_Cortex_Nuclear.xls")
        csv_path = path.joinpath("Data_Cortex_Nuclear.csv")
        df = read_excel(xls_path)
        df.to_csv(csv_path, index=False)

    X = np.genfromtxt(
        path.joinpath("Data_Cortex_Nuclear.csv"),
        delimiter=",",
        skip_header=1,
        usecols=range(1, 78),
    )
    y = np.genfromtxt(
        path.joinpath("Data_Cortex_Nuclear.csv"),
        delimiter=",",
        skip_header=1,
        usecols=[81],
        dtype=str,
    )
    labels = dict(
        zip(
            [
                "c-CS-m",
                "c-CS-s",
                "c-SC-m",
                "c-SC-s",
                "t-CS-m",
                "t-CS-s",
                "t-SC-m",
                "t-SC-s",
            ],
            range(8),
        )
    )
    y = np.vectorize(labels.get)(y)
    return X, y


if __name__ == "__main__":
    X, y = fetch_smartphone_activity("~/datasets", download=True)
    print(X.shape, y.shape)

    X, y = fetch_isolet("~/datasets", download=True)
    print(X.shape, y.shape)

    X, y = fetch_mice("~/datasets", download=True)
    print(X.shape, y.shape)
