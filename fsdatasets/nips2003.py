from pathlib import Path

import numpy as np
from uci import download_zip

urls = {
    "arcene": "https://archive.ics.uci.edu/static/public/167/arcene.zip",
    "dexter": "https://archive.ics.uci.edu/static/public/168/dexter.zip",
    "dorothea": "https://archive.ics.uci.edu/static/public/169/dorothea.zip",
    "gisette": "https://archive.ics.uci.edu/static/public/170/gisette.zip",
    "madelon": "https://archive.ics.uci.edu/static/public/171/madelon.zip",
}


def fetch_nips2003(root, dataset, split="train", download=False):
    path = Path(root, dataset).expanduser()
    if not path.exists() and download:
        download_zip(urls[dataset], path)

    if split == "train":
        X = np.loadtxt(path.joinpath(f"{dataset.upper()}/{dataset}_train.data"))
        y = np.loadtxt(path.joinpath(f"{dataset.upper()}/{dataset}_train.labels"))
    elif split == "valid":
        X = np.loadtxt(path.joinpath(f"{dataset.upper()}/{dataset}_valid.data"))
        y = np.loadtxt(path.joinpath(f"{dataset.upper()}/{dataset}_valid.labels"))
    elif split == "test":
        X = np.loadtxt(path.joinpath(f"{dataset}_test.data"))
        y = np.ones_like(X) * np.inf
    else:
        raise ValueError("split must be one of 'train', 'valid', or 'test'")
    y = np.maximum(0, y)  # -1 -> 0
    return X.astype(np.float32), y.astype(np.float32)


def fetch_arcene(root, split="train", download=False):
    return fetch_nips2003(root, "arcene", split, download)


def fetch_dexter(root, split="train", download=False):
    path = Path(root, "dexter").expanduser()
    if not path.exists() and download:
        download_zip(urls["dexter"], path)

    if split == "train":
        size = 300
        entries = path.joinpath(f"DEXTER/dexter_train.data")
        y = np.loadtxt(path.joinpath(f"DEXTER/dexter_train.labels"))
    elif split == "valid":
        size = 300
        entries = path.joinpath(f"DEXTER/dexter_train.data")
        y = np.loadtxt(path.joinpath(f"DEXTER/dexter_valid.labels"))
    elif split == "test":
        size = 2000
        entries = path.joinpath(f"dexter_test.data")
        y = np.ones(size) * np.inf
    else:
        raise ValueError("split must be one of 'train', 'valid', or 'test'")

    X = np.zeros((size, 20_000))
    with open(entries) as f:
        for i, line in enumerate(f.readlines()):
            for entry in line.strip().split(" "):
                feature, value = map(int, entry.strip().split(":"))
                X[i, feature] = value

    y = np.maximum(0, y)  # -1 -> 0
    return X.astype(np.float32), y.astype(np.float32)


def fetch_dorothea(root, split="train", download=False):
    path = Path(root, "dorothea").expanduser()
    if not path.exists() and download:
        download_zip(urls["dorothea"], path)

    if split == "train":
        size = 800
        entries = path.joinpath(f"DOROTHEA/dorothea_train.data")
        y = np.loadtxt(path.joinpath(f"DOROTHEA/dorothea_train.labels"))
    elif split == "valid":
        size = 350
        entries = path.joinpath(f"DOROTHEA/dorothea_train.data")
        y = np.loadtxt(path.joinpath(f"DOROTHEA/dorothea_valid.labels"))
    elif split == "test":
        size = 800
        entries = path.joinpath(f"dorothea_test.data")
        y = np.ones(size) * np.inf
    else:
        raise ValueError("split must be one of 'train', 'valid', or 'test'")

    X = np.zeros((size, 100_000))
    with open(entries) as f:
        for i, line in enumerate(f.readlines()):
            for entry in line.strip().split(" "):
                feature = int(entry.strip()) - 1
                X[i, feature] = 1

    y = np.maximum(0, y)  # -1 -> 0
    return X.astype(np.float32), y.astype(np.float32)


def fetch_gisette(root, split="train", download=False):
    return fetch_nips2003(root, "gisette", split, download)


def fetch_madelon(root, split="train", download=False):
    return fetch_nips2003(root, "madelon", split, download)


if __name__ == "__main__":
    X, y = fetch_arcene("data", download=True)
    print(X.shape, y.shape)

    X, y = fetch_dexter("data", download=True)
    print(X.shape, y.shape)

    X, y = fetch_dorothea("data", download=True)
    print(X.shape, y.shape)

    X, y = fetch_gisette("data", download=True)
    print(X.shape, y.shape)

    X, y = fetch_madelon("data", download=True)
    print(X.shape, y.shape)
