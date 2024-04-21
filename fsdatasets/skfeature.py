import urllib.request
from pathlib import Path

from scipy.io import loadmat

skfeature_datasets = (
    "BASEHOCK",
    "PCMAC",
    "RELATHE",
    "COIL20",
    "ORL",
    "orlraws10P",
    "pixraw10P",
    "warpAR10P",
    "warpPIE10P",
    "Yale",
    "USPS",
    "ALLAML",
    "Carcinom",
    "CLL_SUB_111",
    "colon",
    "GLI_85",
    "GLIOMA",
    "leukemia",
    "lung",
    "lung_discrete",
    "lymphoma",
    "nci9",
    "Prostate_GE",
    "SMK_CAN_187",
    "TOX_171",
    "arcene",
    "gisette",
    "Isolet",
    "madelon",
)

old_skfeature_datasets = (
    # MATLAB h5 formatted files. Not supported in current implementation
    # "20newsgroups",
    # "Reuters21578",
    "GLA-BRA-180",
)

all_names = skfeature_datasets + old_skfeature_datasets


def fetch_skfeature(name, folder, download=False):
    if name in skfeature_datasets:
        url = f"https://jundongl.github.io/scikit-feature/files/datasets/{name}.mat"
    elif name in old_skfeature_datasets:
        url = f"https://jundongl.github.io/scikit-feature/OLD/datasets/{name}.mat"
    else:
        raise ValueError(f"Unknown dataset '{name}'")
    path = Path(folder, name + ".mat")
    if not path.exists() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    data = loadmat(path)
    return data["X"], data["Y"].ravel()


if __name__ == "__main__":
    for name in all_names:
        X, y = fetch_skfeature(name, "./data", download=True)
        print(name, X.shape, y.shape)
