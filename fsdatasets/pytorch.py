import torch
from torch.utils.data import TensorDataset

from .skfeature import fetch_skfeature
from .synthetic import make_invase
from .uci import fetch_isolet, fetch_mice, fetch_smartphone_activity


class SkfeatureDataset(TensorDataset):
    def __init__(self, root, name, download):
        super().__init__(*map(torch.from_numpy, fetch_skfeature(name, root, download)))


class BASESHOCK(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "BASESHOCK", download)


class PCMAC(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "PCMAC", download)


class RELATHE(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "RELATHE", download)


class COIL20(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "COIL20", download)


class ORL(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "ORL", download)


class OrlRaws10P(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "orlraws10P", download)


class PixRaws10P(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "pixraw10P", download)


class WarpAR10P(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "warpAR10P", download)


class WarpPIE10P(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "warpPIE10P", download)


class Yale(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Yale", download)


class USPS(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "USPS", download)


class ALLAML(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "ALLAML", download)


class Carcinom(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Carcinom", download)


class CLL_SUB_111(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "CLL_SUB_111", download)


class Colon(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "colon", download)


class GLI_85(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "GLI_85", download)


class GLIOMA(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "GLIOMA", download)


class Leukemia(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Leukemia", download)


class Lung(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Lung", download)


class LungDiscrete(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "lung_discrete", download)


class Lymphoma(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Lymphoma", download)


class Nci9(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "nci9", download)


class ProstateGE(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Prostate_GE", download)


class SMK_CAN_187(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "SMK_CAN_187", download)


class TOX_171(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "TOX_171", download)


class Arcene(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "arcene", download)


class Gisette(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "gisette", download)


# class Isolet(SkfeatureDataset):
#     def __init__(self, root, download=False):
#         super().__init__(root, "Isolet", download)


class Madelon(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "madelon", download)


class MiceProtein(TensorDataset):
    def __init__(self, root, download=False):
        x_train, y_train = fetch_mice(root, download=download)
        super().__init__(torch.from_numpy(x_train), y_train)


class Activity(TensorDataset):
    def __init__(self, root, train=True, download=False):
        X, y = fetch_smartphone_activity(root, train=train, download=download)
        super().__init__(torch.from_numpy(X), torch.from_numpy(y))


class Isolet(TensorDataset):
    def __init__(self, root, train=True, download=False):
        X, y = fetch_isolet(root, train=train, download=download)
        super().__init__(torch.from_numpy(X), torch.from_numpy(y))


class Invase(TensorDataset):
    def __init__(self, name, n_samples=20_000, dim=11):
        X, y = make_invase(name, n_samples=n_samples, dim=dim)
        super().__init__(torch.from_numpy(X), torch.from_numpy(y))
