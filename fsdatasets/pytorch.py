import torch
from torch.utils.data import TensorDataset

from .skfeature import fetch_skfeature
from .synthetic import make_invase
from .uci import fetch_isolet, fetch_mice, fetch_smartphone_activity
from .utils import scale, train_val_test_split


class SplitScaleSklearnDataset(TensorDataset):
    def __init__(self, X, y, split="train", scaling=None):
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        if scaling is not None:
            X_train, X_val, X_test = scale(X_train, X_val, X_test, scaling)
        if split == "train":
            super().__init__(torch.from_numpy(X_train), torch.from_numpy(y_train))
        elif split == "val":
            super().__init__(torch.from_numpy(X_val), torch.from_numpy(y_val))
        elif split == "test":
            super().__init__(torch.from_numpy(X_test), torch.from_numpy(y_test))
        else:
            raise ValueError(
                f"Split must be 'train', 'val', or 'test'. Found '{split}'."
            )


class SkfeatureDataset(SplitScaleSklearnDataset):
    def __init__(self, root, name, split="train", download=False, scaling=None):
        super().__init__(
            *fetch_skfeature(name, root, download=download),
            split=split,
            scaling=scaling,
        )


class BASESHOCK(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "BASESHOCK", split=split, download=download, scaling=scaling
        )


class PCMAC(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(root, "PCMAC", split=split, download=download, scaling=scaling)


class RELATHE(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "RELATHE", split=split, download=download, scaling=scaling
        )


class COIL20(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "COIL20", split=split, download=download, scaling=scaling
        )


class ORL(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(root, "ORL", split=split, download=download, scaling=scaling)


class OrlRaws10P(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "orlraws10P", split=split, download=download, scaling=scaling
        )


class PixRaws10P(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "pixraw10P", split=split, download=download, scaling=scaling
        )


class WarpAR10P(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "warpAR10P", split=split, download=download, scaling=scaling
        )


class WarpPIE10P(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "warpPIE10P", split=split, download=download, scaling=scaling
        )


class Yale(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(root, "Yale", split=split, download=download, scaling=scaling)


class USPS(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(root, "USPS", split=split, download=download, scaling=scaling)


class ALLAML(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "ALLAML", split=split, download=download, scaling=scaling
        )


class Carcinom(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "Carcinom", split=split, download=download, scaling=scaling
        )


class CLL_SUB_111(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "CLL_SUB_111", split=split, download=download, scaling=scaling
        )


class Colon(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(root, "colon", split=split, download=download, scaling=scaling)


class GLI_85(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "GLI_85", split=split, download=download, scaling=scaling
        )


class GLIOMA(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "GLIOMA", split=split, download=download, scaling=scaling
        )


class Leukemia(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "Leukemia", split=split, download=download, scaling=scaling
        )


class Lung(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(root, "Lung", split=split, download=download, scaling=scaling)


class LungDiscrete(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "lung_discrete", split=split, download=download, scaling=scaling
        )


class Lymphoma(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "Lymphoma", split=split, download=download, scaling=scaling
        )


class Nci9(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(root, "nci9", split=split, download=download, scaling=scaling)


class ProstateGE(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "Prostate_GE", split=split, download=download, scaling=scaling
        )


class SMK_CAN_187(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "SMK_CAN_187", split=split, download=download, scaling=scaling
        )


class TOX_171(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "TOX_171", split=split, download=download, scaling=scaling
        )


class Arcene(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "arcene", split=split, download=download, scaling=scaling
        )


class Gisette(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "gisette", split=split, download=download, scaling=scaling
        )


# class Isolet(SkfeatureDataset):
#     def __init__(self, root, split="train", download=False, scaling=None):
#         super().__init__(root, "Isolet", split=split, download=download, scaling=scaling)


class Madelon(SkfeatureDataset):
    def __init__(self, root, split="train", download=False, scaling=None):
        super().__init__(
            root, "madelon", split=split, download=download, scaling=scaling
        )


class MiceProtein(SplitScaleSklearnDataset):
    def __init__(self, root, download=False):
        x_train, y_train = fetch_mice(root, download=download)
        super().__init__(torch.from_numpy(x_train), torch.from_numpy(y_train))


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
