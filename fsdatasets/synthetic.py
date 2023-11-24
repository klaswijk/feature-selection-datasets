import numpy as np


def syn1(x):
    return x[:, 0] * x[:, 1]


def syn2(x):
    return x[:, 2] ** 2 + x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2 - 4


def syn3(x):
    return -10 * np.sin(2 * x[:, 6]) + 2 * np.abs(x[:, 7]) + x[:, 8] + np.exp(-x[:, 9])


def syn4(x):
    return np.where(x[:, 10] < 0, syn1(x), syn2(x))


def syn5(x):
    return np.where(x[:, 10] < 0, syn1(x), syn3(x))


def syn6(x):
    return np.where(x[:, 10] < 0, syn2(x), syn3(x))


def make_invase(name, n_samples=20_000, dim=11):
    """6 synthetic datasets from L2X (Chen et al. 2018) and INVASE (Yoon et. al 2019)"""
    rng = np.random.default_rng()
    f = {
        "Syn1": syn1,
        "Syn2": syn2,
        "Syn3": syn3,
        "Syn4": syn4,
        "Syn5": syn5,
        "Syn6": syn6,
    }
    x = rng.normal(0, 1, size=(n_samples, dim))
    logit = np.exp(f[name](x))
    p = 1 / (1 + logit)
    y = rng.binomial(1, p)
    return x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x, y = make_invase("Syn4")
    plt.scatter(x[:, 10], syn1(x), c=y, s=4)
    plt.show()
