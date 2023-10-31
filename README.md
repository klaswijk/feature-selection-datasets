# Feature Selection Datasets
A collection of common feature selection datasets in pytorch. Provides PyTorch wrappers for all datasets from [scikit-feature](https://github.com/jundongl/scikit-feature).

## Example usage
```python
from skfeature.pytorch import COIL20
dataset = COIL20("./data", download=True)
```
