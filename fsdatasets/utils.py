from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def train_val_test_split(X, y, split_ratios=(0.8, 0.1, 0.1), random_state=0):
    train_ratio, val_ratio, test_ratio = split_ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_ratio, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=test_ratio / (test_ratio + val_ratio),
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale(X_train, X_val, X_test, scaling="standard"):
    scaler = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
    }[scaling]
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test
