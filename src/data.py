import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data():
    """Load the Iris dataset and validate its integrity."""
    iris = load_iris()

    X = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.int64)

    if X.shape[0] != y.shape[0]:
        raise ValueError("Feature and target counts do not match.")

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains invalid numerical values.")

    if (X <= 0).any():
        raise ValueError("Flower measurements must be positive.")

    return X, y


def split_data(X, y, test_size=0.25, random_state=42):
    """Perform a reproducible, stratified train/test split."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )