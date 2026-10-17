import numpy as np
import pytest

from src.data import load_data, split_data


def test_dataset_shape():
    X, y = load_data()

    assert X.shape == (150, 4)
    assert y.shape == (150,)


def test_dataset_contains_valid_values():
    X, y = load_data()

    assert np.isfinite(X).all()
    assert (X > 0).all()
    assert set(np.unique(y)) == {0, 1, 2}


def test_train_test_split_sizes():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    assert X_train.shape == (112, 4)
    assert X_test.shape == (38, 4)
    assert len(y_train) == 112
    assert len(y_test) == 38


def test_split_is_reproducible():
    X, y = load_data()

    first_split = split_data(X, y, random_state=42)
    second_split = split_data(X, y, random_state=42)

    for first, second in zip(first_split, second_split):
        np.testing.assert_array_equal(first, second)


def test_stratification_preserves_classes():
    X, y = load_data()
    _, _, y_train, y_test = split_data(X, y)

    assert set(np.unique(y_train)) == {0, 1, 2}
    assert set(np.unique(y_test)) == {0, 1, 2}

    # The Iris dataset contains 50 examples of each species.
    # Stratification should keep class counts balanced.
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)

    assert train_counts.max() - train_counts.min() <= 1
    assert test_counts.max() - test_counts.min() <= 1


@pytest.mark.parametrize("test_size", [0, 1, -0.2, 1.5])
def test_invalid_test_size_rejected(test_size):
    X, y = load_data()

    with pytest.raises(ValueError):
        split_data(X, y, test_size=test_size)


def test_mismatched_features_and_labels_rejected():
    X, y = load_data()

    with pytest.raises(ValueError):
        split_data(X[:-1], y)