import json

import joblib
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.data import load_data, split_data
from src.train import (
    MODEL_PATH,
    METRICS_PATH,
    fit_model,
    evaluate_model,
)


@pytest.fixture(scope="module")
def training_results():
    """Train once and reuse the results across tests."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model, best_params = fit_model(X_train, y_train)

    return model, best_params, X_test, y_test


def test_model_is_decision_tree(training_results):
    model, _, _, _ = training_results

    assert isinstance(model, DecisionTreeClassifier)


def test_model_has_been_fitted(training_results):
    model, _, _, _ = training_results

    assert hasattr(model, "classes_")
    assert hasattr(model, "tree_")


def test_hyperparameter_selection(training_results):
    _, best_params, _, _ = training_results

    assert best_params["max_depth"] in [2, 3, 4, 5]
    assert best_params["min_samples_split"] in [2, 4, 6]
    assert best_params["criterion"] in ["gini", "entropy"]


def test_evaluation_metrics_exist(training_results):
    model, _, X_test, y_test = training_results

    metrics = evaluate_model(model, X_test, y_test)

    expected_metrics = {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    }

    assert set(metrics.keys()) == expected_metrics


def test_evaluation_metrics_in_valid_range(training_results):
    model, _, X_test, y_test = training_results

    metrics = evaluate_model(model, X_test, y_test)

    for value in metrics.values():
        assert 0 <= value <= 1


def test_model_accuracy_above_baseline(training_results):
    model, _, X_test, y_test = training_results

    metrics = evaluate_model(model, X_test, y_test)

    # Three balanced Iris classes give a majority-class baseline
    # of approximately 33%.
    assert metrics["accuracy"] > 0.33


def test_saved_model_exists():
    assert MODEL_PATH.exists()


def test_saved_model_can_predict():
    model = joblib.load(MODEL_PATH)

    predictions = model.predict([[5.1, 3.5, 1.4, 0.2]])

    assert len(predictions) == 1
    assert int(predictions[0]) in {0, 1, 2}


def test_evaluation_report_exists():
    assert METRICS_PATH.exists()


def test_evaluation_report_structure():
    report = json.loads(METRICS_PATH.read_text())

    assert report["model"] == "DecisionTreeClassifier"
    assert "best_parameters" in report
    assert "training_samples" in report
    assert "testing_samples" in report
    assert "metrics" in report