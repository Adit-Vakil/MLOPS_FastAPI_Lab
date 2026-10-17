from pathlib import Path
import json
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from .data import load_data, split_data
except ImportError:
    from data import load_data, split_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "iris_model.pkl"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"


def fit_model(X_train, y_train):
    """Tune a Decision Tree using stratified cross-validation."""

    parameter_grid = {
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 4, 6],
        "criterion": ["gini", "entropy"],
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=parameter_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    """Calculate classification metrics on unseen test data."""

    predictions = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model, best_params = fit_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    report = {
        "model": "DecisionTreeClassifier",
        "best_parameters": best_params,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "metrics": metrics,
    }

    METRICS_PATH.write_text(json.dumps(report, indent=4))

    print(json.dumps(report, indent=4))
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Evaluation report saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()