from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.pkl"

SPECIES_NAMES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained classification model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run training first."
        )

    return joblib.load(MODEL_PATH)


def predict_data(X):
    """
    Predict Iris species for one or more samples.

    Returns a list of dictionaries containing class IDs,
    species names, and confidence scores.
    """
    features = np.asarray(X, dtype=float)

    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError("Input must have shape (n_samples, 4).")

    if features.shape[0] == 0:
        raise ValueError("At least one sample is required.")

    if not np.isfinite(features).all():
        raise ValueError("Input contains NaN or infinite values.")

    if (features <= 0).any():
        raise ValueError("Flower measurements must be positive.")

    model = load_model()

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    results = []

    for prediction, probability in zip(predictions, probabilities):
        class_id = int(prediction)
        class_index = list(model.classes_).index(class_id)

        results.append({
            "class_id": class_id,
            "species": SPECIES_NAMES[class_id],
            "confidence": round(float(probability[class_index]), 4),
        })

    return results