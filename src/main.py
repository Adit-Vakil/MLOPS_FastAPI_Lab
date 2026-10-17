from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import predict_data, load_model


app = FastAPI(
    title="Iris Classification API",
    description=(
        "A machine learning API for classifying Iris flowers "
        "using a tuned Decision Tree classifier."
    ),
    version="2.0.0",
)


class IrisData(BaseModel):
    sepal_length: float = Field(gt=0, le=15, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(gt=0, le=15, description="Petal length in cm")
    petal_width: float = Field(gt=0, le=10, description="Petal width in cm")


class IrisResponse(BaseModel):
    class_id: int
    species: str
    confidence: float


class BatchRequest(BaseModel):
    samples: List[IrisData] = Field(min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: List[IrisResponse]
    total_predictions: int


@app.get("/", tags=["Health"])
def health_check():
    """Check whether the API and trained model are ready."""
    try:
        load_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "version": app.version,
        }
    except (FileNotFoundError, OSError, ValueError):
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable.",
        )


@app.post("/predict", response_model=IrisResponse, tags=["Predictions"])
def predict_iris(iris: IrisData):
    """Classify a single Iris flower."""
    features = [[
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width,
    ]]

    try:
        result = predict_data(features)[0]
        return IrisResponse(**result)
    except (FileNotFoundError, OSError):
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable.",
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate prediction.",
        )


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["Predictions"],
)
def predict_batch(request: BatchRequest):
    """Classify between 1 and 100 Iris flowers in one request."""
    features = [
        [
            sample.sepal_length,
            sample.sepal_width,
            sample.petal_length,
            sample.petal_width,
        ]
        for sample in request.samples
    ]

    try:
        predictions = predict_data(features)
        return BatchResponse(
            predictions=[IrisResponse(**item) for item in predictions],
            total_predictions=len(predictions),
        )
    except (FileNotFoundError, OSError):
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable.",
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate batch predictions.",
        )