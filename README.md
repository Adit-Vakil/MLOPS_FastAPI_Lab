Iris Classification API using FastAPI

Overview

This project is based on FastAPI Lab 1 from Professor Ramin Mohammadi’s MLOps course. The original lab demonstrates how to train a Decision Tree classifier on the Iris dataset and serve predictions through a FastAPI application.

I used the reference implementation as a starting point and made several changes to the training pipeline and API. I wanted to go beyond returning a class number and build something that was easier to test, use, and maintain.

The main additions are hyperparameter tuning, model evaluation, batch predictions, input validation, and automated tests.

Project Structure

MLOPS_Lab1_FastAPI/
├── model/
│   ├── iris_model.pkl
│   └── evaluation_metrics.json
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   ├── predict.py
│   └── main.py
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_train.py
├── requirements.txt
├── test_results.txt
└── README.md

Dataset and Data Preparation

I used the Iris dataset available through scikit-learn. It contains 150 flower samples belonging to three species: Setosa, Versicolor, and Virginica.

Each sample has four measurements:

* Sepal length
* Sepal width
* Petal length
* Petal width

I modified the original data preparation code to include basic checks for missing or invalid numerical values.

I also changed the train/test split from 70/30 to 75/25 and added stratification. This helps maintain a similar distribution of the three species across both datasets.

The final split contains 112 training samples and 38 testing samples.

I used random_state=42 so that the results can be reproduced.

Model Training

The model is a Decision Tree classifier.

The original lab used a fixed maximum depth of 3. Instead of choosing that value directly, I added GridSearchCV to compare different configurations.

The hyperparameters I tested were:

Parameter	Values
max_depth	2, 3, 4, 5
min_samples_split	2, 4, 6
criterion	gini, entropy

This gave me 24 combinations, which I evaluated using five-fold cross-validation.

I used macro F1-score for model selection because it considers performance across all three classes.

The best configuration was:

{
    "criterion": "gini",
    "max_depth": 3,
    "min_samples_split": 2
}

Interestingly, the selected maximum depth was the same as the original implementation. However, this time the value was selected through cross-validation rather than being fixed beforehand.

Model Performance

I evaluated the trained model on the held-out test set.

Metric	Score
Accuracy	89.47%
Precision (macro)	90.30%
Recall (macro)	89.74%
F1-score (macro)	89.68%

The results are saved in model/evaluation_metrics.json.

The accuracy is reasonable for this small dataset, although there is room for improvement. My main focus for this lab was improving the training workflow and API rather than maximizing accuracy.

FastAPI Implementation

I made several changes to the original API.

1. Model Loading

The original prediction function loaded the model from disk every time a prediction was requested.

I changed this to use lru_cache, so the model is loaded once and reused for subsequent requests.

I also replaced the relative model path with a path based on the location of the Python file. This allows the application to find the model regardless of the directory from which it is started.

2. Single Predictions

The /predict endpoint accepts four flower measurements and returns the predicted class.

I extended the response to include the species name and confidence score.

Example request:

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

Response:

{
  "class_id": 0,
  "species": "setosa",
  "confidence": 1.0
}

The confidence value comes from the Decision Tree’s predict_proba() method. It represents the model’s estimated probability and should not be treated as a guarantee that the prediction is correct.

3. Batch Predictions

I added a /predict/batch endpoint that accepts multiple flower samples in a single request.

For example:

{
  "samples": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
    {
      "sepal_length": 6.0,
      "sepal_width": 2.9,
      "petal_length": 4.5,
      "petal_width": 1.5
    }
  ]
}

Response:

{
  "predictions": [
    {
      "class_id": 0,
      "species": "setosa",
      "confidence": 1.0
    },
    {
      "class_id": 1,
      "species": "versicolor",
      "confidence": 1.0
    }
  ],
  "total_predictions": 2
}

I limited batch requests to 100 samples to prevent unexpectedly large requests.

4. Input Validation

I used Pydantic’s Field constraints to validate the measurements before passing them to the model.

The API rejects negative measurements, missing fields, values outside the configured limits, and invalid batch sizes.

For example, sending a negative sepal length returns HTTP 422 with an explanation of the validation error.

I added these checks because the original implementation mainly relied on converting incoming values to floats.

5. Health Check

The / endpoint checks whether the trained model can be loaded.

Example response:

{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0"
}

If the model is unavailable, the endpoint returns HTTP 503 instead of reporting that the application is healthy.

Running the Project

1. Create a Virtual Environment

I used Python 3.13.5 for this project.

From the project directory:

python3 -m venv venv

Activate the environment:

source venv/bin/activate

2. Install Dependencies

python -m pip install -r requirements.txt

The main packages are FastAPI, scikit-learn, Uvicorn, pytest, and HTTPX.

3. Train the Model

Run the following command from the project root:

python -m src.train

This trains the classifier, selects the best hyperparameters, evaluates the model, and saves the results.

The generated files are:

model/iris_model.pkl
model/evaluation_metrics.json

4. Start the API

python -m uvicorn src.main:app --reload

Once the server starts, the interactive API documentation is available at:

http://127.0.0.1:8000/docs

I used the Swagger UI to inspect the endpoints and also tested requests directly using curl.

The --reload option is intended for local development.

Testing

I added automated tests using pytest.

The tests are divided into three files:

File	What it tests	Tests passed
test_api.py	API endpoints, predictions, and validation	11
test_data.py	Dataset checks, splitting, and reproducibility	10
test_train.py	Model training, evaluation, and saved files	10
Total		31

To run all tests:

python -m pytest tests/ -v

All 31 tests passed during my final test run.

I also saved the output in test_results.txt.

The test run produced two deprecation warnings from third-party dependencies, but there were no test failures.

One thing I paid particular attention to was testing the functionality I added myself. For example, the tests check that negative measurements are rejected, batch-size limits are enforced, and the train/test split is reproducible.

Changes from the Original Lab

Here is a summary of the main changes I made.

Component	Original Lab	My Changes
Data splitting	Standard 70/30 split	Stratified 75/25 split
Model training	Fixed hyperparameters	GridSearchCV with five-fold CV
Evaluation	No saved evaluation report	Accuracy, precision, recall, and F1
Model loading	Loaded for every prediction	Cached model loading
Prediction output	Class number	Class ID, species, and confidence
API endpoints	Health and single prediction	Added batch prediction
Validation	Basic Pydantic model	Measurement constraints and batch limits
Testing	No tests in the reference files I used	31 automated tests

The most useful change for me was adding the testing workflow. It helped me verify the behavior of individual components instead of relying only on whether the API returned a response.

I also learned how to structure the project so that training, prediction logic, API routes, and tests remain separate.

Limitations

The Iris dataset is relatively small, so this project does not represent the complexity of a production ML application.

The confidence scores are also not calibrated, meaning a confidence of 1.0 does not necessarily imply that the model is correct 100% of the time.

The automated tests cover the main functionality, but further testing would be needed for production use, particularly around concurrent requests, model failures, and larger workloads.

If I extend this project, I would like to add Docker support, GitHub Actions for automated testing, and model versioning.

Reference

This project is based on FastAPI Lab 1 from Professor Ramin Mohammadi’s MLOps course.

I followed the original lab’s objective of training a Decision Tree classifier and exposing it through FastAPI, while extending the implementation with additional training, prediction, validation, and testing functionality.