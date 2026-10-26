# Real-Time ML Inference REST API & Capstone

**Project:** Iris Species Prediction API  
**Student:** Vishal Gujrati  
**College:** Mahakal Institute of Technology  
**Task:** 06 — Real-Time ML Inference REST API & Capstone

## 1. Overview

This project trains a Random Forest classifier on the built-in Iris dataset and serves predictions through a FastAPI REST API. The `/predict` endpoint accepts JSON flower measurements and returns the predicted species plus a probability for each class.

## 2. Architecture

1. Client sends JSON to `POST /predict`.
2. FastAPI/Pydantic validates the four numeric measurements.
3. The pre-trained `model.joblib` generates class probabilities.
4. The API returns the predicted class and probabilities as JSON.

## 3. Files

- `train_model.py` — trains and saves the model.
- `model.joblib` — trained model and class metadata.
- `main.py` — FastAPI application and endpoints.
- `tests/test_api.py` — API validation and response tests.
- `requirements.txt` — pinned Python dependencies.
- `Dockerfile` — container build instructions.

## 4. Run locally

Use Python 3.11 in a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. Health check: `GET /health`.

## 5. Test the prediction endpoint

Send a `POST` request to `http://127.0.0.1:8000/predict` with this JSON body:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response shape (the exact probabilities are model-dependent):

```json
{
  "prediction": "setosa",
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.013,
    "virginica": 0.007
  }
}
```

## 6. Run unit tests

```bash
pytest -v
```

Tests cover the health endpoint, successful prediction, missing fields, invalid measurements, and probability output.

## 7. Build and run with Docker

Ensure Docker Desktop or Docker Engine is running:

```bash
docker build -t iris-ml-api .
docker run --rm -p 8000:8000 iris-ml-api
```

Then visit `http://localhost:8000/docs`.

## 8. GitHub submission

Create a new GitHub repository named `real-time-ml-inference-api`, then run these commands from this project folder:

```bash
git init
git add .
git commit -m "Complete Task 6 ML inference API"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/real-time-ml-inference-api.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Do not share your password or access token. Submit the repository URL in the training portal after pushing.

## 9. Limitations

This is an educational demonstration using the Iris dataset, not a production service for sensitive or safety-critical decisions. For production deployment, add authentication, rate limiting, monitoring, model versioning, and managed secrets.
