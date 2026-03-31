# Phase 2 — Flask API

## Objective

Expose the trained XGBoost model via a REST API so predictions can be requested by other applications.

---

## Folder Structure

* **backend/** — contains the Flask API (`app.py`)
* **models/** — contains the trained model (`xgb_model.pkl`)
* **src/** — training scripts (`train.py`)
* **tests/** — test scripts (`api_test.py`)
* **docs/** — documentation files

---

## API Overview

* **Root Route (`/`)**: Returns a message that the API is running.
* **Health Check (`/health`)**: Returns status OK to confirm server is alive.
* **Prediction (`/predict`)**: Accepts POST requests with JSON containing employee features and returns burnout score.

---

## Running the API

1. Navigate to the project root.
2. Start the server using Python: `python backend/app.py`
3. Access the API at `http://127.0.0.1:5000`.

---

## Testing the API

* **Postman** can be used to test endpoints:

  * GET `/` → should return API running message
  * GET `/health` → should return status OK
  * POST `/predict` → provide sample feature data in JSON, get burnout score in response

* **Automated tests** are in `tests/api_test.py` and can be run after the server is up. These check root, health, and prediction endpoints.

---

## Notes

* Ensure the Flask server is running before testing.
* The model path must point to `models/xgb_model.pkl`.
* All responses are JSON.

