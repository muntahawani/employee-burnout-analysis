# tests/api_test.py
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_root():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200, "Root route failed"
    assert "message" in response.json(), "No message in root response"
    print("Root route test passed:", response.json())

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200, "Health route failed"
    assert response.json().get("status") == "ok", "Health status not ok"
    print("Health check test passed:", response.json())

def test_predict():
    sample_data = {
        "Resource Allocation": 0.5,
        "Mental Fatigue Score": 0.6,
        "WFH Setup Available": 1,
        "Company Type": 0,
        "Gender": 1,
        "Designation": 2
    }
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    assert response.status_code == 200, "Predict route failed"
    resp_json = response.json()
    assert "burnout_score" in resp_json, "Prediction key missing"
    print("Predict test passed:", resp_json)

if __name__ == "__main__":
    test_root()
    test_health()
    test_predict()
    print("All API tests passed successfully")