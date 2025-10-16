import pytest
from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

def test_home_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_endpoint():
    """Test prediction endpoint with valid data"""
    test_data = {
        "RAM": 8.0,
        "ROM": 128.0,
        "Battery_in_mAh": 5000.0,
        "Display_size_cm": 16.5,
        "Rating": 4.5
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "predicted_discounted_price" in response.json()
    assert response.json()["predicted_discounted_price"] > 0

def test_predict_endpoint_invalid_data():
    """Test prediction with invalid data"""
    test_data = {
        "RAM": "invalid",
        "ROM": 128.0,
        "Battery_in_mAh": 5000.0,
        "Display_size_cm": 16.5,
        "Rating": 4.5
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error

def test_model_file_exists():
    """Check if model file exists"""
    # Note: This might fail in CI if model isn't included
    # You may need to use DVC or include the model in repo
    assert os.path.exists("best_model.pkl") or os.path.exists("models/best_model.pkl")