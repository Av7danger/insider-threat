"""
Unit Tests for API Validation

Purpose: Test that FastAPI endpoints validate input correctly and return proper errors.

How to run:
    pytest tests/test_api_validation.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.inference_api import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_valid_request():
    """Test prediction with valid request."""
    valid_request = {
        "total_events": 150.0,
        "unique_src_ip": 3.0,
        "unique_dst_ip": 10.0,
        "distinct_files": 45.0,
        "avg_success": 0.95,
        "start_hour": 9.0,
        "end_hour": 17.0,
        "peak_hour": 14.0
    }
    
    response = client.post("/predict", json=valid_request)
    # Should return 200 or 503 (if models not available)
    assert response.status_code in [200, 503]

def test_predict_missing_field():
    """Test prediction with missing required field."""
    invalid_request = {
        "total_events": 150.0,
        # Missing other required fields
    }
    
    response = client.post("/predict", json=invalid_request)
    assert response.status_code == 422  # Validation error

def test_predict_invalid_range():
    """Test prediction with out-of-range values."""
    invalid_request = {
        "total_events": 150.0,
        "unique_src_ip": 3.0,
        "unique_dst_ip": 10.0,
        "distinct_files": 45.0,
        "avg_success": 1.5,  # Invalid: should be 0-1
        "start_hour": 9.0,
        "end_hour": 17.0,
        "peak_hour": 25.0  # Invalid: should be 0-23
    }
    
    response = client.post("/predict", json=invalid_request)
    assert response.status_code == 422

def test_models_status():
    """Test models status endpoint."""
    response = client.get("/models/status")
    assert response.status_code == 200
    assert "xgb_loaded" in response.json()
    assert "iso_loaded" in response.json()

