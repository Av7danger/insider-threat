"""
FastAPI Inference Service

Purpose: REST API service for making real-time predictions using trained models.
This allows security systems to integrate insider threat detection into their workflows.

Request validation:
- Pydantic models ensure incoming data matches expected schema
- Automatic error messages for missing/invalid fields
- Type checking prevents common mistakes

Serialization:
- Models are loaded lazily (on first request) to save memory
- Predictions are returned as JSON with probabilities and explanations
- Error responses follow REST API best practices

How this file fits into the project:
- Production endpoint for real-time anomaly detection
- Can be integrated with SIEM systems, alerting pipelines
- Supports both XGBoost and Isolation Forest models

Usage:
    uvicorn app.inference_api:app --host 0.0.0.0 --port 8000
    Or use: bash scripts/run_api.sh
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import joblib
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Insider Threat Detection API",
    description="ML-based anomaly detection for user behavior analysis",
    version="1.0.0"
)

# Global variables for lazy loading
xgb_model = None
xgb_scaler = None
iso_model = None
iso_scaler = None
feature_names = None

class FeatureRequest(BaseModel):
    """
    Request model for prediction endpoint.
    
    This defines the exact schema expected by the API.
    All fields should match the feature columns from features.csv.
    """
    total_events: float = Field(..., description="Total number of events")
    unique_src_ip: float = Field(..., description="Number of unique source IPs")
    unique_dst_ip: float = Field(..., description="Number of unique destination IPs")
    distinct_files: float = Field(..., description="Number of distinct files accessed")
    avg_success: float = Field(..., ge=0.0, le=1.0, description="Average success rate (0-1)")
    start_hour: float = Field(..., ge=0, le=23, description="Hour when activity started (0-23)")
    end_hour: float = Field(..., ge=0, le=23, description="Hour when activity ended (0-23)")
    peak_hour: float = Field(..., ge=0, le=23, description="Peak activity hour (0-23)")
    
    class Config:
        schema_extra = {
            "example": {
                "total_events": 150.0,
                "unique_src_ip": 3.0,
                "unique_dst_ip": 10.0,
                "distinct_files": 45.0,
                "avg_success": 0.95,
                "start_hour": 9.0,
                "end_hour": 17.0,
                "peak_hour": 14.0
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    xgb_prediction: Optional[Dict] = Field(None, description="XGBoost prediction if model available")
    iso_score: Optional[float] = Field(None, description="Isolation Forest anomaly score")
    iso_prediction: Optional[str] = Field(None, description="Isolation Forest prediction (normal/anomaly)")
    shap_explanation: Optional[str] = Field(None, description="SHAP explanation if available")
    message: Optional[str] = Field(None, description="Status message")

def load_models():
    """Lazy load models on first request."""
    global xgb_model, xgb_scaler, iso_model, iso_scaler, feature_names
    
    if xgb_model is None:
        xgb_path = Path('models/xgb_model.pkl')
        xgb_scaler_path = Path('models/xgb_scaler.pkl')
        
        if xgb_path.exists() and xgb_scaler_path.exists():
            try:
                xgb_model = joblib.load(xgb_path)
                xgb_scaler = joblib.load(xgb_scaler_path)
                logger.info("✓ Loaded XGBoost model and scaler")
            except Exception as e:
                logger.warning(f"Could not load XGBoost model: {e}")
    
    if iso_model is None:
        iso_path = Path('models/iso_model.pkl')
        iso_scaler_path = Path('models/iso_scaler.pkl')
        
        if iso_path.exists() and iso_scaler_path.exists():
            try:
                iso_model = joblib.load(iso_path)
                iso_scaler = joblib.load(iso_scaler_path)
                logger.info("✓ Loaded Isolation Forest model and scaler")
            except Exception as e:
                logger.warning(f"Could not load Isolation Forest model: {e}")
    
    # Set feature names (order matters for models)
    if feature_names is None:
        feature_names = [
            'total_events', 'unique_src_ip', 'unique_dst_ip', 'distinct_files',
            'avg_success', 'start_hour', 'end_hour', 'peak_hour'
        ]

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Status of the API
    """
    return {"status": "ok", "service": "insider-threat-detection"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: FeatureRequest):
    """
    Make prediction on user-day features.
    
    This endpoint:
    1. Validates the input (automatically via Pydantic)
    2. Loads models if not already loaded
    3. Scales features using the same scaler used during training
    4. Makes predictions from both models (if available)
    5. Returns results with probabilities and explanations
    
    Args:
        request: FeatureRequest with user-day features
    
    Returns:
        PredictionResponse with predictions from available models
    """
    # Load models if needed
    load_models()
    
    # Convert request to feature array
    features = np.array([[
        request.total_events,
        request.unique_src_ip,
        request.unique_dst_ip,
        request.distinct_files,
        request.avg_success,
        request.start_hour,
        request.end_hour,
        request.peak_hour
    ]])
    
    response = PredictionResponse()
    
    # XGBoost prediction
    if xgb_model is not None and xgb_scaler is not None:
        try:
            features_scaled = xgb_scaler.transform(features)
            prediction = xgb_model.predict(features_scaled)[0]
            probability = xgb_model.predict_proba(features_scaled)[0]
            
            response.xgb_prediction = {
                "prediction": int(prediction),
                "probability_anomaly": float(probability[1]),
                "probability_normal": float(probability[0]),
                "label": "anomaly" if prediction == 1 else "normal"
            }
            response.shap_explanation = "SHAP explanations require batch processing. Use explain_xgb_shap.py script for detailed explanations."
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {e}")
            response.xgb_prediction = {"error": str(e)}
    
    # Isolation Forest prediction
    if iso_model is not None and iso_scaler is not None:
        try:
            features_scaled = iso_scaler.transform(features)
            score = iso_model.score_samples(features_scaled)[0]
            prediction = iso_model.predict(features_scaled)[0]
            
            response.iso_score = float(score)
            response.iso_prediction = "anomaly" if prediction == -1 else "normal"
        except Exception as e:
            logger.error(f"Error in Isolation Forest prediction: {e}")
            response.iso_score = None
            response.iso_prediction = None
    
    if response.xgb_prediction is None and response.iso_score is None:
        raise HTTPException(
            status_code=503,
            detail="No models available. Please train models first."
        )
    
    return response

@app.post("/predict_batch")
def predict_batch(requests: List[FeatureRequest]):
    """
    Make predictions on a batch of user-day features.
    
    This endpoint accepts multiple feature requests and returns predictions for all.
    Useful for processing multiple user-days at once (e.g., from Streamlit demo).
    
    Args:
        requests: List of FeatureRequest objects
    
    Returns:
        List of PredictionResponse objects
    """
    load_models()
    
    responses = []
    
    for request in requests:
        # Convert request to feature array
        features = np.array([[
            request.total_events,
            request.unique_src_ip,
            request.unique_dst_ip,
            request.distinct_files,
            request.avg_success,
            request.start_hour,
            request.end_hour,
            request.peak_hour
        ]])
        
        response = PredictionResponse()
        
        # XGBoost prediction
        if xgb_model is not None and xgb_scaler is not None:
            try:
                features_scaled = xgb_scaler.transform(features)
                prediction = xgb_model.predict(features_scaled)[0]
                probability = xgb_model.predict_proba(features_scaled)[0]
                
                response.xgb_prediction = {
                    "prediction": int(prediction),
                    "probability_anomaly": float(probability[1]),
                    "probability_normal": float(probability[0]),
                    "label": "anomaly" if prediction == 1 else "normal"
                }
            except Exception as e:
                logger.error(f"Error in XGBoost prediction: {e}")
                response.xgb_prediction = {"error": str(e)}
        
        # Isolation Forest prediction
        if iso_model is not None and iso_scaler is not None:
            try:
                features_scaled = iso_scaler.transform(features)
                score = iso_model.score_samples(features_scaled)[0]
                prediction = iso_model.predict(features_scaled)[0]
                
                response.iso_score = float(score)
                response.iso_prediction = "anomaly" if prediction == -1 else "normal"
            except Exception as e:
                logger.error(f"Error in Isolation Forest prediction: {e}")
                response.iso_score = None
                response.iso_prediction = None
        
        responses.append(response)
    
    return responses

@app.get("/models/status")
def models_status():
    """
    Check which models are loaded and available.
    
    Returns:
        dict: Status of each model
    """
    load_models()
    
    return {
        "xgb_loaded": xgb_model is not None,
        "iso_loaded": iso_model is not None,
        "models_path": str(Path('models').absolute())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

