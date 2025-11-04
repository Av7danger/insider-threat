#!/bin/bash
# API Run Script
#
# Purpose: Start the FastAPI inference service with recommended settings
#
# Usage:
#   bash scripts/run_api.sh
#   Or on Linux/Mac: chmod +x scripts/run_api.sh && ./scripts/run_api.sh

echo "Starting Insider Threat Detection API..."
echo ""
echo "API will be available at: http://localhost:8000"
echo "Health check: http://localhost:8000/health"
echo "API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "Warning: models/ directory not found. Train models first."
fi

# Start uvicorn server
# --reload: auto-reload on code changes (development only)
# --host 0.0.0.0: listen on all interfaces
# --port 8000: use port 8000
uvicorn app.inference_api:app --host 0.0.0.0 --port 8000 --reload

