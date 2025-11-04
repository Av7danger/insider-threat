#!/bin/bash
# Streamlit Demo Launcher Script
#
# Purpose: Launch the Streamlit demo app with optional FastAPI backend
#
# Usage:
#   bash scripts/run_demo.sh
#
# On Windows PowerShell (alternative):
#   # Start API in one terminal:
#   bash scripts/run_api.sh
#   # Start Streamlit in another terminal:
#   streamlit run demo_app.py

set -e

# Get project root (parent of scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo ""
echo "========================================"
echo "  Streamlit Demo Launcher"
echo "========================================"
echo ""

# Check if FastAPI should be started
echo "Do you want to start the FastAPI backend? (y/n)"
read -r start_api

if [ "$start_api" = "y" ] || [ "$start_api" = "Y" ]; then
    echo ""
    echo "Starting FastAPI backend in background..."
    echo "API will be available at: http://localhost:8000"
    echo ""
    
    # Start API in background
    bash scripts/run_api.sh &
    API_PID=$!
    
    # Wait a moment for API to start
    sleep 3
    
    echo "FastAPI started (PID: $API_PID)"
    echo "You can stop it later with: kill $API_PID"
    echo ""
fi

# Start Streamlit
echo "Starting Streamlit demo app..."
echo "The app will open in your browser automatically."
echo ""
echo "If it doesn't open, visit: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the demo."
echo ""

streamlit run demo_app.py

# Cleanup: if API was started, kill it when Streamlit exits
if [ -n "$API_PID" ]; then
    echo ""
    echo "Stopping FastAPI backend..."
    kill $API_PID 2>/dev/null || true
fi

echo ""
echo "Demo stopped."

