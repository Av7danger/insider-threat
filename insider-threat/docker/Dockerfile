# Dockerfile for Insider Threat Detection API
#
# Purpose: Containerize the FastAPI inference service for easy deployment
#
# Build: docker build -t insider-threat-api -f docker/Dockerfile .
# Run: docker run -p 8000:8000 -v $(pwd)/models:/app/models insider-threat-api

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Create directories for models if they don't exist
RUN mkdir -p models artifacts

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the API
ENTRYPOINT ["uvicorn", "app.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]

