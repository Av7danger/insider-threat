# Docker Deployment Guide

This guide explains how to build and run the Insider Threat Detection API using Docker.

## Quick Start

### Build and Run API

```bash
cd docker
docker-compose up --build
```

The API will be available at: http://localhost:8000

### Build Only

```bash
docker build -t insider-threat-api -f docker/Dockerfile .
```

### Run Container

```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models insider-threat-api
```

## Services

### API Service

- **Port**: 8000
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

### Jupyter Notebook Service (Optional)

To start the notebook service:

```bash
docker-compose --profile notebook up
```

Then access Jupyter at: http://localhost:8888

## Volume Mounts

- `models/`: Mounted for development to allow updating models without rebuilding
- `artifacts/`: For storing evaluation results and plots

## Troubleshooting

### Models Not Found

**Problem**: API returns "No models available"

**Solution**: 
1. Ensure models are trained: `python scripts/train_xgb.py ...`
2. Check that models/ directory contains `.pkl` files
3. Verify volume mount: `docker-compose.yml` should mount `../models:/app/models`

### Port Already in Use

**Problem**: `Error: bind: address already in use`

**Solution**: Change port in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use port 8001 instead
```

### Container Won't Start

**Problem**: Container exits immediately

**Solution**:
1. Check logs: `docker-compose logs api`
2. Verify Python dependencies: `docker-compose exec api pip list`
3. Check file permissions: `ls -la models/`

### Slow Performance

**Problem**: API responses are slow

**Solution**:
1. Increase container resources in `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
   ```
2. Use GPU if available (requires nvidia-docker)

## Production Deployment

For production, consider:

1. **Use specific versions**: Pin Python and package versions
2. **Multi-stage build**: Separate build and runtime stages
3. **Security**: Run as non-root user
4. **Logging**: Configure proper logging and monitoring
5. **HTTPS**: Use reverse proxy (nginx) with SSL certificates
6. **Resource limits**: Set CPU/memory limits

Example production Dockerfile:

```dockerfile
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app/ ./app/
ENV PATH=/root/.local/bin:$PATH
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
ENTRYPOINT ["uvicorn", "app.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

