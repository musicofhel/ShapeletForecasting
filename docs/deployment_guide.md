# Financial Wavelet Prediction API - Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Production Deployment](#production-deployment)
6. [API Documentation](#api-documentation)
7. [Monitoring & Logging](#monitoring--logging)
8. [Troubleshooting](#troubleshooting)

## Overview

The Financial Wavelet Prediction API provides REST endpoints for financial time series prediction using advanced wavelet analysis and machine learning models.

### Key Features
- Single and batch prediction endpoints
- Real-time streaming predictions
- Model management and updates
- Comprehensive monitoring and metrics
- Docker containerization
- Horizontal scaling support

## Prerequisites

### System Requirements
- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- 8GB+ RAM recommended
- 2+ CPU cores recommended

### Required Files
- Trained model files in `models/` directory:
  - `xgboost_optimized.pkl`
  - `lstm_optimized.pth` (with `lstm_optimized_config.json`)
  - `transformer_optimized.pth` (with `transformer_optimized_config.json`)
  - `ensemble_optimized.pkl`
  - `feature_names.json`

## Local Development

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API Locally

```bash
# Set environment variables (optional)
export MODEL_DIR=./models
export LOG_LEVEL=INFO

# Run with uvicorn
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python -m src.api.app
```

### 3. Test the API

```bash
# Run all tests
python test_api.py

# Run specific test
python test_api.py --test health

# Performance test
python test_api.py --test performance --requests 1000
```

## Docker Deployment

### 1. Build Docker Image

```bash
# Build the image
docker build -t financial-wavelet-api:latest .

# Verify the image
docker images | grep financial-wavelet-api
```

### 2. Run with Docker

```bash
# Run container
docker run -d \
  --name financial-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/logs:/app/logs \
  -e MODEL_DIR=/app/models \
  financial-wavelet-api:latest

# Check logs
docker logs -f financial-api

# Stop container
docker stop financial-api
```

### 3. Docker Compose Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Production Deployment

### 1. Environment Configuration

Create `.env` file for production:

```env
# API Configuration
MODEL_DIR=/app/models
LOG_LEVEL=INFO
WORKERS=4

# Security
API_KEY=your-secure-api-key
ALLOWED_ORIGINS=https://yourdomain.com

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# Redis Cache (optional)
REDIS_URL=redis://redis:6379/0
CACHE_TTL=300
```

### 2. Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financial-wavelet-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: financial-wavelet-api
  template:
    metadata:
      labels:
        app: financial-wavelet-api
    spec:
      containers:
      - name: api
        image: financial-wavelet-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_DIR
          value: "/app/models"
        - name: WORKERS
          value: "4"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: financial-wavelet-api-service
spec:
  selector:
    app: financial-wavelet-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy to Kubernetes:

```bash
kubectl apply -f k8s-deployment.yaml
kubectl get pods
kubectl get services
```

### 3. Cloud Deployment

#### AWS ECS

1. Push image to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [your-ecr-uri]
docker tag financial-wavelet-api:latest [your-ecr-uri]/financial-wavelet-api:latest
docker push [your-ecr-uri]/financial-wavelet-api:latest
```

2. Create ECS task definition and service

#### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/[PROJECT-ID]/financial-wavelet-api

# Deploy to Cloud Run
gcloud run deploy financial-wavelet-api \
  --image gcr.io/[PROJECT-ID]/financial-wavelet-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

#### Azure Container Instances

```bash
# Create container instance
az container create \
  --resource-group myResourceGroup \
  --name financial-wavelet-api \
  --image financial-wavelet-api:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables MODEL_DIR=/app/models
```

## API Documentation

### Interactive Documentation

Once the API is running, access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, -0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "model_type": "ensemble"
  }'
```

#### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      [0.1, 0.2, -0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
      [0.2, 0.3, -0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    ],
    "model_type": "ensemble",
    "return_confidence": true
  }'
```

#### Streaming Prediction
```bash
curl -X POST http://localhost:8000/predict/stream \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, -0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "window_size": 10,
    "model_type": "ensemble"
  }'
```

#### Get Metrics
```bash
curl http://localhost:8000/metrics
```

## Monitoring & Logging

### 1. Prometheus Metrics

Configure Prometheus to scrape metrics:

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'financial-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

### 2. Grafana Dashboard

Import the provided dashboard or create custom visualizations for:
- Request rate and latency
- Model usage distribution
- Error rates
- System resource usage

### 3. Log Aggregation

For production, consider using:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Splunk
- CloudWatch Logs (AWS)
- Stackdriver Logging (GCP)

### 4. Alerts

Configure alerts for:
- High error rate (>10%)
- High response time (P95 > 1s)
- High CPU/Memory usage (>80%)
- Model prediction anomalies

## Troubleshooting

### Common Issues

#### 1. Models Not Loading
```
Error: Model xgboost not loaded
```
**Solution**: Ensure model files exist in the MODEL_DIR directory with correct names.

#### 2. High Memory Usage
**Solution**: 
- Reduce worker count
- Enable model quantization
- Use model-specific endpoints instead of loading all models

#### 3. Slow Response Times
**Solution**:
- Enable Redis caching
- Use batch predictions for multiple samples
- Scale horizontally with more replicas

#### 4. Connection Refused
```
Error: Connection refused at localhost:8000
```
**Solution**: 
- Check if the service is running
- Verify port mapping in Docker
- Check firewall rules

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
uvicorn src.api.app:app --log-level debug
```

### Performance Tuning

1. **Worker Configuration**:
   ```bash
   # Optimal workers = (2 x CPU cores) + 1
   uvicorn src.api.app:app --workers 9
   ```

2. **Connection Pooling**:
   - Configure keep-alive settings
   - Use connection pooling for Redis

3. **Model Optimization**:
   - Use quantized models
   - Implement model caching
   - Load models on-demand

## Security Considerations

1. **API Authentication**:
   - Implement API key authentication
   - Use OAuth2 for user authentication
   - Enable CORS restrictions

2. **HTTPS**:
   - Use SSL/TLS certificates
   - Configure nginx for SSL termination

3. **Rate Limiting**:
   - Implement request rate limiting
   - Use Redis for distributed rate limiting

4. **Input Validation**:
   - Validate all input data
   - Implement request size limits
   - Sanitize error messages

## Maintenance

### Model Updates

1. Prepare new model files
2. Test locally with new models
3. Update model version in metadata
4. Deploy using blue-green deployment
5. Monitor performance metrics

### Backup Strategy

- Regular model file backups
- Database backups (if using)
- Configuration backups
- Log retention policy

### Scaling Guidelines

- **Vertical Scaling**: Increase CPU/Memory for single instance
- **Horizontal Scaling**: Add more replicas behind load balancer
- **Auto-scaling**: Configure based on CPU/Memory or request rate

## Support

For issues or questions:
1. Check the logs: `docker logs financial-api`
2. Review metrics: `http://localhost:8000/metrics`
3. Run diagnostics: `python test_api.py`
4. Check model compatibility
5. Verify environment variables

---

Last Updated: January 2025
