# Financial Wavelet Dashboard Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Financial Wavelet Prediction Dashboard in a production environment using Docker and Docker Compose.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Deployment Steps](#deployment-steps)
6. [Security Considerations](#security-considerations)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)
9. [Scaling Guidelines](#scaling-guidelines)

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended) or Windows Server 2019+
- **CPU**: Minimum 4 cores, 8 cores recommended
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: 50GB+ SSD storage
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+

### Software Dependencies

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## Architecture Overview

The dashboard deployment consists of the following services:

```
┌─────────────────┐
│   Nginx (80)    │ ← Load Balancer & Reverse Proxy
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────────┐
│  API  │ │ Dashboard │
│ (8000)│ │  (8050)   │
└───┬───┘ └─────┬─────┘
    │           │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │   Redis   │ ← Cache & Session Store
    │  (6379)   │
    └───────────┘
```

### Service Descriptions

- **Dashboard**: Plotly Dash application serving the web interface
- **API**: FastAPI backend for data processing and model predictions
- **Redis**: In-memory cache for performance optimization
- **Nginx**: Reverse proxy for routing and load balancing
- **Prometheus/Grafana**: Optional monitoring stack

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/financial-wavelet-prediction.git
   cd financial-wavelet-prediction
   ```

2. **Configure environment**
   ```bash
   cp .env.dashboard.example .env.dashboard
   # Edit .env.dashboard with your settings
   ```

3. **Build and start services**
   ```bash
   docker-compose up -d
   ```

4. **Verify deployment**
   ```bash
   # Check service status
   docker-compose ps
   
   # View logs
   docker-compose logs -f dashboard
   ```

5. **Access the dashboard**
   - Dashboard: http://localhost (via Nginx)
   - Direct Dashboard: http://localhost:8050
   - API: http://localhost/api
   - Grafana: http://localhost:3000 (admin/admin)

## Configuration

### Environment Variables

Key environment variables in `.env.dashboard`:

```bash
# Dashboard Settings
DASH_DEBUG=false                    # Set to true for development
DASH_HOST=0.0.0.0                  # Bind address
DASH_PORT=8050                     # Dashboard port
DASH_WORKERS=4                     # Number of worker processes

# Security
SECRET_KEY=your-secret-key-here    # Generate with: openssl rand -hex 32
API_KEY=your-api-key-here          # API authentication key

# Redis
REDIS_URL=redis://redis:6379       # Redis connection string
REDIS_CACHE_TTL=3600              # Cache TTL in seconds

# Performance
MAX_DATA_POINTS=100000            # Maximum data points to process
UPDATE_INTERVAL=5000              # Dashboard update interval (ms)

# External Services (optional)
YAHOO_FINANCE_API_KEY=            # For real-time data
ALPHA_VANTAGE_API_KEY=            # Alternative data source
```

### Data Persistence

Configure volumes in `docker-compose.yml`:

```yaml
volumes:
  # Model storage
  - ./models:/app/models:ro
  
  # Data directory
  - ./data:/app/data:ro
  
  # Logs
  - ./logs/dashboard:/app/logs
  
  # Cache
  - dashboard-cache:/app/cache
```

### Nginx Configuration

The Nginx configuration (`nginx/nginx.conf`) includes:

- **Rate limiting**: Prevents abuse
- **Gzip compression**: Reduces bandwidth
- **Security headers**: XSS protection, clickjacking prevention
- **WebSocket support**: For real-time updates
- **SSL/TLS**: Configuration ready (certificates required)

## Deployment Steps

### 1. Pre-deployment Checklist

- [ ] Generate secure SECRET_KEY
- [ ] Configure API_KEY for authentication
- [ ] Set up SSL certificates (production)
- [ ] Configure firewall rules
- [ ] Set up backup strategy
- [ ] Configure monitoring alerts

### 2. Production Deployment

```bash
# 1. Set production environment
export COMPOSE_FILE=docker-compose.yml
export COMPOSE_PROJECT_NAME=wavelet-dashboard

# 2. Build images
docker-compose build --no-cache

# 3. Run database migrations (if applicable)
# docker-compose run --rm api python -m src.db.migrate

# 4. Start services
docker-compose up -d

# 5. Check health
docker-compose exec dashboard curl -f http://localhost:8050/health
```

### 3. SSL/TLS Setup

For HTTPS support:

1. **Obtain SSL certificates**
   ```bash
   # Using Let's Encrypt
   sudo certbot certonly --standalone -d your-domain.com
   ```

2. **Mount certificates**
   ```yaml
   # In docker-compose.yml
   nginx:
     volumes:
       - /etc/letsencrypt/live/your-domain.com:/etc/nginx/ssl:ro
   ```

3. **Update Nginx config**
   - Uncomment SSL sections in `nginx/nginx.conf`
   - Update server_name to your domain

### 4. Database Setup (Optional)

If using PostgreSQL for data persistence:

```yaml
# Add to docker-compose.yml
postgres:
  image: postgres:14-alpine
  environment:
    POSTGRES_DB: wavelet_db
    POSTGRES_USER: wavelet_user
    POSTGRES_PASSWORD: ${DB_PASSWORD}
  volumes:
    - postgres-data:/var/lib/postgresql/data
```

## Security Considerations

### 1. Network Security

```bash
# Firewall rules (UFW example)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
sudo ufw enable
```

### 2. Container Security

```yaml
# Security options in docker-compose.yml
dashboard:
  security_opt:
    - no-new-privileges:true
  read_only: true
  tmpfs:
    - /tmp
    - /app/cache
```

### 3. Secrets Management

```bash
# Use Docker secrets for sensitive data
echo "your-secret-key" | docker secret create dashboard_secret_key -
```

### 4. Regular Updates

```bash
# Update base images regularly
docker-compose pull
docker-compose up -d
```

## Monitoring and Maintenance

### 1. Health Checks

```bash
# Check service health
curl http://localhost/health

# Dashboard specific health
curl http://localhost:8050/health

# API health
curl http://localhost/api/health
```

### 2. Log Management

```bash
# View logs
docker-compose logs -f dashboard

# Log rotation (add to host)
cat > /etc/logrotate.d/docker-dashboard << EOF
/var/lib/docker/containers/*/*.log {
    rotate 7
    daily
    compress
    size=10M
    missingok
    delaycompress
}
EOF
```

### 3. Backup Strategy

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backup/dashboard"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup data
docker-compose exec -T dashboard tar czf - /app/data > $BACKUP_DIR/data_$DATE.tar.gz

# Backup models
docker-compose exec -T dashboard tar czf - /app/models > $BACKUP_DIR/models_$DATE.tar.gz

# Backup Redis
docker-compose exec -T redis redis-cli SAVE
docker cp redis-cache:/data/dump.rdb $BACKUP_DIR/redis_$DATE.rdb
```

### 4. Monitoring with Prometheus

Metrics available at `/metrics`:

- Request latency
- Active connections
- Cache hit rate
- Model prediction times
- Memory usage

## Troubleshooting

### Common Issues

1. **Dashboard not loading**
   ```bash
   # Check logs
   docker-compose logs dashboard
   
   # Verify port binding
   netstat -tlnp | grep 8050
   ```

2. **High memory usage**
   ```bash
   # Check memory limits
   docker stats
   
   # Adjust in docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 2G
   ```

3. **Slow performance**
   ```bash
   # Check Redis connection
   docker-compose exec dashboard redis-cli ping
   
   # Monitor callback times
   docker-compose logs dashboard | grep "callback_time"
   ```

4. **WebSocket errors**
   ```bash
   # Verify Nginx WebSocket config
   # Check browser console for errors
   # Ensure UPDATE_INTERVAL is reasonable (5000ms+)
   ```

### Debug Mode

For troubleshooting:

```bash
# Enable debug mode
sed -i 's/DASH_DEBUG=false/DASH_DEBUG=true/' .env.dashboard
docker-compose restart dashboard
```

## Scaling Guidelines

### Horizontal Scaling

1. **Multiple Dashboard Instances**
   ```yaml
   dashboard:
     deploy:
       replicas: 3
   ```

2. **Load Balancing**
   - Nginx automatically load balances between instances
   - Use sticky sessions for WebSocket connections

3. **Redis Clustering**
   ```yaml
   redis:
     image: redis:alpine
     command: redis-server --cluster-enabled yes
   ```

### Vertical Scaling

Adjust resources based on load:

```yaml
dashboard:
  deploy:
    resources:
      limits:
        cpus: '4.0'
        memory: 8G
      reservations:
        cpus: '2.0'
        memory: 4G
```

### Performance Optimization

1. **Enable caching**
   ```python
   # In dashboard code
   @cache.memoize(timeout=300)
   def expensive_computation():
       pass
   ```

2. **Optimize callbacks**
   - Use `prevent_initial_call=True`
   - Implement debouncing for frequent updates
   - Cache expensive computations

3. **Data optimization**
   - Implement pagination for large datasets
   - Use data sampling for visualizations
   - Pre-compute aggregations

## Maintenance Schedule

### Daily
- Monitor logs for errors
- Check disk space
- Verify backup completion

### Weekly
- Review performance metrics
- Update dependencies
- Test disaster recovery

### Monthly
- Security updates
- Performance optimization
- Capacity planning

## Support and Resources

- **Documentation**: `/docs`
- **API Reference**: http://localhost/api/docs
- **Issue Tracker**: https://github.com/your-org/repo/issues
- **Community Forum**: https://forum.example.com

## Appendix

### A. Environment Variable Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| DASH_DEBUG | Debug mode | false | No |
| DASH_PORT | Dashboard port | 8050 | No |
| SECRET_KEY | Session encryption | - | Yes |
| API_KEY | API authentication | - | Yes |
| REDIS_URL | Redis connection | redis://redis:6379 | No |

### B. Port Reference

| Service | Internal Port | External Port | Purpose |
|---------|--------------|---------------|---------|
| Nginx | 80 | 80 | HTTP traffic |
| Dashboard | 8050 | - | Dash application |
| API | 8000 | - | REST API |
| Redis | 6379 | - | Cache |
| Prometheus | 9090 | 9090 | Metrics |
| Grafana | 3000 | 3000 | Monitoring |

### C. Useful Commands

```bash
# View all logs
docker-compose logs

# Restart specific service
docker-compose restart dashboard

# Scale service
docker-compose up -d --scale dashboard=3

# Execute command in container
docker-compose exec dashboard python -m src.dashboard.health_check

# Clean up
docker-compose down -v
