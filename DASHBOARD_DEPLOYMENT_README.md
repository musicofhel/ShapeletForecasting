# Financial Wavelet Dashboard - Production Deployment

## Quick Start

The dashboard has been prepared for production deployment with Docker. Here's how to get started:

### 1. Quick Deployment

```bash
# Make deployment script executable (if not already)
chmod +x deploy_dashboard.sh

# Run the deployment
./deploy_dashboard.sh
```

This script will:
- Check Docker and Docker Compose installation
- Create necessary directories
- Generate secure keys automatically
- Build and start all services
- Verify service health

### 2. Manual Deployment

If you prefer manual deployment:

```bash
# Copy and configure environment
cp .env.dashboard.example .env.dashboard
# Edit .env.dashboard with your settings

# Build and start services
docker-compose up -d

# Check status
docker-compose ps
```

## Files Created

1. **Dockerfile.dashboard** - Multi-stage Docker build for the dashboard
2. **docker-compose.yml** - Updated with dashboard service configuration
3. **.env.dashboard** - Environment configuration (sensitive - not in git)
4. **.env.dashboard.example** - Example configuration template
5. **nginx/nginx.conf** - Reverse proxy configuration with:
   - Load balancing
   - SSL/TLS support (certificates required)
   - Rate limiting
   - WebSocket support
   - Security headers
6. **docs/dashboard_deployment.md** - Comprehensive deployment guide
7. **deploy_dashboard.sh** - Automated deployment script
8. **.gitignore** - Updated to exclude sensitive files

## Access Points

After deployment, the services are available at:

- **Dashboard**: http://localhost (via Nginx)
- **Direct Dashboard**: http://localhost:8050
- **API**: http://localhost/api
- **API Docs**: http://localhost/api/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Architecture

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

## Key Features

### Security
- Non-root user in containers
- Environment-based configuration
- SSL/TLS ready (certificates required)
- Security headers configured
- Rate limiting enabled

### Performance
- Redis caching
- Multi-worker support
- Gzip compression
- Static file caching
- WebSocket support for real-time updates

### Monitoring
- Health checks for all services
- Prometheus metrics
- Grafana dashboards
- Structured JSON logging

### Data Persistence
- Models mounted as read-only
- Separate volumes for cache and logs
- Data directory mounted for analysis

## Management Commands

```bash
# View logs
./deploy_dashboard.sh logs dashboard

# Stop services
./deploy_dashboard.sh stop

# Restart services
./deploy_dashboard.sh restart

# Check status
./deploy_dashboard.sh status

# Clean up (removes volumes)
./deploy_dashboard.sh clean
```

## SSL/TLS Setup

For production HTTPS:

1. Obtain SSL certificates (see nginx/ssl/README.md)
2. Place certificates in nginx/ssl/
3. Uncomment SSL sections in nginx/nginx.conf
4. Update server_name in nginx configuration
5. Restart nginx service

## Troubleshooting

If services don't start:

1. Check logs: `docker-compose logs dashboard`
2. Verify ports are available: `netstat -tlnp | grep -E '80|8050|8000'`
3. Ensure environment variables are set correctly
4. Check Docker resources: `docker system df`

## Next Steps

1. **Configure SSL**: Set up HTTPS for production
2. **Set API Keys**: Add external service API keys if needed
3. **Configure Backups**: Set up automated backups
4. **Monitor Performance**: Use Grafana dashboards
5. **Scale as Needed**: Adjust replicas and resources

For detailed information, see `docs/dashboard_deployment.md`.
