#!/bin/bash

# Financial Wavelet Dashboard Deployment Script
# This script helps deploy the dashboard with Docker Compose

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_info "Docker is installed: $(docker --version)"
}

# Check if Docker Compose is installed
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_info "Docker Compose is installed: $(docker-compose --version)"
}

# Check if .env.dashboard exists
check_env_file() {
    if [ ! -f ".env.dashboard" ]; then
        print_warning ".env.dashboard not found. Creating from example..."
        cp .env.dashboard.example .env.dashboard
        print_info "Created .env.dashboard. Please edit it with your configuration."
        print_info "Generating secure keys..."
        
        # Generate secure keys
        SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || echo "please-change-this-secret-key")
        API_KEY=$(openssl rand -hex 16 2>/dev/null || echo "please-change-this-api-key")
        
        # Update the keys in the file (works on both Linux and macOS)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s/SECRET_KEY=your-secret-key-here/SECRET_KEY=$SECRET_KEY/" .env.dashboard
            sed -i '' "s/API_KEY=your-api-key-here/API_KEY=$API_KEY/" .env.dashboard
        else
            # Linux
            sed -i "s/SECRET_KEY=your-secret-key-here/SECRET_KEY=$SECRET_KEY/" .env.dashboard
            sed -i "s/API_KEY=your-api-key-here/API_KEY=$API_KEY/" .env.dashboard
        fi
        
        print_info "Generated secure keys in .env.dashboard"
    else
        print_info ".env.dashboard found"
    fi
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    mkdir -p logs/dashboard
    mkdir -p data/processed
    mkdir -p models/saved_pattern_predictors
    mkdir -p nginx/ssl
    print_info "Directories created"
}

# Build Docker images
build_images() {
    print_info "Building Docker images..."
    docker-compose build --no-cache
    print_info "Docker images built successfully"
}

# Start services
start_services() {
    print_info "Starting services..."
    docker-compose up -d
    print_info "Services started"
}

# Check service health
check_health() {
    print_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check dashboard health
    if curl -f http://localhost:8050/health &> /dev/null; then
        print_info "Dashboard is healthy"
    else
        print_warning "Dashboard health check failed. Check logs with: docker-compose logs dashboard"
    fi
    
    # Check API health
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_info "API is healthy"
    else
        print_warning "API health check failed. Check logs with: docker-compose logs api"
    fi
}

# Show service status
show_status() {
    print_info "Service status:"
    docker-compose ps
}

# Main deployment function
deploy() {
    print_info "Starting Financial Wavelet Dashboard deployment..."
    
    check_docker
    check_docker_compose
    check_env_file
    create_directories
    
    # Ask user if they want to build images
    read -p "Do you want to build Docker images? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_images
    fi
    
    start_services
    check_health
    show_status
    
    print_info "Deployment complete!"
    print_info "Dashboard available at: http://localhost"
    print_info "API available at: http://localhost/api"
    print_info "Direct dashboard access: http://localhost:8050"
    print_info ""
    print_info "Useful commands:"
    print_info "  View logs: docker-compose logs -f dashboard"
    print_info "  Stop services: docker-compose down"
    print_info "  Restart services: docker-compose restart"
}

# Handle command line arguments
case "$1" in
    "start")
        start_services
        ;;
    "stop")
        print_info "Stopping services..."
        docker-compose down
        print_info "Services stopped"
        ;;
    "restart")
        print_info "Restarting services..."
        docker-compose restart
        print_info "Services restarted"
        ;;
    "logs")
        docker-compose logs -f ${2:-}
        ;;
    "status")
        show_status
        ;;
    "build")
        build_images
        ;;
    "clean")
        print_warning "This will remove all containers and volumes. Are you sure? (y/n)"
        read -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v
            print_info "Cleanup complete"
        fi
        ;;
    *)
        deploy
        ;;
esac
