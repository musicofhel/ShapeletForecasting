#!/bin/bash

# Financial Wavelet Forecasting Dashboard - Unified Startup Script
# This script provides a single command to start the entire system

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Financial Wavelet Forecasting Dashboard${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check Python installation
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    print_info "Python 3 found: $(python3 --version)"
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_info "Installing dependencies..."
    pip install -r requirements.txt
    
    print_info "Dependencies installed successfully"
}

# Check if Docker is available
check_docker() {
    if command -v docker &> /dev/null; then
        print_info "Docker is available: $(docker --version)"
        return 0
    else
        print_warning "Docker not found - production mode unavailable"
        return 1
    fi
}

# Start development mode
start_dev() {
    print_info "Starting development mode..."
    source venv/bin/activate
    python app.py --mode dev --port 8050
}

# Start demo mode
start_demo() {
    print_info "Starting demo mode with sample data..."
    source venv/bin/activate
    python app.py --mode demo --port 8050
}

# Start production mode
start_prod() {
    if ! check_docker; then
        print_error "Docker is required for production mode"
        exit 1
    fi
    
    print_info "Starting production mode with Docker..."
    
    # Check if .env.dashboard exists
    if [ ! -f ".env.dashboard" ]; then
        print_warning ".env.dashboard not found. Creating from example..."
        cp .env.dashboard.example .env.dashboard
        print_info "Created .env.dashboard. Please edit it with your configuration."
    fi
    
    # Create necessary directories
    mkdir -p logs/dashboard
    mkdir -p data/processed
    mkdir -p models/saved_pattern_predictors
    
    # Start services
    docker-compose up -d
    
    print_info "Production services started!"
    print_info "Dashboard: http://localhost:8050"
    print_info "API: http://localhost:8000"
}

# Stop all services
stop_all() {
    print_info "Stopping all services..."
    
    # Stop Docker services
    if command -v docker-compose &> /dev/null; then
        docker-compose down 2>/dev/null || true
    fi
    
    # Kill any running Python processes
    pkill -f "python.*app.py" 2>/dev/null || true
    
    print_info "All services stopped"
}

# Show status
show_status() {
    print_info "Checking service status..."
    
    # Check if dashboard is running
    if curl -s http://localhost:8050 > /dev/null 2>&1; then
        print_info "✓ Dashboard is running on http://localhost:8050"
    else
        print_info "✗ Dashboard is not running"
    fi
    
    # Check if API is running
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_info "✓ API is running on http://localhost:8000"
    else
        print_info "✗ API is not running"
    fi
    
    # Check Docker services
    if command -v docker-compose &> /dev/null; then
        docker-compose ps 2>/dev/null || echo "No Docker services running"
    fi
}

# Show logs
show_logs() {
    if [ "$1" = "docker" ]; then
        docker-compose logs -f
    else
        tail -f logs/dashboard/*.log 2>/dev/null || echo "No log files found"
    fi
}

# Clean up
clean_up() {
    print_warning "This will remove all temporary files and caches. Continue? (y/n)"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning up..."
        
        # Stop services
        stop_all
        
        # Clean Python cache
        find . -type f -name "*.pyc" -delete
        find . -type d -name "__pycache__" -exec rm -rf {} +
        find . -type d -name "*.egg-info" -exec rm -rf {} +
        
        # Clean Docker
        if command -v docker &> /dev/null; then
            docker system prune -f 2>/dev/null || true
        fi
        
        print_info "Cleanup complete"
    fi
}

# Show help
show_help() {
    print_header
    echo "Usage: ./start.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  dev      Start development mode (default)"
    echo "  demo     Start demo mode with sample data"
    echo "  prod     Start production mode with Docker"
    echo "  stop     Stop all running services"
    echo "  status   Show service status"
    echo "  logs     Show application logs"
    echo "  clean    Clean up temporary files"
    echo "  help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./start.sh dev"
    echo "  ./start.sh prod"
    echo "  ./start.sh demo --port 8080"
    echo ""
}

# Main function
main() {
    print_header
    
    case "${1:-dev}" in
        "dev")
            check_python
            check_dependencies
            start_dev
            ;;
        "demo")
            check_python
            check_dependencies
            start_demo
            ;;
        "prod")
            start_prod
            ;;
        "stop")
            stop_all
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "clean")
            clean_up
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
